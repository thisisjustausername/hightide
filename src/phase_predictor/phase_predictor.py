"""
Wave phase Predictor module.
============================

Model phase for wave with given parameters and unknown parameter phase.
"""
import json
import os
import pickle
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import Parameter

import src.parameters as params
from src.dataset.generate_wave import SuperWave, WaveParams, wave_series_to_imgs


class WavePredictor:
    def __init__(
        self,
        original_wave: SuperWave,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        initial_phases: Tensor | None = None,
        learning_rate: float = 10e-5,
        early_stopping: int | None = None,
        # device: torch.device | None = None
    ):
        # self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.wave = original_wave

        self.timesteps = self.wave.wave_img.shape[0]
        self.img_shape: tuple[int, int] = self.wave.wave_img.shape[1:] # type: ignore
        self.wave_without_phase = tuple({k: v for k, v in wave.items() if k != 'phase'} for wave in self.wave.params)
        self.early_stopping = early_stopping
        self.losses = []
        self.loss_func = loss_func
        self.original_phases = torch.tensor([wave.get('phase', torch.nan) for wave in self.wave.params], dtype=torch.float32)


        if initial_phases is not None and len(initial_phases) != len(self.wave.params):
            raise ValueError(f"Length of initial_phases ({len(initial_phases)}) must match the number of waves ({len(self.wave.params)}).")

        initial_phases = initial_phases if initial_phases is not None else torch.zeros(self.timesteps, dtype=torch.float32)
        self.phases = Parameter(initial_phases)
        self.optimizer = torch.optim.Adam([self.phases], lr=learning_rate)


    def generate_super_wave(self, phases: Tensor) -> SuperWave:
        """
        Generate a wave for the specified phase and the other parameters from the original wave.

        Args:
            phases (Tensor): The phases parameter to generate the wave.
        Returns:
            Tensor: The generated wave as a tensor.
        """
        return SuperWave(
            wave_params=tuple(WaveParams(wp | {'phase': p}) for wp, p in zip(self.wave_without_phase, phases)), # type: ignore
            time_steps=self.timesteps,
            image_shape=self.img_shape
        )

    def difference(self) -> Tensor:
        """
        Compute the difference as image between the target wave and the generated wave with the given phase.

        Arg:
            phases (Tensor): The phase parameters to evaluate the difference.
        Returns:
            Tensor: The computed difference as image.
        """
        return self.wave.wave_img - self.generate_super_wave(self.phases).wave_img

    def loss(self, phases: Tensor | Parameter) -> Tensor:
        """
        Compute the loss between the target wave and the generated wave with the given phase.

        Args:
            phases (Tensor | Parameter): The phase parameters to evaluate the loss.
        Returns:
            Tensor: The computed loss value.
        """
        return self.loss_func(self.generate_super_wave(phases).wave_img, self.wave.wave_img)

    def backpropagate(self) -> Tensor:
        """
        Perform a single optimization step to update the phase parameter.

        Returns:
            Tensor: The computed loss value after the optimization step.
        """
        self.optimizer.zero_grad()
        loss = self.loss(self.phases)
        loss.backward()
        self.optimizer.step()
        return loss

    def fit(self, steps: int = 200, verbose: bool = True) -> float:
        """
        Fit the model to the target wave by optimizing the phase parameter.

        Args:
            steps (int): Number of optimization steps to perform.
            verbose (bool): Whether to print information during fitting.
        Returns:
            float: The optimized phase value after fitting.
        """
        self.losses = []
        for i in range(steps):
            loss = self.backpropagate()
            self.losses.append(loss.item())
            # TODO: compute early_stopping cheaper maybe
            if self.early_stopping is not None and len(self.losses) >= self.early_stopping and min(self.losses[-self.early_stopping:]) > min(self.losses):
                    if verbose: print(f'Early stopping at step {i} with loss {loss.item():.6e}')
                    break
            if i % 10 == 0 and verbose:
                print(f'Step {i:<5} | Loss = {loss.item():<10.6e} | Phases = {self.phases.detach().numpy().tolist()}')
        return self.phases.detach().numpy()

    def predict(self) -> SuperWave:
        """
        Predict the wave using the optimized phase parameter.

        Returns:
            SuperWave: The predicted wave as a SuperWave object.
        """
        return self.generate_super_wave(self.phases.detach())


def prediction_pipeline(
    item: int,
    time_steps: int = 7,
    image_shape: tuple[int, int] = (256, 256),
    loss_func: Callable[[Tensor, Tensor], Tensor] | None = None,
    learning_rate: float = 10e-3,
    early_stopping: int = 30,
    steps: int = 1500,
    verbose: bool = True
) -> None:
    """
    Pipeline to predict and save the phases of a wave given its parameters.

    Args:
        item (int): The index of the wave item in the dataset.
        time_steps (int): The number of time steps for the wave.
        image_shape (tuple[int, int]): The shape of the wave images.
        loss_func (Callable[[Tensor, Tensor], Tensor] | None): The loss function to use for optimization. If None, mean squared error is used. Create the function using def and add its code into its docstring for documetation purposes in results.json.
        learning_rate (float): The learning rate for the optimizer.
        early_stopping (int): The number of steps to wait for improvement before stopping early.
        steps (int): The maximum number of optimization steps to perform.
        verbose (bool): Whether to print information during the prediction process.

    """

    if loss_func is None:
        def mse(pred, true):
            '''
            ((pred - true) ** 2).mean()
            '''
            return ((pred - true) ** 2).mean()
        loss_func = mse

    with open(os.path.join(params.dataset_base_path, 'wave_dataset', str(item), 'params.pkl'), 'rb') as f:
        parameters = pickle.load(f)

    super_wave = SuperWave(
        wave_params=tuple(WaveParams(**p) for p in parameters),
        time_steps=time_steps,
        image_shape=image_shape
    )

    initial_phases = torch.full((len(super_wave.params),), 0.0, dtype=torch.float64)

    wave_predictor = WavePredictor(
        original_wave=super_wave,
        loss_func=loss_func,
        initial_phases=initial_phases,
        learning_rate=learning_rate,
        early_stopping=early_stopping
    )

    wave_predictor.fit(steps=steps, verbose=verbose)

    if verbose:
        print(f'Difference in wave phases: {wave_predictor.phases.detach().numpy() - wave_predictor.original_phases.numpy()}')

    base_path = os.path.join(params.dataset_base_path, 'wave_dataset', str(item), 'phase_pred')
    os.makedirs(base_path, exist_ok=True)

    # save the results
    with open(os.path.join(base_path, 'predicted_params.pkl'), 'wb') as f:
        pickle.dump(tuple(wave_predictor.generate_super_wave(wave_predictor.phases).params), f)
    # TODO: add amount of actually done steps (e.g. less with early stopping raised)
    with open(os.path.join(base_path, 'results.json'), 'w') as f:
        json.dump(
            {
                'phases': wave_predictor.phases.detach().numpy().tolist(),
                'original_phases': wave_predictor.original_phases.numpy().tolist(),
                'loss': wave_predictor.loss(wave_predictor.phases).item(), # recompute, since not done after last step of optimizer
                'loss_func_name': wave_predictor.loss_func.__name__ if hasattr(wave_predictor.loss_func, '__name__') else str(wave_predictor.loss_func),
                'loss_func_doc': wd.strip() if hasattr(wave_predictor.loss_func, '__doc__') and (wd := wave_predictor.loss_func.__doc__) is not None else None,
                'initial_phases': initial_phases.detach().numpy().tolist(),
                'time_steps': time_steps,
                'image_shape': image_shape,
                'learning_rate': learning_rate,
                'early_stopping': early_stopping,
                'steps': steps,
                'number_of_waves': len(super_wave.params),
                'losses': wave_predictor.losses,
            }, f, indent=4
        )

    # Save the difference between waves and predicted waves
    diff = wave_predictor.difference()

    imgs = wave_series_to_imgs(diff, scale='normalize', match_range=None)
    os.makedirs(os.path.join(base_path, 'diffs'), exist_ok=True)
    for index, i in enumerate(imgs):
        i.save(os.path.join(base_path, 'diffs', f'{index}.png'))

    # Save the predicted waves
    imgs = wave_series_to_imgs(wave_predictor.generate_super_wave(wave_predictor.phases).wave_img, scale='normalize', match_range=None)
    os.makedirs(os.path.join(base_path, 'predicted'), exist_ok=True)
    for index, i in enumerate(imgs):
        i.save(os.path.join(base_path, 'predicted', f'{index}.png'))

    # Plot the loss over iterations

    plt.plot(wave_predictor.losses)
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(base_path, 'loss.png'), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # euclidean_chord_distance = lambda x_pred, x_true : ((torch.cos(x_pred) - torch.cos(x_true))**2 + (torch.sin(x_pred) - torch.sin(x_true))**2).mean()
    # geodesic_angular_loss = lambda x_pred, x_true : (torch.atan2(torch.sin(x_pred - x_true), torch.cos(x_pred - x_true))**2).mean()
    # mse = lambda pred, true: ((pred - true) ** 2).mean()

    from tqdm import tqdm

    # NOTE: give function a name, so that it can be saved in the results.json file.
    # NOTE: add code into docstring, so that it can be saved in the results.json file.
    def mse(pred, true):
        '''
        ((pred - true) ** 2).mean()
        '''
        return ((pred - true) ** 2).mean()


    for i in tqdm(range(8984)):
        prediction_pipeline(
            item=i,
            time_steps=7,
            image_shape=(256, 256),
            loss_func=mse,
            learning_rate=10e-3,
            early_stopping=30,
            steps=2000,
            verbose=False
        )
