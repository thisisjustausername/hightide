"""
Wave phase Predictor module.
============================

Model phase for wave with given parameters and unknown parameter phase.
"""

import os
import pickle
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Parameter

import src.parameters as params
from src.dataset.generate_wave import SuperWave, WaveParams


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

    @DeprecationWarning
    def difference(self, phases: Tensor) -> Tensor:
        """
        Compute the difference as image between the target wave and the generated wave with the given phase.

        Arg:
            phases (Tensor): The phase parameters to evaluate the difference.
        Returns:
            Tensor: The computed difference as image.
        """
        return self.wave.wave_img - self.generate_super_wave(phases).wave_img

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

    def fit(self, steps: int = 200) -> float:
        """
        Fit the model to the target wave by optimizing the phase parameter.

        Args:
            steps (int): Number of optimization steps to perform.
        Returns:
            float: The optimized phase value after fitting.
        """
        self.losses = []
        for i in range(steps):
            loss = self.backpropagate()
            self.losses.append(loss.item())
            # TODO: compute early_stopping cheaper maybe
            if self.early_stopping is not None and len(self.losses) >= self.early_stopping and min(self.losses[-self.early_stopping:]) > min(self.losses):
                    print(f'Early stopping at step {i} with loss {loss.item():.6e}')
                    break
            if i % 10 == 0:
                print(f'Step {i:<5} | Loss = {loss.item():<10.6e} | Phases = {self.phases.detach().numpy().tolist()}')
        return self.phases.detach().numpy()

    def predict(self) -> SuperWave:
        """
        Predict the wave using the optimized phase parameter.

        Returns:
            SuperWave: The predicted wave as a SuperWave object.
        """
        return self.generate_super_wave(self.phases.detach())


if __name__ == '__main__':
    euclidean_chord_distance = lambda x_pred, x_true : ((torch.cos(x_pred) - torch.cos(x_true))**2 + (torch.sin(x_pred) - torch.sin(x_true))**2).mean()
    geodesic_angular_loss = lambda x_pred, x_true : (torch.atan2(torch.sin(x_pred - x_true), torch.cos(x_pred - x_true))**2).mean()
    mse = lambda pred, true: ((pred - true) ** 2).mean()
    # fft_phase_loss =

    loss = mse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    item = 878

    with open(os.path.join(params.dataset_base_path, 'wave_dataset', str(item), 'params.pkl'), 'rb') as f:
        parameters = pickle.load(f)

    super_wave = SuperWave(
        wave_params=tuple(WaveParams(**p) for p in parameters),
        time_steps=7,
        image_shape=(256, 256)
    )


    wave_predictor = WavePredictor(
        original_wave=super_wave,
        loss_func=loss,
        initial_phases=torch.full((len(super_wave.params),), 0.0, dtype=torch.float64),
        learning_rate=10e-3,
        early_stopping=None
    )

    wave_predictor.fit(steps=2000)

    print(f'Difference in wave phases: {wave_predictor.phases.detach().numpy() - wave_predictor.original_phases.numpy()}')

    import matplotlib.pyplot as plt

    plt.plot(wave_predictor.losses)
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss.png', dpi=200, bbox_inches='tight')
    plt.show()
