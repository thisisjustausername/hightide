"""
Wave Dataset Generation
=======================

This module generates a wave image dataset
"""

import os
import pickle
from typing import Literal, overload

import numpy as np
from torch import Tensor
from tqdm import tqdm

import src.parameters as params
from datasets import Array3D, Dataset, Features, Value, disable_progress_bars
from src import image_size
from src.dataset.generate_wave import (
    SuperWave,
    WaveGenerator,
    WaveParams,
    save,
)

disable_progress_bars()

def save_batch_parquet(batch: list[SuperWave], path: str) -> None:
    """
    Save a batch of SuperWave objects to disk as a dataset in parquet file format.

    Args:
        batch (list[SuperWave]): A list of SuperWave objects to save.
        path (str): The path where the dataset will be saved.
    Returns:
        None
    """
    scalars = {
        bool: Value("bool"),
        int: Value("int64"),
        float: Value("float32"),
        str: Value("string"),
    }

    features = Features(
        {
            'wave_img': Array3D(dtype='float32', shape=batch[0].wave_img.shape),
            'params': [{
                'resolution_px_per_km': scalars[int],
                'resolution_seconds_per_time_step': scalars[int],
                'angle': scalars[float],
                'wavelength': scalars[float],
                'amplitude': scalars[float],
                'phase': scalars[float],
                'period': scalars[float],
                'alpha': scalars[float],
            }],
            'image_shape': [Value(dtype='string')],
            'time_steps': Value(dtype='int64'),
        }
    )

    def gen():
        for w in batch:
            yield {
                'wave_img': w.wave_img,
                'params': w.clean_params,
                'image_shape': w.image_shape,
                'time_steps': w.time_steps,
            }

    ds = Dataset.from_generator(gen, features=features)

    ds.save_to_disk(path)
    # ds.to_parquet(path, compression = None)


@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None,
    match_range: None,
    folder_name: None,
    scale: None = None,
    return_dataset: Literal[True] = True,
) -> list[SuperWave] | list[list[SuperWave]]:
    """Do not save data, just return the dataset"""

@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int,
    match_range: None,
    folder_name: str,
    scale: Literal['normalize'] = 'normalize',
    return_dataset: bool = False,
) -> None:
    """Save the dataset in batches and scale the images using normalize scale."""

@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int,
    match_range: tuple[float, float],
    folder_name: str,
    scale: Literal['cut'] = 'cut',
    return_dataset: bool = False,
) -> None:
    """Save the dataset in batches and scale the images using cut scale."""

def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None = None,
    match_range: tuple[float, float] | None = None,
    folder_name: str | None = None,
    scale: Literal['normalize', 'cut'] | None = None,
    return_dataset: bool = True,
) -> list[SuperWave] | list[list[SuperWave]] | None:
    """
    Generate a wave image dataset

    Args:
        wave_generator (WaveGenerator): The wave generator to use for generating the dataset
        num_samples (int): The number of samples to generate
        batch_size (int | None): The interval at which to save the dataset; if None, the dataset will not be saved. If not None, num_samples mod batch_size MUST be 0. Defaults to None.
        match_range (tuple[float, float] | None): How to scale the wave when converting to Image. Only needed when interval_save is not None. Defaults to None.
        folder_name (str | None): The name of the folder to save the dataset. Only needed when interval_save is not None.
        scale (Literal['normalize', 'cut'] | None): How to scale the wave when converting to Image. Only needed when interval_save is not None. Defaults to None.
        return_dataset (bool): Whether to return the dataset. Setting it to False can save RAM for big datasets. Has to be True for interval_save = False. Defaults to True.

    Raises:
        ValueError: If a combination of parameters is specified, that does not match any of the overload combinations.
        ValueError: If num_waves = 1 and interval_save is not None, then num_waves should be set to lambda : 1 for the same result.
    Returns:
        list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]: A list of tuples each containing a tuple of WaveParams objects if num_samples is not set to 1 otherwise a single WaveParams object and a Tensor that is the wave
    """

    if batch_size is not None and num_samples % batch_size != 0:
        raise ValueError("num_samples mod batch_size must be 0 so that each batch has the same amount of samples.")

    avoid_batches = batch_size is None or batch_size == 0
    batch_size: int = (batch_size or num_samples) # if batch_size is None or 0, set it to num_samples so that it generates a single batch
    batches = []
    for _ in tqdm(range(batch_count := (num_samples // batch_size)), desc='Generating Dataset', total=batch_count):
        batch = [wave_generator.generate_super_wave() for _ in range(batch_size)]
        if folder_name is not None:
            save_batch_parquet(batch, os.path.join(params.dataset_base_path, f'{folder_name}/batch_{len(batches)}'))
        if return_dataset:
            batches.append(batch)

    if not return_dataset:
        return None
    if avoid_batches:
        return batches[0]


@overload
def save_dataset(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['normalize'],
    match_range: None
) -> None:
    """On save scale the wave series to the range of (0, 255) with min -> 0, max -> 255."""

@overload
def save_dataset(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['cut'],
    match_range: tuple[float, float]) -> None:
    """On save cut of the wave series to the image. match_range[0] -> 0, match_range[1] -> 255."""

@DeprecationWarning
def save_dataset(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['normalize', 'cut'],
    match_range: tuple[float, float] | None) -> None:
    """
    Save the generated wave dataset to a folder.

    Args:
        data (list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]): The generated wave dataset.
        folder_name (str): The name of the folder to save the dataset in.
    Raises:
        ValueError: If the folder already exists.
    Returns:
        None
    """
    os.mkdir(os.path.join(params.dataset_base_path, folder_name))
    print(type(data[0][0]))
    raise NotImplementedError("This function is not fully implemented yet as it is missing a clean way for saving params.pkl.")
    with open(os.path.join(params.dataset_base_path, f'{folder_name}/tensors.pkl'), 'wb') as f:
        pickle.dump(data, f)
    for i in tqdm(range(len(data))):
        os.mkdir(os.path.join(params.dataset_base_path, f'{folder_name}/{i}'))
        save(wave_series=data[i][1], path=os.path.join(params.dataset_base_path, f'{folder_name}/{i}'), scale=scale, match_range=match_range) # type: ignore




@overload
def save_batch(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['normalize'],
    match_range: None,
    start_index: int,
    use_tqdm: bool = False) -> None:
    """Scale images using normalize."""

@overload
def save_batch(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['cut'],
    match_range: tuple[float, float],
    start_index: int,
    use_tqdm: bool = False) -> None:
    """Scales images using cut."""

@DeprecationWarning
def save_batch(
    data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]],
    folder_name: str,
    scale: Literal['normalize', 'cut'],
    match_range: tuple[float, float] | None,
    start_index: int,
    use_tqdm: bool = False) -> None:
    """
    Saves a batch of the dataset to a folder.

    Args:
        data: A list of tuples containing wave parameters and wave series.
        folder_name: The folder name where the dataset will be saved.
        scale: The scaling method to apply to the wave series.
        match_range: The range of the wave series to match.
        use_tqdm: Whether to use tqdm for progress bars.

    Returns:
        None
    """
    show = lambda x : tqdm(x) if use_tqdm else x
    for index, (d_params, d) in show(enumerate(data)):
        idx = start_index + index
        path = os.path.join(params.dataset_base_path, f'{folder_name}/{idx}')
        os.mkdir(path)
        save(wave_params=d_params, wave_series=d, path=os.path.join(params.dataset_base_path, f'{folder_name}/{idx}'), scale=scale, match_range=match_range) # type: ignore
        with open(os.path.join(path, 'tensors.pkl'), 'wb') as f:
            pickle.dump(d, f) # save whole d as it contains params and the unnormalized wave tensor
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(d_params, f) # save params that generated the wave series

if __name__ == "__main__":
    os.mkdir(os.path.join(params.dataset_base_path, 'wave_dataset_2'))

    wave_generator = WaveGenerator(
        resolution_px_per_km = lambda : 2,
        resolution_seconds_per_time_step = lambda : 30,
        angle = lambda : np.random.uniform(0, 360), # 180 would be sufficient
        wavelength = lambda : np.random.uniform(5, 30),
        amplitude = lambda : np.random.uniform(2, 10),
        phase = lambda : np.random.uniform(0, 2 * np.pi),
        period = lambda : np.random.uniform(5, 25),
        alpha = lambda : np.random.uniform(0.2, 0.8),
        num_waves = lambda : np.random.randint(1, 5), # randint is uniformly distributed, too
        time_steps = 7,
        img_shape = image_size,
        to_cpu = True
    )

    # Generate a wave image dataset
    data = generate_wave_dataset(
        wave_generator=wave_generator,
        num_samples=100_000,
        batch_size=10_000,
        match_range=None,
        folder_name='wave_dataset_2',
        scale='normalize',
        return_dataset=False,
)
    # save_dataset(data, 'wave_dataset_2', scale='normalize', match_range=None)
