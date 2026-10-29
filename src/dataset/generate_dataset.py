"""
Wave Dataset Generation
======================

This module generates a wave image dataset
"""

import os
import pickle
from collections.abc import Callable
from copy import copy
from typing import Literal, overload

import numpy as np
import src.parameters as params
from src import image_size
from src.dataset.generate_wave import (
    WaveCollection,
    WaveGenerator,
    WaveParams,
    save,
)
from torch import Tensor
from tqdm import tqdm

# create single wave generator
single_wave_generator = WaveGenerator(
    resolution_px_per_km = lambda : 2,
    resolution_seconds_per_time_step = lambda : 30,
    angle = lambda : np.random.uniform(0, 360), # 180 would be sufficient
    wavelength = lambda : np.random.uniform(5, 30),
    amplitude = lambda : np.random.uniform(2, 10),
    phase = lambda : np.random.uniform(0, 2 * np.pi),
    period = lambda : np.random.uniform(5, 25),
    time_steps = 7,
    img_shape = image_size
)

# create general wave generator with uniformly distributed alpha valuse
wave_generator = copy(single_wave_generator) # single copy is enough as functions are allowed to be referenced
wave_generator.alpha = lambda : np.random.uniform(0.2, 0.8)



@overload
def generate_wave_dataset(
    num_waves: int | Callable[[], int],
    num_samples: int,
    interval_save: int | None,
    match_range: None,
    folder_name: None,
    scale: None = None,
    return_dataset: Literal[True] = True
) -> list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]:
    """Do not save data, just return the dataset"""

@overload
def generate_wave_dataset(
    num_waves: int | Callable[[], int],
    num_samples: int,
    interval_save: int,
    match_range: None,
    folder_name: str,
    scale: Literal['normalize'] = 'normalize',
    return_dataset: bool = False
) -> list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]:
    """Save the dataset in batches and scale the images using normalize scale."""

@overload
def generate_wave_dataset(
    num_waves: int | Callable[[], int],
    num_samples: int,
    interval_save: int,
    match_range: tuple[float, float],
    folder_name: str,
    scale: Literal['cut'] = 'cut',
    return_dataset: bool = False
) -> list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]:
    """Save the dataset in batches and scale the images using cut scale."""

def generate_wave_dataset(
    num_waves: int | Callable[[], int],
    num_samples: int,
    interval_save: int | None = None,
    match_range: tuple[float, float] | None = None,
    folder_name: str | None = None,
    scale: Literal['normalize', 'cut'] | None = None,
    return_dataset: bool = True
) -> list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]:
    """
    Generate a wave image dataset

    Args:
        num_waves (int | Callable[[], int]): The number of waves to generate; choose int for a fixed number of waves or input a func to generate the amount of waves for each image indepenently.
        num_samples (int): The number of samples to generate
        interval_save (int | None): The interval at which to save the dataset; if None, the dataset will not be saved. Defaults to None.
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
    if callable(num_waves) or num_waves != 1:
        wave_collection = WaveCollection(wave_generator, num_waves)
        if interval_save is not None:
            if folder_name is None or scale not in ['normalize', 'cut'] or (scale == 'cut' and match_range is None):
                raise ValueError('Please match one of the overloads for successful execution')
            amount, extra = divmod(num_samples, interval_save)
            counts = [interval_save for _ in range(amount)] + ([extra] if extra > 0 else [])
            dataset = []
            for idx, count in tqdm(enumerate(counts), total=len(counts), desc='Create Batches'):
                batch = [wave_collection.create() for _ in range(count)]
                save_batch(data=batch, folder_name=folder_name, scale=scale, match_range=match_range, start_index=sum(counts[:idx]), use_tqdm=False) # type: ignore
                if return_dataset:
                    dataset.extend(batch)
                del batch # free RAM
        else:
            dataset = [wave_collection.create() for _ in tqdm(range(num_samples))]
    else:
        if interval_save is not None:
            raise ValueError("Interval save is not implemented for num_waves=1, to make it work set num_waves=lambda : 1")
        dataset = WaveCollection(single_wave_generator, num_samples).create_wave_collection(use_tqdm=True)
    return dataset


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
    for index, d in show(enumerate(data)):
        idx = start_index + index
        path = os.path.join(params.dataset_base_path, f'{folder_name}/{idx}')
        os.mkdir(path)
        save(wave_series=d[1], path=os.path.join(params.dataset_base_path, f'{folder_name}/{idx}'), scale=scale, match_range=match_range) # type: ignore
        with open(os.path.join(path, 'tensors.pkl'), 'wb') as f:
            pickle.dump(d, f) # save whole d as it contains params and the unnormalized wave tensor

if __name__ == "__main__":
    os.mkdir(os.path.join(params.dataset_base_path, 'wave_dataset'))
    # Generate a wave image dataset
    data = generate_wave_dataset(
        num_waves = lambda : np.random.randint(1, 5), # randint is uniformly distributed, too
        num_samples=100_000,
        interval_save=10_000,
        match_range=None,
        folder_name='wave_dataset',
        scale='normalize',
        return_dataset = False
)
    # save_dataset(data, 'wave_dataset', scale='normalize', match_range=None)
