"""
Wave Dataset Generator
======================

This module generates a wave image dataset
"""

import os
import pickle
from collections.abc import Callable
from copy import copy

import numpy as np
from torch import Tensor
from tqdm import tqdm

from src import image_size
from src.generate_wave import (
    WaveCollection,
    WaveGenerator,
    WaveParams,
    save,
)

single_wave_generator = WaveGenerator(
    resolution_px_per_km = lambda : 2,
    resolution_seconds_per_time_step = lambda : 30,
    timestep = lambda : 0,
    angle = lambda : np.random.uniform(0, 360), # 180 would be sufficient
    wavelength = lambda : np.random.uniform(5, 30),
    amplitude = lambda : np.random.uniform(2, 10),
    phase = lambda : np.random.uniform(0, 2 * np.pi),
    period = lambda : np.random.uniform(5, 25),
    img_shape = lambda : image_size
)

double_wave_generator = copy(single_wave_generator) # single copy is enough as functions are allowed to be referenced
double_wave_generator.alpha = lambda : np.random.uniform(0.2, 0.8)


def generate_wave_dataset(num_waves: int | Callable[[], int], num_samples: int) -> list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]:
    """
    Generate a wave image dataset

    Args:
        num_waves (int | Callable[[], int]): The number of waves to generate; choose int for a fixed number of waves or input a func to generate the amount of waves for each image indepenently.
        num_samples (int): The number of samples to generate
    Returns:
        list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]: A list of tuples each containing a tuple of WaveParams objects if num_samples is not set to 1 otherwise a single WaveParams object and a Tensor that is the wave
    """
    if callable(num_waves) or num_waves != 0:
        wave_collection = WaveCollection(double_wave_generator, num_waves)
        dataset = [wave_collection.create() for _ in tqdm(range(num_samples))]
    else:
        dataset = WaveCollection(single_wave_generator, num_samples).create_wave_collection(use_tqdm=True)
    return dataset


def save_dataset(data: list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]], folder_name: str) -> None:
    os.mkdir(f'datasets/{folder_name}')
    with open(f'datasets/{folder_name}/tensors.pkl', 'wb') as f:
        pickle.dump(data, f)
    for i in range(len(data)):
        save(data[i][1], f'datasets/{folder_name}/{i}.png')

if __name__ == "__main__":
    # Generate a wave image dataset
    data = generate_wave_dataset(lambda : np.random.randint(1, 5), 100) # randint is uniformly distributed, too
    # for i in data[:15]:
    #     assert torch.max(i[1]) <= 1
    #     assert torch.min(i[1]) >= 0
    # save_dataset(data, 'wave_dataset')
