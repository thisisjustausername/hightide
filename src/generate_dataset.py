"""
Wave Dataset Generator
======================

This module generates a wave image dataset
"""

import os
import pickle
from typing import Literal

import numpy as np
from tqdm import tqdm

from src import image_size
from src.generate_wave import generate_wave, save

dataset_parameters = {
    'resolution_px_per_km': 2,
    'resolution_seconds_per_time_step': 30,
    'timestep': 0,
    'angle': np.random.uniform(0, 360),
    'wavelength': np.random.uniform(5, 30),
    'amplitude': np.random.uniform(2, 10),
    'phase': np.random.uniform(0, 2 * np.pi),
    'period': np.random.uniform(5, 25),
    'img_shape': image_size,
}

def generate_wave_dataset(num_samples: int):
    """
    Generate a wave image dataset
    """
    # TODO: stack data to tensor and do single func call so that the GPU can be properly used
    dataset = [generate_wave(**dataset_parameters) for _ in tqdm(range(num_samples))]
    return dataset

def save_dataset(data, folder_name: str): # , format: list[Literal['pickle', 'png']]):
    os.mkdir(f'datasets/{folder_name}')
    # if 'pickle' in format:
    with open(f'datasets/{folder_name}/tensors.pkl', 'wb') as f:
        pickle.dump(data, f)
    # if 'png' in format:
    for i in range(len(data)):
        save(data[i], f'datasets/{folder_name}/{i}.png')

if __name__ == "__main__":
    # Generate a wave image dataset
    data = generate_wave_dataset(100_000)
    save_dataset(data, 'wave_dataset')
