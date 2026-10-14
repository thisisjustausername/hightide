import math

import numpy as np
import torch
from PIL import Image

from src import device, image_size


def _wavevector(angle_math, wavelength):
    """Calculate wave vector based on polar coordinates (Differentiable).

    Args:
        angle_math ("""
    angle_math = angle_math - 90
    pre = 2 * math.pi / wavelength
    return (pre * math.cos(math.pi / 180 * angle_math), pre * math.sin(math.pi / 180 * angle_math))

def _get_current_phase(phase: float, period: float, timestep: float = 0):
    """Calculate current phase based on given timestep."""
    return phase + (timestep / period * 2 * math.pi)


def generate_wave(resolution_px_per_km: float,
                        resolution_seconds_per_time_step: float,
                        timestep: int, angle, wavelength, amplitude, phase, period, img_shape: tuple[int, int] = image_size):

    # Add device=device to the linspace creations
    y = torch.linspace(0, img_shape[0] - 1, img_shape[0], dtype=torch.float32) # , device=device)
    x = torch.linspace(0, img_shape[1] - 1, img_shape[1], dtype=torch.float32) # , device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    # Get wave vector
    k = _wavevector(angle, wavelength)

    # Calculate current phase
    current_phase = _get_current_phase(phase, period, float(timestep) * resolution_seconds_per_time_step)

    # Create the wave pattern using tensor operations
    wave = amplitude * (torch.cos(k[0] * x_grid / resolution_px_per_km + k[1] * y_grid / resolution_px_per_km - current_phase) -
                        torch.sin(k[0] * x_grid / resolution_px_per_km + k[1] * y_grid / resolution_px_per_km - current_phase))

    return wave

def wave_to_img(wave):
    img_wave = wave.cpu().numpy()
    img_wave = (img_wave - np.min(img_wave)) / (np.max(img_wave) - np.min(img_wave)) * 255
    return Image.fromarray(img_wave.astype(np.uint8))

def save(wave, path):
    image = wave_to_img(wave)
    image.save(path)

if __name__ == '__main__':
    wave = generate_wave(img_shape=(256, 256),
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        timestep=0,
        angle=0,
        wavelength=15,
        amplitude=4,
        phase=0,
        period=10)

    save(wave=wave, path='wave.png')
