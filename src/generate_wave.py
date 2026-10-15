import math
from typing import Callable, NotRequired, TypedDict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from src import image_size


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


def generate_wave(
    resolution_px_per_km: float,
    resolution_seconds_per_time_step: float,
    timestep: int,
    angle,
    wavelength,
    amplitude,
    phase,
    period,
    alpha: float | None = None,
    img_shape: tuple[int, int] = image_size):
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

    return wave * (alpha if alpha else 1)

class WaveParams(TypedDict):
    """
    Class for Parameters needed to create a wave
    """
    resolution_px_per_km: int
    resolution_seconds_per_time_step: int
    timestep: float
    angle: float
    wavelength: float
    amplitude: float
    phase: float
    period: float
    img_shape: tuple[int, int]
    alpha: NotRequired[float]


class WaveGenerator:
    """
    Class for parameter generator functions to create Wave object
    """

    def __init__(self, resolution_px_per_km: Callable[[], int],
                 resolution_seconds_per_time_step: Callable[[], int],
                 timestep: Callable[[], float],
                 angle: Callable[[], float],
                 wavelength: Callable[[], float],
                 amplitude: Callable[[], float],
                 phase: Callable[[], float],
                 period: Callable[[], float],
                 img_shape: Callable[[], tuple[int, int]],
                 alpha: Callable[[], float] | None = None):

        self.resolution_px_per_km: Callable[[], int] = resolution_px_per_km
        self.resolution_seconds_per_time_step: Callable[[], int] = resolution_seconds_per_time_step
        self.timestep: Callable[[], float] = timestep
        self.angle: Callable[[], float] = angle
        self.wavelength: Callable[[], float] = wavelength
        self.amplitude: Callable[[], float] = amplitude
        self.phase: Callable[[], float] = phase
        self.period: Callable[[], float] = period
        self.alpha: Callable[[], float] | None = alpha
        self.img_shape: Callable[[], tuple[int, int]] = img_shape

    def create(self) -> WaveParams:
        return WaveParams(**{k: v() for k, v in self.__dict__.items() if v is not None})

    def generate_wave(self) -> tuple[WaveParams, Tensor]:
        return (params := self.create()), generate_wave(**params) # type: ignore


class WaveCollection:
    def __init__(self, wave_generator: WaveGenerator, num_waves: int | Callable[[], int]):
        self.wave_generator = wave_generator
        self.num_waves = num_waves

    def create_wave_collection(self, use_tqdm: bool = False) -> list[tuple[WaveParams, Tensor]]:
        show = lambda x: tqdm(x) if use_tqdm else x
        return [self.wave_generator.generate_wave() for _ in show(range(self.num_waves() if callable(self.num_waves) else self.num_waves))]

    def create(self, use_tqdm: bool = False) -> tuple[tuple[WaveParams, ...], Tensor]:
        params, waves = zip(*self.create_wave_collection(use_tqdm=use_tqdm))
        wave: Tensor = torch.add(*waves)
        # TODO: maybe remove normalization
        # wave = (wave - (mx := torch.max(wave))) / (mx - torch.min(wave)) # type: ignore
        # TODO: backpropagate the wave normalization onto the parameters
        # assert torch.max(wave) <= 1
        # assert torch.min(wave) >= 0
        return params, wave


@DeprecationWarning
def combine_waves(waves: list[WaveParams]):
    """
    Combine multiple waves into a single wave
    """
    return sum(wave.pop('alpha') * generate_wave({k: v() for k, v in wave.items()}) for wave in waves) # type: ignore


def wave_to_img(wave: Tensor) -> Image.Image:
    img_wave = wave.cpu().numpy()
    img_wave = (img_wave - np.min(img_wave)) / (np.max(img_wave) - np.min(img_wave)) * 255
    return Image.fromarray(img_wave.astype(np.uint8))

def save(wave: Tensor, path: str) -> None:
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
