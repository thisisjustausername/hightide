"""
Wave Dataset Generator
======================

This module contains functions to generate overlapping waveforms for different parameters.
"""

import math
import os
from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict, overload

import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from src import image_size


def _wavevector(angle_math: float, wavelength: float) -> tuple[float, float]:
    """
    Calculate wave vector based on polar coordinates (Differentiable).

    Args:
        angle_math (float): Polar coordinates (Differentiable).
        wavelength (float): Polar coordinates (Differentiable).
    Returns:
        tuple[float, float]: Wave vector (Differentiable).
    """
    angle_math = angle_math - 90
    pre = 2 * math.pi / wavelength
    return (pre * math.cos(math.pi / 180 * angle_math), pre * math.sin(math.pi / 180 * angle_math))


def _get_current_phase(phase: float, period: float, timestep: float = 0) -> float:
    """
    Calculate current phase based on given timestep.

    Args:
        phase (float): Current phase (Differentiable).
        period (float): Period (Differentiable).
        timestep (float): Timestep, the step in time this moment represents (Differentiable).
    Returns:
        float: Current phase (Differentiable).
    """
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
    img_shape: tuple[int, int] = image_size,
    device: torch.device | None = None) -> Tensor:
    """
    Generate a wave pattern(s) based on the given parameters.

    Args:
        resolution_px_per_km (float): Number of pixels per kilometer
        resolution_seconds_per_time_step (float): Time step in seconds between timeframes
        timestep (int): Amount of images in the timeseries
        angle (float): Angle in radians
        wavelength (float): Wavelength of the wave
        amplitude (float): Amplitude of the wave
        phase (float): Phase of the wave
        period (int): Period of the wave
        alpha (float): Alpha value for the wave, it sets the opacity (used when overlapping waves)
        img_shape (tuple[int, int]): Size of the tensor (later image, too)
        device (torch.device | None): The device to create the tensor on. If None, the default device is used.
    Returns:
        Tensor: Tensor containing the generated wave pattern(s)
    """

    # Add device=device to the linspace creations
    y = torch.linspace(0, img_shape[0] - 1, img_shape[0], dtype=torch.float32, device=device)
    x = torch.linspace(0, img_shape[1] - 1, img_shape[1], dtype=torch.float32, device=device)
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
    angle: float
    wavelength: float
    amplitude: float
    phase: float
    period: float
    img_shape: NotRequired[tuple[int, int]]
    timestep: NotRequired[int]
    alpha: NotRequired[float]


class WaveGenerator:
    """
    Class for parameter generator functions to create single Wave object with timesteps.

    Attributes:
        resolution_px_per_km (Callable[[], int]): Number of pixels per kilometer
        resolution_seconds_per_time_step (Callable[[], int]): Time step in seconds between timeframes
        time_steps (Callable[[], int]): Amount of images in the timeseries
        angle (Callable[[], float]): Angle in radians
        wavelength (Callable[[], float]): Wavelength of the wave
        amplplitude (Callable[[], float]): Amplitude of the wave
        phase (Callable[[], float]): Phase of the wave
        period (Callable[[], float]): Period of the wave
        alpha (Callable[[], float] | None = None): Alpha value for the wave, it sets the opacity (used when overlapping waves)
        img_shape (Callable[[], tuple[int, int]]): Size of the tensor (later image, too)

    Methods:
        create(): Create a WaveParams object
        generate_wave(): Generate a Wave object with WaveParams and simulation Tensor
    """

    def __init__(self, resolution_px_per_km: Callable[[], int],
                 resolution_seconds_per_time_step: Callable[[], int],
                 angle: Callable[[], float],
                 wavelength: Callable[[], float],
                 amplitude: Callable[[], float],
                 phase: Callable[[], float],
                 period: Callable[[], float],
                 time_steps: int,
                 img_shape: tuple[int, int],
                 alpha: Callable[[], float] | None = None):

        self.resolution_px_per_km: Callable[[], int] = resolution_px_per_km
        self.resolution_seconds_per_time_step: Callable[[], int] = resolution_seconds_per_time_step
        self.angle: Callable[[], float] = angle
        self.wavelength: Callable[[], float] = wavelength
        self.amplitude: Callable[[], float] = amplitude
        self.phase: Callable[[], float] = phase
        self.period: Callable[[], float] = period
        self.alpha: Callable[[], float] | None = alpha
        self.time_steps: int = time_steps
        self.img_shape:tuple[int, int] = img_shape

    def create(self) -> WaveParams:
        """
        Generate an WaveParams instance for the first wave of the time series by executing the functions defined for each parameter.

        Returns:
            WaveParams: A WaveParams instance for the first wave of the time series.
        """
        # starting timestep is always 0
        return WaveParams(**({k: v() if callable(v) else v for k, v in self.__dict__.items() if v is not None and k != 'time_steps'} | {'timestep': 0})) # type: ignore

    def generate_wave(self) -> tuple[WaveParams, Tensor]:
        """
        Generate a WaveParams instance and the matching Tensor from the wave simulation for those parameters.

        Returns:
            tuple[WaveParams, Tensor]: A tuple containing the WaveParams instance and the corresponding Tensor with `time_steps` many frames (wave images).
        """
        return (params := self.create()), torch.stack(
            [
                generate_wave(
                    # timestep overrides timestep from params
                    **(params | {'timestep': timestep})
                ) for timestep in range(self.time_steps)
            ]
        )


class WaveCollection:
    """
    A Generator for (either a Collection of single wave Tensors or) a single Tensor consisting of overlapping wave Tensors.

    Attributes:
        wave_generator (WaveGenerator): The WaveGenerator instance to use for generating wave parameters.
        num_waves (int | Callable[[], int]): The number of wave Tensors to generate. If int then a fixed amount of waves is being generated, otherwise a func() amount of waves is being generated.

    Methods:
        create_wave_collection(ise_tqdm: bool = False): Create a collection of tuples each containing a WaveParams object and a wave Tensor.
        create(use_tqdm: bool = False): Create a wave made up of `num_waves` overlapping waves.
    """
    def __init__(self, wave_generator: WaveGenerator, num_waves: int | Callable[[], int]):
        self.wave_generator = wave_generator
        self.num_waves = num_waves

    def create_wave_collection(self, use_tqdm: bool = False) -> list[tuple[WaveParams, Tensor]]:
        """
        Create a collection of tuples each containing a WaveParams object and a wave Tensor.
        Default use for overlapping the created waves in the collection to a single wave Tensor.

        Args:
            use_tqdm (bool): Whether to show a progress bar during generation. Defaults to False.
        Returns:
            list[tuple[WaveParams, Tensor]]: A list of tuples, each containing a WaveParams object and a wave frames Tensor containing `time_steps` many frames (wave images).
        """
        show = lambda x: tqdm(x) if use_tqdm else x
        return [self.wave_generator.generate_wave() for _ in show(range(self.num_waves() if callable(self.num_waves) else self.num_waves))] # type: ignore

    def create(self, use_tqdm: bool = False) -> tuple[tuple[WaveParams, ...], Tensor]:
        """
        Create a wave from `num_waves` overlapping waves.

        Args:
            use_tqd (bool): Whether to show a progress bar during generation. Defaults to False.
        Returns:
            tuple[tuple[WaveParams, ...], Tensor]: A tuple of WaveParams objects the wave was generated from and the final wave timeframes Tensor containing `time_steps` many frames (wave images).
        """
        params, waves = zip(*self.create_wave_collection(use_tqdm=use_tqdm))
        wave_series: Tensor = torch.stack(waves).sum(dim=0)
        # TODO: maybe remove normalization
        # wave = (wave - (mx := torch.max(wave))) / (mx - torch.min(wave))
        # TODO: backpropagate the wave normalization onto the parameters
        # assert torch.max(wave) <= 1
        # assert torch.min(wave) >= 0
        return params, wave_series


class SuperWave:
    """
    A SuperWave class for wave tensors, that are a timeseries and are made up of overlapping waves.
    """
    def __init__(
        self,
        wave_params: tuple[WaveParams, ...],
        time_steps: int = 7,
        image_shape: tuple[int, int] = (256, 256),
        device: torch.device | None = None
    ):
        """
        Initialize a SuperWave instance.

        Args:
            wave_params (tuple[WaveParams, ...]): A tuple of WaveParams objects to generate the overlapping waves.
            time_steps (int): The number of time steps for the wave series.
            image_shape (tuple[int, int]): The shape of the generated wave images.
            device (torch.device | None): The device to create the tensor on. If None, the default device is used.
        """
        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = tuple(i | {'timestep': 0, 'img_shape': image_shape} for i in wave_params) # set timestep to 0, image_shape to image_shape
        self.wave_img: Tensor = torch.stack(
            [
                torch.stack(
                    [
                        generate_wave(**(params | {'timestep': ts}), device=device) for params in wave_params # single wave at single point in time
                    ]
                    ).sum(dim=0) # wave_stack at single point in time
            for ts in range(time_steps)
            ] # list of wave_stacks each at a single point in time
        ) # stacked waves over time

    def save_to_images(self, folder_path: str):
        """
        Save the wave tensor as a series of images.

        Args:
            folder_path (str): The folder path to save the image series to.
        """
        os.makedirs(folder_path, exist_ok=True)
        for index, image in enumerate(wave_series_to_imgs(self.wave_img, scale='normalize', match_range=None)):
            image.save(os.path.join(folder_path, str(index) + '.png'))


@overload
def wave_series_to_imgs(
    wave_series: Tensor,
    scale: Literal['normalize'],
    match_range: None
) -> list[Image.Image]:
    """Scale the wave to the range of (0, 255) with min -> 0, max -> 255."""

@overload
def wave_series_to_imgs(
    wave_series: Tensor,
    scale: Literal['cut'],
    match_range: tuple[float, float]) -> list[Image.Image]:
    """Cut of the wave to the image. match_range[0] -> 0, match_range[1] -> 255."""

def wave_series_to_imgs(
    wave_series: Tensor,
    scale: Literal['normalize', 'cut'],
    match_range: tuple[float, float] | None) -> list[Image.Image]:
    """
    Convert a wave series to an series of images.


    Args:
        wave_series (Tensor): The wave series tensor to convert to an image.
        scale (Literal['normalize', 'cut')]: The scale to apply. Options are 'normalize' to scale the wave to a range of (0, 255) and 'cut' to scale the wave in the match_range to (0, 255) and cap the values outside to either 0 or 255.
        match_range (tuple[float, float] | None): The range to scale the wave series in the match range. Needs to be supplied if and only if scale is set to 'cut'. Values inside the range are being scaled to (0, 255), values outside are capped to either 0 or 255.
    Raises:
        ValueError: If an unsupported scale is provided or if match_range is provided with scale set to 'normalize' or not provided with scale set to 'cut'.
    Returns:
        list[Image.Image]: The image representation of the wave.
    """
    if scale not in (options := ['normalize', 'cut']):
        raise ValueError(f'Unsupported scale: {scale}. Only {", ".join(options)} are supported')

    img_wave = wave_series.cpu()
    # always normalize over whole time series
    if scale == 'normalize':
        if match_range is not None:
            raise ValueError('Match range cannot be used together with scale = cut.')
        img_wave = (img_wave - torch.min(img_wave)) / (torch.max(img_wave) - torch.min(img_wave)) * 255
    elif scale == 'cut': # use elif for simple expansion of list
        if match_range is None:
            raise ValueError("match_range must be provided for scale 'cut'.")
        img_wave = (img_wave - match_range[0]) / (match_range[1] - match_range[0]) * 255
    int8_wave_series = img_wave.to(torch.uint8)
    split_waves = [i.numpy() for i in torch.unbind(int8_wave_series, dim=0)]
    return [Image.fromarray(wave) for wave in split_waves]


@overload
def save(
    wave_series: Tensor,
    path: str,
    scale: Literal['normalize'],
    match_range: None
) -> None:
    """On save scale the wave series to the range of (0, 255) with min -> 0, max -> 255."""

@overload
def save(
    wave_series: Tensor,
    path: str,
    scale: Literal['cut'],
    match_range: tuple[float, float]) -> None:
    """On save cut of the wave series to the image. match_range[0] -> 0, match_range[1] -> 255."""

def save(
    wave_series: Tensor,
    path: str,
    scale: Literal['normalize', 'cut'],
    match_range: tuple[float, float] | None) -> None:
    """
    Save wave tensor as image.

    Args:
        wave_series (Tensor): The wave series tensor to save as an image.
        path (str): The folder path to save the image series to.
        scale (Literal['normalize', 'cut']): The scale to apply. Options are 'normalize' to scale the wave to a range of (0, 255) and 'cut' to scale the wave in the match_range to (0, 255) and cap the values outside to either 0 or 255.
        match_range (tuple[float, float] | None): The range to scale the wave series in the match range. Needs to be supplied if and only if scale is set to 'cut'. Values inside the range are being scaled to (0, 255), values outside are capped to either 0 or 255.
    Raises:
        ValueError: If an unsupported scale is provided or if match_range is provided with scale set to 'normalize' or not provided with scale set to 'cut'.
    Returns:
        None
    """
    images = wave_series_to_imgs(wave_series, scale, match_range) # type: ignore
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for index, image in enumerate(images):
        image.save(os.path.join(path, str(index) + '.png'))
