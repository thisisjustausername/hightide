"""
Offset parameters phase predictor
=================================

Predict phases with offset parameters (angle, etc.).

Two options:
1. Compare the generation with the predicted phases and the ORIGINAL parameters to the original image.
   Take the original params when generating the image with the predicted phase so that the error of that image is only being caused by the predicted phase and not by any other parameters.
2. Also save image with offset parameters to see whether model is actually closer than with original params.
"""

# %% Import modules
import numpy as np
import torch
from IPython.display import display

from src.dataset.generate_wave import SuperWave, WaveParams, wave_series_to_imgs
from src.phase_predictor.phase_predictor import PhasePredictor


# %% Define example parameters
def mse(pred, true):
    '''
    ((pred - true) ** 2).mean()
    '''
    return ((pred - true) ** 2).mean()

wave_params = (
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=150,
        wavelength=80.0,
        amplitude=3,
        phase=np.pi,
        period=10,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.8
    ),
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=80,
        wavelength=20.0,
        amplitude=7,
        phase=0.5 * np.pi,
        period=4,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.5
    ),
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=20,
        wavelength=60.0,
        amplitude=4,
        phase=5,
        period=8,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.3
    )
)

original_wave = SuperWave(
    wave_params=wave_params,
    time_steps=7,
    image_shape=(256, 256),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    to_cpu=False,
    to_numpy=False
)

# modify wave_params to create approximated wave
wave_params_approx = (
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=160,
        wavelength=75.0,
        amplitude=2.8,
        phase=0,
        period=9,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.8
    ),
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=80,
        wavelength=20.0,
        amplitude=7,
        phase=0,
        period=4,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.5
    ),
    WaveParams(
        resolution_px_per_km=2,
        resolution_seconds_per_time_step=30,
        angle=20,
        wavelength=60.0,
        amplitude=4,
        phase=0,
        period=8,
        img_shape=(256, 256),
        timestep=0,
        alpha=0.3
    )
)

approx_wave = SuperWave(
    wave_params=wave_params_approx,
    time_steps=7,
    image_shape=(256, 256),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    to_cpu=False,
    to_numpy=False
)

# %% Initialize PhasePredictor
wave_predictor = PhasePredictor(
    original_wave=original_wave,
    approximated_wave=approx_wave,
    loss_func=mse,
    initial_phases=None,
    learning_rate=10e-3,
    early_stopping=None,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# %% Fit the model and compute difference
steps = 2000

# predict phases
result = wave_predictor.fit(steps=steps)

# compute difference of fitted wave to original wave
diff = wave_predictor.difference()

# scale image when displaying
max_val, min_val = float(original_wave.wave_img.max()), float(original_wave.wave_img.min())

# compute difference between approximated wave and original wave
unfitted_diff = original_wave.wave_img - approx_wave.wave_img

# %% Convert wave series to images
fitted_imgs = wave_series_to_imgs(
    wave_series=diff,
    scale='cut',
    match_range=(min_val, max_val),
    coloured=True
)

unfitted_imgs = wave_series_to_imgs(
    wave_series=unfitted_diff,
    scale='cut',
    match_range=(min_val, max_val),
    coloured=True
)

# %% Display images
print("Fitted wave difference to original wave:")
for i in fitted_imgs:
    display(i)
print("Unfitted wave difference to original wave:")
for i in unfitted_imgs:
    display(i)

# TODO: add offset params option to PhasePredictor so the image is of the original parameters but the prediciton wave is being generated with the offset parameters.
# TODO: compute entropy on images, entropy of fitted should far lower as more like noise and less like hard waves.
