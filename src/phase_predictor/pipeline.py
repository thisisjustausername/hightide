# %% Load imports

import json
import os
import pickle

import src.parameters as params
from src.dataset.generate_wave import WaveParams, generate_wave

# %% Read first timeframe datapoint
i = 878
with open(os.path.join(params.dataset_base_path, 'wave_dataset', str(i), 'tensors.pkl'), 'rb') as f:
    tensor = pickle.load(f)
with open(os.path.join(params.dataset_base_path, 'wave_dataset', str(i), 'params.pkl'), 'rb') as f:
    params = pickle.load(f)
params = tuple(WaveParams(**p) for p in params)
# %% Get Parameters
print(f"Parameters: {json.dumps(params, indent=2)}")
