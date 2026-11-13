# hightide
Extract wave parameters from images

# Setup
Set up the virtual environment and install the required dependencies.
```bash
uv venv --python 3.13 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

# Generate Dataset
To generate the dataset containing timeseries of overlapping waves, run following command.
```bash
python3 -m src.dataset.generate_dataset
```

# Predict Phases
To predict the phases of the waves in the dataset, run following command.\
For each wave timeseries, made up of timesteps each containing an image of overlapping waves, it predicts the phases of all waves.
```bash
python3 -m src.predict_phases.phase_predictor
```

# TODOs
* Add GPU-Support to `src.predict_phases.phase_predictor` pipeline and specific classes.
