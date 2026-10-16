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
python3 -m src.generate_dataset
```
