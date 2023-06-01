# python-time-series
Time-Series analysis, statistical and machine learning models for forecasting, regression, and classification

You can install your local environment with conda (recommended) or pip. The environment configurations for conda and pip are provided. Please note that if you choose pip as you installation tool, you might need additional tweaking.

## Conda
```bash
conda env create --file time_series.yml
```

The conda environment is called `time_series`. You can activate it as follows:
```bash
conda activate time_series
```

## Pip
```bash
pip install -r requirements.txt
```

## Contributing

If you find anything amiss with the notebooks or dependencies, please feel free to create a pull request.

If you want to change the conda dependency specification (the yaml file), you can test it like this:
```bash
conda env create --file time_series.yml --force
```

You can update the pip requirements like this:
```bash
pip freeze > requirements.txt
```

Please make sure that you keep these two ways of maintaining dependencies in sync.

Then make sure, you test the notebooks in the new environment to see that they run.
