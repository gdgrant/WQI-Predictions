# WQI-Predictions

This code implements predictions of water quality based on various water parameters.  Note that the data used for optimizing these models is not included in the repository.

This codebase consists of six files:
- `run_model.py` -- This is the entry point into the code.  Run this file.
- `utils.py` -- This file holds utility functions for processing data and models.
- `wqi.py` -- This file holds a set of functions for evaluating the actual water quality index (WQI) from provided data.
- `parameters.py` -- Any settings for loading or using data, plus importing of the grid search hyperparameters, are kept here.
- `grid_search.yaml` -- This file holds the grid search hyperparamters to be used when running models.
- `analysis.py` -- Once the desired models have been run, this file may be edited and run to select for the best results.
