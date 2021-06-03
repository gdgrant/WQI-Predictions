#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### June 3, 2021          ###
#############################

import yaml

par = {}

# Number of folds for K-folds cross-validation
par['n_folds'] = 209 # 5, 10, 209 (209=LOO)

# List out shorter names for data headers as keys
par['input_keys'] = ['pH', 'turb', 'DO', 'CHLA', 'CDOM', 'cond', 'TRPN', 'FC', 'BOD', 'T']
par['data_keys'] = par['input_keys'] + ['WQI']
par['exclude_keys'] = ['FC', 'BOD', 'CHLA', 'turb', 'T']

# Load the hyperparameters to be used for grid searching
with open('./grid_search.yaml', 'r') as f:
	par['grid_search_settings'] = yaml.load(f.read(), Loader=yaml.CLoader)
	