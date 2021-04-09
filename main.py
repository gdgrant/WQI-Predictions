#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### April 9, 2021         ###
#############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from parameters import par
from wqi import compute_wqi
from utils import normalize_dataset, unnormalize_WQI, build_input_data, error_metric, iterate_grid_search, fit_and_predict



def load_data(fn):

	# List out headers in the file
	raw_keys = ['(SEN) pH ', '(SEN) Turbidity NTU', '(SEN) DO ppm',
	       '(SEN) CHLA RFU', '(SEN) CDOM RFU', '(SEN) Conductivity',
	       '(SEN) TRPN RFU', 'FC cfu/100ml', 'BOD ppm', '(SEN) Temperature']

	# Load the data
	dataset = pd.read_csv(fn, error_bad_lines=False)

	# Select the desired data types
	dataset = dataset[raw_keys]

	# Cull missing or invalid entries
	dataset = dataset.apply(pd.to_numeric, errors='coerce').dropna()

	# Convert dataframe to a dictionary of numpy arrays
	data = {}
	for rk, k in zip(raw_keys, par['input_keys']):
		data[k] = dataset[rk].to_numpy()

	# Compute WQI from existing data
	data['WQI'] = compute_wqi(data['DO'], data['FC'], data['pH'], data['BOD'], data['T'])

	print('\nDataset loaded.\n\n')

	return data



def run_algorithm(method, xdata, ydata, hyperparams=None):

	# Print out conditions for this run
	print('Method: {}'.format(method))
	if hyperparams is not None:
		for k in hyperparams.keys():
			print('> {} = {}'.format(k, hyperparams[k]))

	# Set up a way to record across folds
	train_record = []
	test_record = []

	# Iterate over folds
	kfold = KFold(par['n_folds'], shuffle=True, random_state=1)
	for i, (train_ind, test_ind) in enumerate(kfold.split(xdata)):
		print('{} running fold {} of {}'.format(method, i+1, par['n_folds']), end='\r')

		# Designate training and testing data
		xtrain = xdata[train_ind,:]
		ytrain = ydata[train_ind]
		xtest  = xdata[test_ind,:]
		ytest  = ydata[test_ind]

		# Select a model using the given hyperparameters
		if method == 'MLR':
			reg = LinearRegression()
		elif method == 'ADA':
			reg = AdaBoostRegressor(**hyperparams)
		elif method == 'SVR':
			reg = SVR(**hyperparams)
		elif method == 'ANN':
			reg = MLPRegressor(**hyperparams)
		else:
			raise Exception('{} not implemeted'.format(method))

		# Fit and run the model to obtain predictions
		ptrain, ptest = fit_and_predict(reg, xtrain, ytrain, xtest)

		# Un-regularize the WQI data for meaningful error metrics
		ytest  = unnormalize_WQI(ytest)
		ptest  = unnormalize_WQI(ptest)
		ytrain = unnormalize_WQI(ytrain)
		ptrain = unnormalize_WQI(ptrain)

		# Obtain a siute of error metrics and record
		train_error = error_metric(ytrain, ptrain)
		test_error = error_metric(ytest, ptest)

		train_record.append(train_error)
		test_record.append(test_error)

	# Summarize the recorded error metrics for this method		
	for key in ['mae', 'rmse', 'r2']:
		a = np.mean([train_record[i][key] for i in range(par['n_folds'])])
		b = np.mean([test_record [i][key] for i in range(par['n_folds'])])
		print('{:>4} | Training = {:>6.3f} | Testing = {:>6.3f}'.format(key, a, b))
	print('\n\n')


# Load and preprocess the dataset
raw_data = load_data('./set_03-test.csv')
norm_data = normalize_dataset(raw_data)

# Build the input and output data arrays for use in predictions
input_data = build_input_data(norm_data)
output_data = norm_data['WQI']


# Run each algorithm
run_algorithm('MLR', input_data, output_data)

for index, hyperparams in iterate_grid_search(par['ada_hyper']):
	print('Grid index {}'.format(index))
	run_algorithm('ADA', input_data, output_data, hyperparams)

# for index, hyperparams in iterate_grid_search(par['svr_hyper']):
# 	print('Grid index {}'.format(index))
# 	run_algorithm('SVR', input_data, output_data, hyperparams)

# for index, hyperparams in iterate_grid_search(par['ann_hyper']):
# 	print('Grid index {}'.format(index))
# 	run_algorithm('ANN', input_data, output_data, hyperparams)