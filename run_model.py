#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### June 3, 2021          ###
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
from utils import NormalizeDataset, error_metric, iterate_grid_search, fit_and_predict



def load_data(fn):
	""" Load the dataset from file into a set of arrays """

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


def run_algorithm(method, raw_data, hyperparams=None):
	""" Given a method (MLR, Ada, SVR, ANN) with optional hyperparameters
		along with input and output data, evaluate the method """

	# Print out conditions for this run
	if hyperparams is not None:
		for k in hyperparams.keys():
			print('> {} = {}'.format(k, hyperparams[k]))

	# Record results across folds
	records = {
		'ytest'	: [],
		'ptest'	: [],
		'ytrain' : [],
		'ptrain' : []
	}

	# Iterate over folds (WQI is used as a dummy index source)
	kfold = KFold(par['n_folds'], shuffle=True, random_state=1)
	for i, (train_ind, test_ind) in enumerate(kfold.split(np.array(raw_data['WQI']))):

		# Print status to screen
		print('{} running fold {} of {}'.format(method, i+1, par['n_folds']), end='\r')

		# Divide the data into training and testing sets, normalizing the input
		# and output training data individually plus normalizing the intput
		# testing data to the input training data
		ND = NormalizeDataset(raw_data, train_ind, test_ind)
		
		# Obtain dataset layout for this fold
		# xtrain = input training data
		# ytrain = output training data (fitted)
		# xtest  = input testing data
		# ytest  = output testing data (prediction comparison)
		xtrain, ytrain, xtest, ytest = ND.designate_data()

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
		# ptrain = training data prediction (for baseline)
		# ptest  = testing data prediction (for results)
		ptrain, ptest = fit_and_predict(reg, xtrain, ytrain, xtest)

		# Un-normalize the WQI data for the training and testing data,
		# for meaningful error metrics
		ytest, ptest, ytrain, ptrain = ND.unnormalize_WQI(ytest, ptest, ytrain, ptrain)

		# Record this fold's results
		records['ytest'].append(ytest)
		records['ptest'].append(ptest)
		records['ytrain'].append(ytrain)
		records['ptrain'].append(ptrain)

	# Combine all recorded data for analysis
	ytest  = np.concatenate(records['ytest'], axis=0)
	ptest  = np.concatenate(records['ptest'], axis=0)
	ytrain = np.concatenate(records['ytrain'], axis=0)
	ptrain = np.concatenate(records['ptrain'], axis=0)

	# Obtain a suite of error metrics and record
	train_error = error_metric(ytrain, ptrain)
	test_error = error_metric(ytest, ptest)

	# Summarize the recorded error metrics for this method	
	metrics = {}	
	for key in ['mae', 'rmse', 'r2']:
		a = train_error[key]
		b = test_error[key]
		print('{:>4} | Training = {:>6.3f} | Testing = {:>6.3f}'.format(key, a, b))

		metrics[key+'_training'] = a
		metrics[key+'_testing'] = b

	print('\n\n')

	return metrics


##########################################################################


# Load the dataset
raw_data = load_data('./set_03-test.csv')

# Run each algorithm in sequence
methods = ['MLR', 'ADA', 'ANN']#, 'SVR']
methods = ['SVR']
for method in methods:

	# Open a file to save the results
	with open('./output/{}_poly-{}fold-results.csv'.format(method, par['n_folds']), 'w') as f:

		f.write('Method:,{}\n'.format(method))
		f.write('Iteration,MAE Training,RMSE Training,R2 Training,MAE Testing,RMSE Testing,R2 Testing,,Hyperparameters')

		for [i, i0], hyperparams in iterate_grid_search(method):
			print('{} | Grid index {} of {}'.format(method, i, i0))
			results = run_algorithm(method, raw_data, hyperparams)

			f.write('\n' + str(i))
			keys = ['mae_training', 'rmse_training', 'r2_training',\
				'mae_testing', 'rmse_testing', 'r2_testing']
			for key in keys:
				f.write(',{:.5f}'.format(results[key]))

			f.write(',,' + ' | '.join(['{}={}'.format(k, v) for k, v in hyperparams.items()]))

print('\n\nAll models complete!')