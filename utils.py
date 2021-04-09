#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### April 9, 2021         ###
#############################

import numpy as np
import sklearn.metrics as metrics
from itertools import product
from parameters import par


##########################################################################

class NormWQI:

	def normalize_WQI(self, x):
		self.xmin = x.min()
		self.xmax = x.max()
		return 0.6 * (x-self.xmin) / (self.xmax-self.xmin) + 0.2

	def unnormalize_WQI(self, y):
		return ((y - 0.2) / 0.6) * (self.xmax-self.xmin) + self.xmin

nWQI = NormWQI()

##########################################################################

def normalize_data(x):
	""" Regularize the given data """
	return 0.6 * (x-x.min()) / (x.max()-x.min()) + 0.2


def normalize_dataset(raw_data):
	""" Regularize the dataset for each data type """

	norm_data = {}
	for key in par['data_keys']:
		if key == 'WQI':
			norm_data[key] = nWQI.normalize_WQI(raw_data[key])
		else:
			norm_data[key] = normalize_data(raw_data[key])
	return norm_data


def unnormalize_WQI(wqi):
	""" Inverse the regularization for WQI """
	return nWQI.unnormalize_WQI(wqi)


def build_input_data(data):
	""" Convert the input data to an array """

	input_data = []
	for key in par['input_keys']:

		# Remove any fields not to be used for training and predictions
		if key in par['exclude_keys']:
			pass

		input_data.append(data[key])

	input_data = np.stack(input_data, axis=1)

	return input_data


def error_metric(x, y):
	""" Compute three different error metrics for the given data """

	mae = metrics.mean_absolute_error(x, y)
	rmse = np.sqrt(metrics.mean_squared_error(x, y))
	r_squared = metrics.r2_score(x, y)

	error = {
		'mae'  : mae,
		'rmse' : rmse,
		'r2'   : r_squared
	}

	return error

##########################################################################

def iterate_grid_search(hyperparams):
	""" Given a dictionary of lists of hyperparameters, yield dictionaries of
		hyperparameters for one iteration of a grid search """

	hp_names   = hyperparams.keys()
	hp_lists   = [hyperparams[k] for k in hyperparams.keys()]
	grid_total = np.product([len(hyperparams[k]) for k in hyperparams.keys()])

	for i, args in enumerate(product(*hp_lists)):
		hp_dict = {name : hp for name, hp in zip(hp_names, args)}
		yield '{} of {}'.format(i+1, grid_total), hp_dict



def fit_and_predict(reg, xtrain, ytrain, xtest):

	# Fit the provided model
	reg.fit(xtrain, ytrain)

	# Obtain model predictions
	ptrain = reg.predict(xtrain)
	ptest  = reg.predict(xtest)

	return ptrain, ptest