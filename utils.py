#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### May 13, 2021          ###
#############################

import numpy as np
import sklearn.metrics as metrics
from itertools import product

from parameters import par


##########################################################################

class NormalizeDataset:

	def __init__(self, raw_data, train_ind, test_ind):

		# Load data into class
		self.raw_data  = raw_data
		self.train_ind = train_ind
		self.test_ind  = test_ind

		# Separate training and testing data
		self.train_raw = {key : x[train_ind] for key, x in self.raw_data.items()}
		self.test_raw = {key : x[test_ind] for key, x in self.raw_data.items()}

		# Make empty arrays to store normalized data and normalization constants
		self.train_norm = {}
		self.test_norm = {}

		self.mins = {}
		self.maxs = {}

		# Iterate over parameters
		for key in par['data_keys']:

			# Pick out the current parameter
			x = self.train_raw[key]
			y = self.test_raw[key]

			# Obtain and save the min and max of the training data
			xmin = x.min()
			xmax = x.max()

			self.mins[key] = xmin
			self.maxs[key] = xmax

			# Normalize the training and testing data from the training
			# data constants
			self.train_norm[key] = 0.6 * (x-xmin) / (xmax-xmin) + 0.2
			self.test_norm[key]  = 0.6 * (y-xmin) / (xmax-xmin) + 0.2

	def designate_data(self):

		# Iterate over the list of possible inputs collect them
		# into training and testing sets
		xtrain = []
		xtest = []
		for key in par['input_keys']:

			# Omit any parameters not to be used for training and predictions
			if key in par['exclude_keys']:
				continue

			# Include appropriate parameters, in order
			xtrain.append(self.train_norm[key])
			xtest.append(self.test_norm[key])

		# Combine data into arrays
		xtrain = np.stack(xtrain, axis=1)
		ytrain = np.array(self.train_norm['WQI'])
		xtest = np.stack(xtest, axis=1)
		ytest = np.array(self.test_norm['WQI'])

		# Return the training and testing data
		return xtrain, ytrain, xtest, ytest

	def unnormalize_WQI(self, ytest, ptest, ytrain, ptrain):

		# For each of the given WQI vectors, unnormalize them in order based on
		# the min and max of the training data WQI
		for x in [ytest, ptest, ytrain, ptrain]:
			yield ((x - 0.2) / 0.6) * (self.maxs['WQI']-self.mins['WQI']) + self.mins['WQI']


##########################################################################


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

def conditional_grid_generator(with_cond_params, no_cond_params, hp_dict, hp_lists, hp_names, grid_total, iterate_only=False):

	# Iterate over the grid search
	i = 0
	for args in product(*hp_lists):

		# Build the starting dictionary with static and non-conditional hyperparameters
		full_hp_dict = {**hp_dict, **{name : hp for name, hp in zip(hp_names, args)}}

		# Build a list of conditions from the grid search file
		cond_names = []
		for param in with_cond_params.keys():

			pkey  = with_cond_params[param]['cond'][0]
			pvals = with_cond_params[param]['cond'][1:]

			if full_hp_dict[pkey] in pvals:
				cond_names.append(param)

		# If there are conditions, iterate through them and add the requisite hyperparameters
		if len(cond_names) > 0:
			cond_lists = [with_cond_params[k]['vals'] for k in cond_names]
			for args in product(*cond_lists):

				full_hp_dict = {**full_hp_dict, **{name : hp for name, hp in zip(cond_names, args)}}

				i += 1
				yield [i, grid_total], full_hp_dict

		# If there are not conditions, skip that step and return the current hyperparameters
		else:
			i += 1
			yield [i, grid_total], full_hp_dict


def iterate_grid_search(model_name):

	# Get the sub-dictionary of hyperparameters for this model
	hyperparams = par['grid_search_settings'][model_name]

	# See if this model has certain unchanging parameters
	if not hyperparams is None and 'static_params' in hyperparams:
		hp_dict = hyperparams.pop('static_params')
	else:
		hp_dict = {}

	# If the grid search for this model is empty,
	# iterate once on an empty hyperparameters dict and return
	if hyperparams is None:
		yield [1, 1], {}
		return

	# Check for hyperparameters that are conditional on other hyperparameters
	with_cond_params = {}
	no_cond_params = {}
	for param in hyperparams:
		
		if 'cond' in hyperparams[param].keys():
			with_cond_params[param] = hyperparams[param]
		else:
			no_cond_params[param] = hyperparams[param]

	# Build a list of names of hyperparameters without conditions, and a list
	# lists of those hyperparameters for iteration
	hp_names   = no_cond_params.keys()
	hp_lists   = [no_cond_params[k]['vals'] for k in no_cond_params.keys()]

	# Iterate through the grid search to know its length
	i_total = 0
	for _ in conditional_grid_generator(with_cond_params, no_cond_params, hp_dict, hp_lists, hp_names, grid_total=0):
		i_total += 1

	# Iterate through the grid search properly
	yield from conditional_grid_generator(with_cond_params, no_cond_params, hp_dict, hp_lists, hp_names, grid_total=i_total)


##########################################################################


def fit_and_predict(reg, xtrain, ytrain, xtest):
	""" Given a regression model, training data, and test data, fit the
		provided model and generate some model predictions """

	# Fit the provided model
	reg.fit(xtrain, ytrain)

	# Obtain model predictions
	ptrain = reg.predict(xtrain)
	ptest  = reg.predict(xtest)

	return ptrain, ptest