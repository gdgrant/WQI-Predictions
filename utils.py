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

class RegularizeWQI:
	""" This class is meant to regularize the WQI data and store the relevant
		numbers for un-regularizing the data after prediction using a regressor """

	def regularize_WQI(self, x):
		self.xmin = x.min()
		self.xmax = x.max()
		return 0.6 * (x-self.xmin) / (self.xmax-self.xmin) + 0.2

	def unregularize_WQI(self, y):
		return ((y - 0.2) / 0.6) * (self.xmax-self.xmin) + self.xmin

# Create a WQI regularizer object
rWQI = RegularizeWQI()


##########################################################################


def regularize_data(x):
	""" Regularize the given data """
	return 0.6 * (x-x.min()) / (x.max()-x.min()) + 0.2


def regularize_dataset(raw_data):
	""" Regularize the dataset for each data type """

	norm_data = {}
	for key in par['data_keys']:
		if key == 'WQI':
			norm_data[key] = rWQI.regularize_WQI(raw_data[key])
		else:
			norm_data[key] = regularize_data(raw_data[key])
	return norm_data


def unregularize_WQI(wqi):
	""" Inverse the regularization for WQI """
	return rWQI.unregularize_WQI(wqi)


def build_input_data(data):
	""" Convert the input data to an array """

	input_data = []
	for key in par['input_keys']:

		# Omit any fields not to be used for training and predictions
		if key in par['exclude_keys']:
			continue

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

def conditional_grid_generator(with_cond_params, no_cond_params, \
			hp_dict, hp_lists, hp_names, grid_total, iterate_only=False):

	i = 0
	for args in product(*hp_lists):

		full_hp_dict = {**hp_dict, **{name : hp for name, hp in zip(hp_names, args)}}

		cond_names = []
		for param in with_cond_params.keys():

			pkey  = with_cond_params[param]['cond'][0]
			pvals = with_cond_params[param]['cond'][1:]

			if full_hp_dict[pkey] in pvals:
				cond_names.append(param)

		if len(cond_names) > 0:
			cond_lists = [with_cond_params[k]['vals'] for k in cond_names]
			for args in product(*cond_lists):

				full_hp_dict = {**full_hp_dict, **{name : hp for name, hp in zip(cond_names, args)}}

				i += 1
				yield [i, grid_total], full_hp_dict
		else:
			i += 1
			yield [i, grid_total], full_hp_dict


def iterate_grid_search(model_name):

	# Get the sub-dictionary of hyperparameters for this model
	hyperparams = par['grid_search_settings'][model_name]
	
	# If the grid search for this model is empty,
	# iterate once on an empty hyperparameters dict and return
	if hyperparams is None:
		yield [1, 1], {}
		return

	# See if this model has certain unchanging parameters
	if 'static_params' in hyperparams:
		hp_dict = hyperparams.pop('static_params')
	else:
		hp_dict = {}

	# If the grid search for this model is empty after getting
	# any unchanging parameters, iterate once on the base dict and return
	if hyperparams is None:
		yield [1, 1], hp_dict
		return

	# Check if the current model requires any conditional hyperparams
	conditional = False
	for hp in hyperparams.keys():
		if 'cond' in hyperparams[hp].keys():
			conditional = True

	# If the model has no conditions, yield a standard grid search
	if not conditional:

		hp_names   = hyperparams.keys()
		hp_lists   = [hyperparams[k]['vals'] for k in hyperparams.keys()]
		grid_total = np.product([len(hyperparams[k]['vals']) for k in hyperparams.keys()])

		for i, args in enumerate(product(*hp_lists)):
			hp_dict = {**hp_dict, **{name : hp for name, hp in zip(hp_names, args)}}
			yield [i+1, grid_total], hp_dict

	# If the model has conditions, build the grid search accordingly
	if conditional:
		with_cond_params = {}
		no_cond_params = {}
		for param in hyperparams:
			
			if 'cond' in hyperparams[param].keys():
				with_cond_params[param] = hyperparams[param]
			else:
				no_cond_params[param] = hyperparams[param]

		hp_names   = no_cond_params.keys()
		hp_lists   = [no_cond_params[k]['vals'] for k in no_cond_params.keys()]

		# Iterate through the grid search to know its length
		i_total = 0
		for _ in conditional_grid_generator(with_cond_params, no_cond_params, \
				hp_dict, hp_lists, hp_names, grid_total=0):
			i_total += 1

		# Iterate through the grid search properly
		yield from conditional_grid_generator(with_cond_params, no_cond_params, \
				hp_dict, hp_lists, hp_names, grid_total=i_total)


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