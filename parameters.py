#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### April 9, 2021         ###
#############################

par = {}

# Number of folds for K-folds cross-validation
par['n_folds'] = 10

# Possible hyperparameters for SVR method
par['svr_hyper'] = {
	'kernel'	:	['linear', 'sigmoid', 'poly', 'rbf'],
	'gamma'		:	[0.001, 0.01, 0.1, 1, 10, 20, 30,40, 50],
	'C'			:	[0.1, 1, 10, 100, 500, 1000,2500, 5000],
	'epsilon'	:	[0.001, 0.01, 0.1, 1, 10, 20, 30,40, 50, 60, 70, 80, 90, 100],
	'degree'	:	[1, 2, 3, 4],
}

# Possible hyperparameters for ADA method
par['ada_hyper'] = {
	'loss'		:	['linear', 'square', 'exponential'],
}

# Possible hyperparameters for ANN method
par['ann_hyper'] = {
	'max_iter'				:	[5000],
	'early_stopping'		:	[True],
	'n_iter_no_change'		:	[100],
	'activation'			:	['identity', 'logistic', 'tanh', 'relu'],
	'solver'				:	['lbfgs', 'sgd', 'adam'],
	'hidden_layer_sizes'	:	[ (20),   (20, 20),    (20, 20, 20),
								  (50),   (50, 50),    (50, 50, 50),
								 (100), (100, 100), (100, 100, 100)]
}

# List out shorter names for data headers as keys
par['input_keys'] = ['pH', 'turb', 'DO', 'CHLA', 'CDOM', 'cond', 'TRPN', 'FC', 'BOD', 'T']
par['data_keys'] = par['input_keys'] + ['WQI']
par['exclude_keys'] = ['FC', 'BOD']