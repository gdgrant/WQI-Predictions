#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### June 3, 2021          ###
#############################

import numpy as np
from itertools import product

# Analysis parameters
mode           = 'testing' # May be testing or training
num_folds      = 5 # Select models with this number of folds
num_top_models = 5 # Show this many of the top-performing models

# Analyze these types of models
models         = ['ADA', 'ANN', 'MLR', 'SVR']

# Specific parameters to filter out when comparing models
submodels = {
	'ADA' : {'loss' : ['linear', 'square', 'exponential']},
	'SVR' : {'kernel' : ['linear', 'sigmoid', 'poly', 'rbf']},
	'MLR' : {None : [None]},
	'ANN' : {None : [None]}
}

# Select for whether training or testing indices are needed
if mode == 'testing':
	mode_inds = [4,5,6]
elif mode == 'training':
	mode_inds = [1,2,3]

# Iterate over the selected models
for model in models:

	# Iterate over the requisite sub-models
	attr = list(submodels[model].keys())[0]
	for param in submodels[model][attr]:

		# Start collecting data
		num_list = []
		mae_list = []
		rmse_list = []
		r2_list = []
		hyper_list = []

		# Open the file set by the analysis parameters
		with open('./output/{}-{}fold-results.csv'.format(model, num_folds)) as f:

			# Skip first two lines
			line = f.readline()
			line = f.readline()

			# Start with the first line of real data
			while True:

				# Obtain the current line
				line = f.readline()
				if line == '':
					break

				# Split the data into the list
				linedata = line.split(',')

				# Sort the data into the relevant fields
				num = int(linedata[0])
				mae = float(linedata[mode_inds[0]])
				rmse = float(linedata[mode_inds[1]])
				r2 = float(linedata[mode_inds[2]])
				hyper = ''.join(linedata[7:])[:-1]

				# Check if the asked-for special parameter is fulfilled
				if str(attr) + '=' + str(param) in hyper or attr is None:

					# Save this line's data
					num_list.append(num)
					mae_list.append(mae)
					rmse_list.append(rmse)
					r2_list.append(r2)
					hyper_list.append(hyper)

		# Convert the data lists to arrays
		num_list = np.array(num_list)
		mae_list = np.array(mae_list)
		rmse_list = np.array(rmse_list)
		r2_list = np.array(r2_list)

		# Get the indices of the best-performing models
		mae_inds = np.argsort(mae_list)
		rmse_inds = np.argsort(rmse_list)
		r2_inds = np.argsort(np.abs(r2_list-1))

		# Display all the results in a neat format

		print('Results for {}, {} fold -- {}={}'.format(model, num_folds, attr, param))
		print('\nTop models by r2:   {}'.format(num_list[r2_inds][:num_top_models]))

		for ind in r2_inds[:num_top_models]:
			print(' #{:>4} (r2={:5.3f})  '.format(num_list[ind], r2_list[ind]), '>>>', hyper_list[ind])

		print('\nTop models by MAE:  {}'.format(num_list[mae_inds][:num_top_models]))

		for ind in mae_inds[:num_top_models]:
			print(' #{:>4} (mae={:5.3f}) '.format(num_list[ind], mae_list[ind]), '>>>', hyper_list[ind])

		print('\nTop models by RMSE: {}'.format(num_list[rmse_inds][:num_top_models]))

		for ind in rmse_inds[:num_top_models]:
			print(' #{:>4} (rmse={:5.3f})'.format(num_list[ind], rmse_list[ind]), '>>>', hyper_list[ind])


		print('\n'+'-'*40+'\n')