#############################
### WQI Prediction        ###
### Code by Gregory Grant ###
### May 13, 2021          ###
#############################

import numpy as np
from itertools import product

num_top_models = 5
models = ['ADA', 'ANN', 'MLR', 'SVR']
submodels = {
	'ADA' : {'loss' : ['linear', 'square', 'exponential']},
	'SVR' : {'kernel' : ['linear', 'sigmoid', 'poly', 'rbf']},
	'MLR' : {None : [None]},
	'ANN' : {None : [None]}
}

for model in models:
	attr = list(submodels[model].keys())[0]
	for param in submodels[model][attr]:

		num_list = []
		mae_list = []
		rmse_list = []
		r2_list = []
		hyper_list = []

		with open('./output/{}-results.csv'.format(model)) as f:

			# Skip first two lines
			line = f.readline()
			line = f.readline()

			# Start with the first line of real data
			while True:
				line = f.readline()
				if line == '':
					break

				linedata = line.split(',')

				num = int(linedata[0])
				mae = float(linedata[4])
				rmse = float(linedata[5])
				r2 = float(linedata[6])
				hyper = ''.join(linedata[7:])[:-1]

				if str(attr) + '=' + str(param) in hyper or attr is None:

					num_list.append(num)
					mae_list.append(mae)
					rmse_list.append(rmse)
					r2_list.append(r2)
					hyper_list.append(hyper)


		num_list = np.array(num_list)
		mae_list = np.array(mae_list)
		rmse_list = np.array(rmse_list)
		r2_list = np.array(r2_list)

		mae_inds = np.argsort(mae_list)
		rmse_inds = np.argsort(rmse_list)
		r2_inds = np.argsort(np.abs(r2_list-1))

		print('Results for {} -- {}={}'.format(model, attr, param))
		print('\nTop models by MAE:  {}'.format(num_list[mae_inds][:num_top_models]))

		for ind in mae_inds[:num_top_models]:
			print(' #{:>4} (mae={:5.3f}) '.format(num_list[ind], mae_list[ind]), '>>>', hyper_list[ind])

		print('\nTop models by RMSE: {}'.format(num_list[rmse_inds][:num_top_models]))

		for ind in rmse_inds[:num_top_models]:
			print(' #{:>4} (rmse={:5.3f})'.format(num_list[ind], rmse_list[ind]), '>>>', hyper_list[ind])

		print('\nTop models by r2:   {}'.format(num_list[r2_inds][:num_top_models]))

		for ind in r2_inds[:num_top_models]:
			print(' #{:>4} (r2={:5.3f})  '.format(num_list[ind], r2_list[ind]), '>>>', hyper_list[ind])

		print('\n'+'-'*40+'\n')