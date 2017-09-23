import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.path as mplPath
from multiprocessing import Pool
import time
from datetime import datetime
from sklearn import datasets, linear_model
import sys
import csv
import os

counts = pd.read_csv("/home/datascience/shared/lab11/counts.csv",sep=',', error_bad_lines=False)
counts.columns = ['nw_lat','nw_lon','se_lat','se_lon','boxyear','event_type','year','count']
counts = counts.fillna(0)
counts['year'] = counts['year'].astype(int)
counts = (counts.drop_duplicates()).reset_index(drop=True)

trials = pd.read_csv("/home/datascience/shared/lab11/prediction_trials.tsv",sep='\t')
event_types = ['roadwork','accidentsAndIncidents','precipitation','deviceStatus','obstruction','trafficConditions']

counts = counts[~pd.DataFrame(counts['nw_lat']).applymap(np.isreal).all(1)]
counts = counts[~pd.DataFrame(counts['nw_lon']).applymap(np.isreal).all(1)]
counts = counts[~pd.DataFrame(counts['se_lat']).applymap(np.isreal).all(1)]
counts = counts[~pd.DataFrame(counts['se_lon']).applymap(np.isreal).all(1)]
counts = counts[counts['event_type'].isin(event_types)]
# counts = counts[counts['year'].apply(lambda x: type(x) == int)]
counts = counts.reset_index(drop=True)
grouped = counts.groupby(['nw_lat','nw_lon','se_lat','se_lon', 'boxyear','event_type'])
cols = ['nw_lat','nw_lon','se_lat','se_lon','roadwork','accidentsAndIncidents','precipitation','deviceStatus','obstruction','trafficConditions']


def func(grouped):
	prevkey = 0
	index = -1
	res = pd.DataFrame({'nw_lat': [0]})
	result = pd.DataFrame({'nw_lat': [0]})
	
	for key,val in grouped:
		eventname = key[5]	
		if key[0] != prevkey:
			if index < 0:
				index = 0
			else:
				result = result.append(res)
				index += 1
			prevkey = key[0]
			res.ix[0,'nw_lat'] = key[0]
			res.ix[0,'nw_lon'] = key[1]
			res.ix[0,'se_lat'] = key[2]
			res.ix[0,'se_lon'] = key[3]
			for x in event_types:
				res.ix[0,x] = 0

		X = val['year']
		Y = val['count']
		regr = linear_model.LinearRegression()
		regr.fit(X[:,np.newaxis], Y)
		test = key[4]
		pval = regr.predict(test)[0]
		# pval = 5
		res.ix[0,eventname] = pval

	return result

# print(result)	

# result = pd.read_csv("E:/UFL/DataScience/lab11/result.tsv",sep='\t', error_bad_lines=False, header=None)
# result.columns = ['nw_lat','nw_lon','se_lat','se_lon','accidentsAndIncidents','roadwork','precipitation','deviceStatus','obstruction','trafficConditions']
# trials = pd.read_csv("E:/UFL/DataScience/lab11/prediction_trials.tsv",sep='\t')
# trials.nw_lon = trials['nw_lon'].map('{:,.8f}'.format)
# result['nw_lon'] = result['nw_lon'].astype(float)
# result.nw_lon = result['nw_lon'].map('{:,.8f}'.format)
# final = pd.merge(trials, result, on=['nw_lon'], how = 'left')
# final = final.fillna(0)
# event_types = ['roadwork','accidentsAndIncidents','precipitation','deviceStatus','obstruction','trafficConditions']
# for x in event_types:
#     final[x] = final[x]/12
# # print(final)
# final.to_csv("/home/dataScience/shared/lab11/prediction.tsv", float_format="%.2f", sep="\t", columns=['accidentsAndIncidents','roadwork','precipitation','deviceStatus','obstruction','trafficConditions'], index= False, header=False)
# final = pd.read_csv("/home/dataScience/shared/lab11/prediction.tsv",sep='\t', header=None)
# final[final < 0] = 0
# final.to_csv("/home/dataScience/shared/lab11/prediction.tsv", float_format="%.2f", sep="\t", index= False, header=False)

if __name__ == '__main__':
	p = Pool()
	start = time.time()
	res = pd.DataFrame((p.apply(func,grouped)))
	elapsed = time.time() - start
	print elapsed
	res.to_csv("/home/datascience/shared/lab11/res.tsv", sep='\t', index = False, headers = False)