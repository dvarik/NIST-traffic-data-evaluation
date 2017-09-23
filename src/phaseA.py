import pandas as pd
from pandas import DataFrame
from sklearn import datasets, linear_model
import numpy as np
from datetime import datetime
import sys
import csv
import os

path = sys.argv[1]
zone = sys.argv[2]

lanes = 0
with open(path + '/lanes.txt') as csvfile:
	lanereader = csv.reader(csvfile, delimiter=' ')
	lanes = len(next(lanereader))

columns = []
flow = pd.read_csv(path + "/flow.tsv", sep='\t', header = None)
for i in range(flow.shape[1]):
	columns.append('Flow')
flow.columns = columns

columns = []
probability = pd.read_csv(path + "/prob.tsv", sep="\t", 
                header = None)
for i in range(probability.shape[1]):
	columns.append('Probability')
probability.columns = columns


#Phase 1 - Nearby lanes Linear Regression
if lanes == 1:
	lane_outcome = flow['Flow']
	conf = probability['Probability']

else:
	ground_truth = flow.median(axis=1)	
	ground_truth.columns = ['Ground_Truth']

	vector_lane = pd.DataFrame()
	for i in range(lanes):
		vector = pd.concat([flow.ix[:,i],probability.ix[:,i]],axis=1)
		#vector = vector.fillna(0)
		vector_lane = pd.concat([vector_lane,vector],axis=0,ignore_index = True)

	vector_lane['Ground_Truth'] = pd.concat([ground_truth]*flow.shape[1],axis=0, ignore_index=True)
	vector_lane =  vector_lane.dropna()
	X = vector_lane['Flow'].dropna()
	Y = vector_lane['Ground_Truth']
	regr = linear_model.LinearRegression()
	regr.fit(X[:,np.newaxis], Y)

	del vector
	del vector_lane

	columns = []
	for i in range(lanes):
		columns.append('Lane'+str(i))

	flow.columns = columns
	flow = flow.fillna(0)
	probability.columns = columns

	lane_outcome = []
	conf = []
	for lane in range(lanes):

		if lane == 0:
			avg = flow['Lane' + str(lane+1)]
			prob = probability['Lane' + str(lane+1)]
		elif lane == (lanes-1):
			avg = flow['Lane' + str(lane-1)]
			prob = probability['Lane' + str(lane-1)]
		else:
			avg = (flow['Lane' + str(lane-1)] + flow['Lane' + str(lane+1)])/2
			prob = (probability['Lane' + str(lane-1)] + probability['Lane' + str(lane+1)])/2

		predFlow = regr.predict(avg[:,np.newaxis])
		lane_outcome = np.concatenate([lane_outcome,predFlow])
		conf = np.concatenate([conf,prob])

phase1 = pd.DataFrame()
phase1['predicted'] = lane_outcome
phase1['conf'] = conf
phase1.to_csv(zone + "_phase1.tsv", sep="\t", index= False, header=False, float_format="%.6f")

del conf
del lane_outcome
del phase1


# Phase 2 - Compare preceeding and next timestamps flow
columns = []
timestamp = pd.read_csv(path + "/timestamp.tsv", sep="\t", 
                header = None,  names= ['Timestamp'], parse_dates = ['Timestamp'], dtype = {'Timestamp': 'str'})

columns = []
for i in range(lanes):
	columns.append('Flow'+str(i))
flow.columns = columns

columns=[]
for i in range(lanes):
	columns.append('Probability' + str(i))
probability.columns = columns

prevT = (timestamp['Timestamp'] - timestamp['Timestamp'].shift()).astype('timedelta64[m]')
nextT = (timestamp['Timestamp'].shift(periods=-1) - timestamp['Timestamp']).astype('timedelta64[m]')
prevT[prevT > 10] = 0
nextT[nextT > 10] = 0
prevT[prevT <= 10] = 1
nextT[nextT <= 10] = 1
timestamp['prev'] = prevT
timestamp['next'] = nextT
timestamp = timestamp.fillna(value=0)

for lane in range(lanes):

	flowPrev = flow.ix[:,'Flow' + str(lane)].shift().fillna(value=0)
	flowNext = flow.ix[:,'Flow' + str(lane)].shift(periods=-1).fillna(value=0)
	flow['prev'] = flowPrev
	flow['next'] = flowNext

	probabilityP = probability.ix[:,'Probability' + str(lane)].shift().fillna(value=0)
	probabilityN = probability.ix[:,'Probability' + str(lane)].shift(periods=-1).fillna(value=0)
	probability['p'] = probabilityP
	probability['n'] = probabilityN
	probability['sum'] = probability['p'] + probability['n']
	probability['c1'] = probability['p']/probability['sum']
	del probability['sum']
	probability['c1'] = probability['c1'].fillna(value=0)
	probability['c2'] = 1-probability['c1']

	flow['pred'] = flow['prev']*probability['c1']*timestamp['prev'] + flow['next']*probability['c2']*timestamp['next']
	flow['conf'] = np.where((probability['p'] < probability['n']) , probability['p'], probability['n'])

	del flow['next']
	del flow['prev']

	flow.to_csv(zone + "_phase2.tsv", mode="a", index=False, header=False, columns=['pred', 'conf'], sep="\t", float_format="%.6f")

	del flow['pred']
	del flow['conf']
	del probability['p']
	del probability['n']
	del probability['c1']
	del probability['c2']
