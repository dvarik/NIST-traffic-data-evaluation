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

# Phase 3 - merge with weighted average using phase1, 2
phase1 = pd.read_csv(zone+"_phase1.tsv", sep='\t', header = None)
phase1.columns = ['flow','prob']
phase2 = pd.read_csv(zone+"_phase2.tsv", sep="\t", header = None, usecols=[0,1])
phase2.columns = ['flow','prob']

os.remove(zone+"_phase1.tsv")
os.remove(zone+"_phase2.tsv")

columns = []

for i in range(lanes):
	columns.append('Flow')
flow.columns = columns

columns = []

for i in range(lanes):
	columns.append('Probability')
probability.columns = columns

phase3 = pd.DataFrame()
for i in range(lanes):
	vector = pd.concat([flow.ix[:,i],probability.ix[:,i]],axis=1)
	vector = vector.fillna(0)
	phase3 = pd.concat([phase3,vector],axis=0,ignore_index=True)

length = len(flow)

del flow
del probability

phase1['w'] = phase1['prob']/(phase1['prob'] + phase2['prob'] + phase3['Probability'])
phase1['w'] = phase1['w'].fillna(0)
phase2['w'] = phase2['prob']/(phase1['prob'] + phase2['prob'] + phase3['Probability'])
phase2['w'] = phase2['w'].fillna(0)
phase3['w'] = 1 - (phase1['w'] + phase2['w'])

merged = pd.DataFrame()
merged['Merged_Flow'] = phase1['w']*phase1['flow']
del phase1
merged['Merged_Flow'] = merged['Merged_Flow'] + phase2['w']*phase2['flow']
del phase2
merged['Merged_Flow'] = merged['Merged_Flow'] + phase3['w']*phase3['Flow']
del phase3

final = pd.DataFrame()
ctr = 0
for i in range(lanes):
	final = pd.concat([final,merged.ix[ctr:ctr+(length-1),:].reset_index(drop=True)],axis=1,ignore_index=True)
	ctr = ctr + (length)
final = final.astype(int)

del merged
final.to_csv(zone + ".flow.txt", sep="\t", index = False, header=False)