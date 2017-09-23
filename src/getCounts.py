import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.path as mplPath
from multiprocessing import Pool
import time
from datetime import datetime
# from sklearn import datasets, linear_model
import sys
import csv
import os

trials = pd.read_csv("E:/UFL/DataScience/lab11/prediction_trials.tsv",sep='\t')
events = pd.read_csv("E:/UFL/DataScience/lab11/events_train.tsv",sep='\t')
events = events[['event_type','closed_tstamp','latitude','longitude']]

events['year'] = pd.to_datetime(events['closed_tstamp']).dt.year
trials['year'] = pd.to_datetime(trials['end']).dt.year
trials['year'] = trials['year'].astype(int)

# eventsVal = events[events['year'] <= 2013]

filteredEvents = events[(events.event_type == 'roadwork') | (events.event_type == 'accidentsAndIncidents') | (events.event_type == 'precipitation')
 | (events.event_type == 'deviceStatus') | (events.event_type == 'obstruction') | (events.event_type == 'trafficConditions')]

filteredEvents = filteredEvents.reset_index(drop=True)
filteredEvents['year'] = filteredEvents['year'].astype(int)
print(len(filteredEvents))

for i in range(len(trials)):
	
	lat1 = trials.get_value(i,'nw_lat')
	lon1 = trials.get_value(i,'nw_lon')
	lat2 = trials.get_value(i,'se_lat')
	lon2 = trials.get_value(i,'se_lon')

	points = filteredEvents[filteredEvents.longitude > lon1]
	points = points[points.longitude < lon2]
	points = points[points.latitude < lat1]
	points = points[points.latitude > lat2]
	print(len(points),"points")
	matchedEvents = points
	counts_df = pd.DataFrame({'count' : matchedEvents.groupby(['event_type','year']).size()}).reset_index()
	if len(counts_df)!=0:
		matchedEvents = pd.merge(matchedEvents, counts_df, on=['event_type','year'], how = 'inner')
		print(matchedEvents)
		matchedEvents['nw_lat'] = lat1;
		matchedEvents['nw_lon'] = lon1;
		matchedEvents['se_lat'] = lat2;
		matchedEvents['se_lon'] = lon2;
		matchedEvents['boxyear'] = trials.get_value(i,'year')
	matchedEvents.to_csv("E:/UFL/DataScience/lab11/counts.csv", sep=",", mode="a", 
		columns=['nw_lat','nw_lon','se_lat','se_lon','boxyear','event_type','year','count'], index= False, header=False, float_format='%.8f')


