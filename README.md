# NIST-traffic-data-evaluation
1) Given NIST traffic lane detector measurements containing incorrect flow values, cleaned the values using linear regression model on nearby lanes and nearby timestamps to predict correct flow values and took a weighted average.
2) Given NIST past traffic event data for geographical regions, predicted future event occurrences using regression models like SVM (used SciKit). 

Parallelized computation using Python multiprocessing.
