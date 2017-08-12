#!/usr/bin/env python

import numpy as np

def normalize(dataset):
    colMeans = np.mean(dataset, 0)
    #print(colMeans)
    stdDev = np.std(dataset, 0)
    #print(stdDev)
    #normalize each column
    for feature in range(0, dataset.shape[1]):
        if stdDev[feature] != 0:
            dataset[:, feature] = ( dataset[:, feature] - colMeans[feature] ) / stdDev[feature]
    
    return dataset