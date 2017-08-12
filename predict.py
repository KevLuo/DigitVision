#!/usr/bin/env python

import numpy as np
import sigmoid as sig

def predict(Theta1, Theta2, X):
    
    m = X.shape[0]
    
    #initialize cost
    J = 0.0
    
    #Add ones to account for bias term -> X will now be 42,000 x 785
    onesCol = np.ones((X.shape[0], 1))
    X = np.c_[onesCol, X]
    a_1 = X
    
    #Theta1 is 25 x 785
    #a_1 is 42,000 x 785
    #z_2 is 42,000 x 25
    z_2 = np.dot(a_1, Theta1.T)
    a_2 = sig.sigmoid(z_2)
    
    #append column of ones for bias term
    #a_2 will now be 42,000 x 26
    #Theta2 is 10 x 26
    a_2 = np.c_[onesCol, a_2]
    z_3 = np.dot(a_2, Theta2.T)
    #hypothesis is 42,000 x 10
    hypothesis = sig.sigmoid(z_3)
    
    predictions = np.argmax(hypothesis, axis=1)
    
    return predictions