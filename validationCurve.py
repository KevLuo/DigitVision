#!/usr/bin/env python

import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt
import cost

def generateValidation(X_train, y_train, X_val, y_val, initial_params, input_layer_size, hidden_layer_size, num_labels):
    lambda_vec = [0.0, 0.01, 0.05, 0.1, 1.0, 5.0, 10.0]
    #lambda_vec = [0, 10]
    error_train = []
    error_val = []
    
    for i in range(0, len(lambda_vec)):
        args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_vec[i])
        #get the corresponding theta
        theta = op.fmin_cg(cost.nn_cost, x0 = initial_params, fprime = cost.backprop, args = args, maxiter = 60)
        #get the error w/ that theta on training set
        ith_error_train = cost.nn_cost(theta, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 0)
        error_train.append(ith_error_train)
        #get the error w/ that theta on validation set
        ith_error_val = cost.nn_cost(theta, input_layer_size, hidden_layer_size, num_labels, X_val, y_val, 0)
        error_val.append(ith_error_val)
    
    print(error_train)
    print('break')
    print(error_val)
    
    plt.plot(lambda_vec, error_train, 'r', label = "Training Error")
    plt.plot(lambda_vec, error_val, 'b', label = "Validation Error")
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.title('Validation Curve -- Automating Lambda Selection')
    plt.show()
    
    return None