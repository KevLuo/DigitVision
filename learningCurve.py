#!/usr/bin/env python

import cost
from scipy import optimize as op
import matplotlib.pyplot as plt

def generateLearningCurve(X_train, y_train, X_val, y_val, initial_params, input_layer_size, hidden_layer_size, num_labels, lambda_param):
    m = X_train.shape[0]
    error_train = []
    error_val = []
    for i in range(3500, m + 1, 3500):
        #get theta from the trainng data
        args = (input_layer_size, hidden_layer_size, num_labels, X_train[0:i, :], y_train[0:i], lambda_param)
        theta = op.fmin_cg(cost.nn_cost, x0 = initial_params, fprime=cost.backprop, args = args, maxiter = 150)
        #get the train error
        specific_error_train = cost.nn_cost(theta, input_layer_size, hidden_layer_size, num_labels,
                                            X_train[0:i, :], y_train[0:i], lambda_param)
        error_train.append(specific_error_train)
        #get the validation error
        specific_error_val = cost.nn_cost(theta, input_layer_size, hidden_layer_size, num_labels, X_val, y_val, lambda_param)
        error_val.append(specific_error_val)
    
    print('break1')    
    print(error_train)
    print('break')
    print(error_val)
    #plot
    plt.plot(range(3500, m + 1, 3500), error_train, 'r', label = "training error")
    plt.plot(range(3500, m + 1, 3500), error_val, 'b', label = "validation error")
    plt.xlabel('Number of training examples')
    plt.ylabel('Cost')
    plt.title('Learning Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    