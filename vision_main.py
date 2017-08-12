#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cost
import mean_normalization
import initializeWeights
from scipy import optimize as op

#NOTES:
    #SHOULD PROBABLY LOOK INTO RANDINITIALIZEWEIGHTS WHICH USES EPSIOLONINIT AND STUFF -- BREAKS SYMMETRY
    #SHOULD PROBABLY IMPLEMENT THE CHECKING LIKE GRADIENT CHECK AND STUFF

#read in training data
train = pd.read_csv("train.csv")
#training data contains 42000 examples, 785 columns each

#read in test data
test = pd.read_csv("test.csv")
#test data is 28,000 x 784

#Split into features and labels sets
X_train = (train.ix[:, 1:].values).astype('float32')
y_train = (train.ix[:, 0].values).astype('int32')
X_test = test.values.astype('float32')

#Apply mean normalization
X_train = mean_normalization.normalize(X_train)
X_test = mean_normalization.normalize(X_test)

#visualize a few images
#X_train_images = X_train.reshape(X_train.shape[0], 28, 28)
#for i in range(6, 9):
    #plt.subplot(330 + (i+1))
    #plt.imshow(X_train_images[i], cmap=plt.get_cmap('gray'))
    #plt.title(y_train[i])
    #plt.show()

#Set up network architecture
input_layer_size = 784
hidden_layer_size = 25
num_labels = 10
#Theta1 will be 25 x 785
Theta1 = initializeWeights.randInitialize(25, 784)
#Theta2 will be 10 x 26
Theta2 = initializeWeights.randInitialize(10, 25)
#Theta1 = np.random.random((25, 785))
#Theta2 = np.random.random((10, 26))
initialParams = np.r_[Theta1.ravel(), Theta2.ravel()]
#initialParams = np.concatenate([Theta1.T.ravel(), Theta2.T.ravel()])

#Store number of training examples
num_training_examples = X_train.shape[0]


#Train the Neural Network
def network_cost(initialParams):
    return cost.nn_cost(initialParams, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 0.0)[0]

def gradient(initialParams):
    return cost.nn_cost(initialParams, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 0.0)[1]


#result = op.fmin_tnc(func = network_cost, x0 = initialParams, fprime = gradient)
result = op.fmin_cg(network_cost, x0 = initialParams, fprime=gradient)
print(result)








