#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cost
import mean_normalization
import initializeWeights
import predict as pred
import validationCurve as vc
import learningCurve as lc
from scipy import optimize as op
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn import datasets

#NOTES:
    #SHOULD PROBABLY IMPLEMENT THE CHECKING LIKE GRADIENT CHECK
    #Perhaps try dropout?
    #More nodes in hidden layer?
    
    #submission_1.csv -- (93.243% accurate)
    #submission_2.csv -- (92.457% accurate) -> increased num_iterations from 100 to 300
    #submission_3.csv -- (93.7% accurate) -> increased lambda to 5
    #submission_4.csv -- (94.543% accurate) -> PCA of 300, higher lambda of 8, and 200 iters
    #submission_5.csv -- (94.086% accurate) -> changed to 100 iters
    #submission_6.csv -- (94.557% accurate) -> changed back to 200 iters, PCA of only 200
    #submission_7.csv -- (94.357% accurate) -> PCA of only 100
    #submission_8.csv -- (94.314% accurate) -> Added lambda of 5
    #submission_9.csv -- (93.986& accurate) -> Doubled numiters to 400
    #submission_10.csv -- (93.657% accurate) -> PCA of only 50
    #submission_11.csv -- (94.557% accurate) -> PCA of 300, lambda of 10, num_iters 150
    #submission_13.csv -- (96.629% accurate) -> PCA of 200, hidden layer w/100 nodes, lambda of 8
    
    #Originally, when cost and gradient were in the same function:  47 minutes to run validation curves
    #When separated out into separate functions and the creation of feedforward helper -> 26 min -> key is less loops to run

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

#pca = decomposition.PCA(n_components=700)
#pca.fit(X_train)
#plt.plot(pca.explained_variance_ratio_)
#plt.ylabel('% of variance explained')
#plt.show()

pca = decomposition.PCA(n_components = 200)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#Split training data into train set and cross-validation set
#X_train is 31,500 x 784
#y_train is 31,500 x 1
#X_val is 10,500 x 784
#y_val is 10,500 x 1
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25)

#visualize a few images
#X_train_images = X_train.reshape(X_train.shape[0], 28, 28)
#for i in range(6, 9):
    #plt.subplot(330 + (i+1))
    #plt.imshow(X_train_images[i], cmap=plt.get_cmap('gray'))
    #plt.title(y_train[i])
    #plt.show()

#Set up network architecture
#input_layer_size = 784
input_layer_size = X_train.shape[1]
hidden_layer_size = 100
#hidden_layer_size = 25
num_labels = 10
#Theta1 will be 25 x 785
Theta1 = initializeWeights.randInitialize(hidden_layer_size, input_layer_size)
#Theta2 will be 10 x 26
Theta2 = initializeWeights.randInitialize(num_labels, hidden_layer_size)
#Theta1 = np.random.random((25, 785))
#Theta2 = np.random.random((10, 26))
initialParams = np.r_[Theta1.ravel(), Theta2.ravel()]
#initialParams = np.concatenate([Theta1.T.ravel(), Theta2.T.ravel()])

#Store number of training examples
num_training_examples = X_train.shape[0]


#Running time of 100 iterations:  10 min
#Running time of 300 iterations:  56 min --> lower test accuracy than 100 (92.457%)
args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 8.0)
optimalParams = op.fmin_cg(cost.nn_cost, x0 = initialParams, fprime=cost.backprop, args = args, maxiter = 150)

#optimalTheta1 is 25 x 785
optimalTheta1 = optimalParams[0:(hidden_layer_size*(input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
#optimalTheta2 is 10 x 26
optimalTheta2 = optimalParams[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels, (hidden_layer_size + 1))

training_predictions = pred.predict(optimalTheta1, optimalTheta2, X_train)
trainingAccuracy = np.mean(training_predictions == y_train)

validation_predictions = pred.predict(optimalTheta1, optimalTheta2, X_val)
validationAccuracy = np.mean(validation_predictions == y_val)

print('training accuracy')
print(trainingAccuracy)

print('validation accuracy')
print(validationAccuracy)
#lc.generateLearningCurve(X_train, y_train, X_val, y_val, initialParams, input_layer_size, hidden_layer_size, num_labels, 0.0)
#vc.generateValidation(X_train, y_train, X_val, y_val, initialParams, input_layer_size, hidden_layer_size, num_labels)

test_predictions = pred.predict(optimalTheta1, optimalTheta2, X_test)

np.savetxt("submission_13.csv", np.c_[range(1, X_test.shape[0] + 1), test_predictions], delimiter = ',',
           header = "ImageId,Label", comments = '', fmt = "%d")







