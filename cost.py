import numpy as np
import sigmoid as sig
import sigmoidGradient as sg

def nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_param):

################################################################################################################## 
    #FEED FORWARD TO OBTAIN HYPOTHESIS MATRIX
    Theta1 = nn_params[0:(hidden_layer_size*(input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels, (hidden_layer_size + 1))
    #store the number of training examples for future use
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
    
##################################################################################################################
    #COMPUTE COST FUNCTION USING HYPOTHESIS MATRIX
    
    #convert 1D labels into 10d labels, with a 1 in the chosen class and a 0 everywhere else
    temp_eye = np.eye(num_labels)
    #y_expanded is 42,000 x 10
    y_labels = temp_eye[y]
    
    #loop through each ith example and cumulatively record cost
    for i in range(0, m):
        J += (-1.0/m) * ( (np.dot(y_labels[i, :], np.log(hypothesis[i, :].T))) + np.dot((1.0 - y_labels[i, :]), np.log(1.0 - hypothesis[i, :].T )) )
        
##################################################################################################################
    #REGULARIZE COST
    
    #obtain the regularization term component relating to Theta1
    reg_theta1 = 0
    #remove bias term for calculation -> Theta1_temp will be 25 x 784
    Theta1_temp = Theta1[:, 1:]
    for j in range(0, Theta1_temp.shape[0]):
        for k in range(0, Theta1_temp.shape[1]):
            reg_theta1 += np.square(Theta1_temp[j, k])
    
    #obtain the regularization term component relating to Theta2
    reg_theta2 = 0
    #remove bias term for calculation -> Theta2_temp will be 10 x 25
    Theta2_temp = Theta2[:, 1:]
    for j in range(0, Theta2_temp.shape[0]):
        for k in range(0, Theta2_temp.shape[1]):
            reg_theta2 += np.square(Theta2_temp[j, k])
    
    #calculate regularization term using above components
    reg_term = (lambda_param/(2*m)) * (reg_theta1 + reg_theta2)
    
    #Add regularization term to the unregularized cost
    J += reg_term
    
##################################################################################################################   
    #BACKPROPOGATION TO COMPUTE GRADIENTS
    
    #Compute delta_3
    #delta_3 is 42,000 x 10
    delta_3 = hypothesis - y_labels

    #Compute delta_2
    #delta_2 is 42,000 x 25
    delta_2 = np.dot(delta_3, Theta2[:, 1:]) * sg.sigmoidGradient(z_2)
    
    #Compute Deltas using deltas -> the triangles/actual gradient of cost function
    #Delta_1 is 25 x 785
    Delta_1 = np.dot(delta_2.T, a_1)
    #Delta_2 is 10 x 26
    Delta_2 = np.dot(delta_3.T, a_2)
    #print(Delta_2)
    #print(Delta_2.shape)
    #Average out over all training examples and apply regularization to obtain final gradients
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    #Treat bias term separately (w/o regularization)
    Theta1_grad[:, 0] = (1.0/m) * Delta_1[:, 0]
    Theta1_grad[:, 1:] = (1.0/m) * Delta_1[:, 1:] + (lambda_param/m) * Theta1[:, 1:]
    #Treat bias term separately (w/o regularization)
    Theta2_grad[:, 0] = (1.0/m) * Delta_2[:, 0]
    Theta2_grad[:, 1:] = (1.0/m) * Delta_2[:, 1:] + (lambda_param/m) * Theta2[:, 1:]
    
    #unroll gradients into one variable
    grad = np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()]
    #grad = np.concatenate([Theta1_grad.T.ravel(), Theta2_grad.T.ravel()])
    #grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    #print(J)
    return [J, grad]
    
    
    
    
    