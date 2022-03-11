# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:42:26 2022

@author: Simone Fumagalli
"""
from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pylab as plt
from sklearn.metrics import accuracy_score

def variables(X,y, i):
    """
    This method splits the variables into a train set and test set.
    
    Parameters
    ----------
    X : Dataframe
        Dataframe containing informations about the pixels of MNIST images.
    
    y : Series
        List containing the labels related to the MNIST images.
    
    i : int
        It indicates which variable the function returns.

    Returns
    -------
    The four variables x_train, y_train, x_test, y_test composing the train and
    test sets.
    """
    
    X_1=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    #Transformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    
    #Splitting of the variables in training and test sets
    x_train, x_test, y_train, y_test = \
    train_test_split(X_1, y_transform, test_size=1./8, random_state=42)
    variable = [x_train, y_train, x_test, y_test]
    return variable[i]

def fitting(X,y):
    """
    This method define the model used for classification and uses the train
    variables to train the model.

    Parameters
    ----------
    X : Dataframe
        Dataframe of the train set.
    
    y : Series
        List containing the labels in train set.
    
    Returns
    -------
    The trained model.
    """
    model = BCM(outputs=1000, num_epochs=10, optimizer=Adam(lr=4e-2), 
                interaction_strength=0.,
                weights_init=GlorotNormal(),
                activation='Relu', batch_size=10000)
    
    model.fit(X,y) #training of the model
    return model

#PREDICTION

def predict(X, y, model, x_predict):
    """
    This method uses the model to make predictions on a new set, the test set.
    It outputs the raw images and the predicted one.
    
    Parameters
    ----------
    X : Dataframe
        Dataframe cointaining the test set.
    
    y : Series
        List containing the labels in test set.
    
    model : trained model used to make predictions.
    
    x_predict : int
                Number corresponding to the MNIST image that must be shown.
    
    Returns
    -------
    It returns the predicted labels. It is useful for testing purpose.
    
    """
    
    if type(x_predict) == int and x_predict < len(y):
        predict = model.predict(X.values[x_predict].reshape(1, -1), \
                                y=np.zeros_like(y[x_predict].reshape(1, -1)))
    
        # select the neuron connection with the highest response
        highest_response = model.weights[np.argmax(predict)][:28*28].reshape(28, 28)
        
        # collecting the predicted labels
        labels = model.weights[:, 28*28:].argmax(axis=1)
        
        #sorting the label with the highest response
        predicted_label = sorted(zip(predict.ravel(), labels), \
                                 key=lambda x : x[0], reverse=True)[:1]
            
        predicted_label = [x[1] for x in predicted_label]
        
        nc = np.amax(np.abs(model.weights))
        
        #plotting

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
        ax1.set_title('Image from the MNIST dataset:{:d}'.format(y[x_predict].argmax()))
        ax1.imshow(X.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
    
        ax2.set_title('Prediction using BCM:{:d}'.format(predicted_label[0]))
        ax2.imshow(highest_response, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')
    
    else:
        print("Error: x_predict must be an integer number lower than", len(y)-1,\
          ". Please, enter a valid number next time")



######## EVALUATION OF THE ACCURACY

def accuracy(X,y,model):
    """
    This method evaluates the accuracy of the prediction.

    Parameters
    ----------
    X : Dataframe
        Dataframe containing the test set.
    y : Series
        List containing the labels in test set.
    
    model : trained model used to make predictions.
    
    Returns
    -------
    It returns the types and lenghts of the variable for testing purpose.
    
    """
    accuracy = model.predict(X, y)
    
    # prediction of the labels
    y_values = [model.weights[np.argmax(x)][28*28:].argmax() for x in accuracy]

    y_true = y.argmax(axis=1)
    
    y_pred = np.asarray(y_values)
    
    values = [y_values, y_true, y_pred]
    return values
    print('Prediction Accuracy on the test set: {:.3f}'.format(accuracy_score(y_true, y_pred)))
