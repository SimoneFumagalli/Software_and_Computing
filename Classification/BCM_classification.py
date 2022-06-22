# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:24:17 2022

@author: Simone Fumagalli
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:09:41 2022

@author: Simone Fumagalli
"""

import numpy as np
from plasticity.model import BCM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from plasticity.utils import view_weights
import pylab as plt
from collections import Counter

class Classification():
    '''
    The idea of this class is to use the BCM model, created by Nico Curty et al.,
    to classify correctly a given image.
    Four major functions has been constructed in order to:
        - Visualize the reached configuration of the neurons;
        - Construct models using the train and test ( or validation) sets;
        - Obtain the performances of the models;
        - Visualize the input and output images, the last corresponding to the 
          best neuron.
    
    Parameters
    
    All the parameters must take the array's form, except the number of 
    attempts.
    
    output: array of int
    The number of neurons to be used.
    
    batch: array of int
    The dimension of the batch.
    
    weights: array of BaseWeights object
    The Weights' initialization strategy.
    
    number_attempts: int
    The number of times the model must be executed.
    
    optimizer: array of Optimizer objects
    Optimizer Objects.
    
    Interaction_strenght: array of int
    Set the interaction strenght between the neurons.
    
    number_epoch: array of int
    The total number of epochs.
    
    --------------------------------------------------------------------------
    In order to go into the detail of the BCM model and of every parameter
    created, look at the Repository 'Plasticity' of Nico-Curti on GitHub.
    '''
    def __init__(self, output : np.array, batch : np.array, weights : np.array, 
                 number_attempts : int, optimizer : np.array, 
                 interaction_strength : np.array, number_epoch : np.array):
        
        self.output = output
        self.batch = batch
        self.weights = weights
        self.number_attempts = number_attempts 
        self.optimizer = optimizer
        self.interaction_strength = interaction_strength
        self.number_epoch = number_epoch

    
    def Variable_Reshape(self,X,y):
        '''
        Function to reshape the images and the labels that enter in the model.        

        Parameters
        ----------
        X : Dataframe
            Dataframe containing informations about the pixels of images.
            
        y : Series
            List containing the labels related to the image.

        Returns
        -------
        X_norm : Dataframe
            Normalized version of the X set.
        y_transform : List
            Categorical form of the labels.

        '''
        X_norm=X*(1./255) # Normalization of the X inputs in range [0,1]
        
        # Transformation of the y vector
        y_int = y.astype('int')
        y_reshape = y_int.values.reshape(-1,1)
        y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
        return X_norm, y_transform

    def Variable_Split(self, X, y, fraction: float):
        '''
        Function to split the input variables X and y in train and test set.        

        Parameters
        ----------
        X : Dataframe
            Dataframe containing informations about the pixels of images.
            
        y : List
            List containing the labels related to the image.
            
        fraction : float
            Fraction representing the ratio between the train and test dataset.

        Returns
        -------
        The four variables x_train, y_train, x_test, y_test composing the train and
        test sets.

        '''
        # Splitting of the variables in train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=fraction,\
                                                                random_state=42)  
        
        return x_train,\
               x_test,\
               y_train,\
               y_test
               
    def modellization(self, num_model):
        '''
        Creation of the model using the parameters of the Classification class.
        
        Parameters
        ----------
        num_model : int
            Parameter corresponding to position of the element of each model 
            considered.
        Returns
        -------
        Construction of the model using the parameters of the class.

        '''
        
        #Construction of the BCM model
        model = [BCM(outputs= self.output[num_model], 
                    num_epochs=self.number_epoch[num_model], 
                    optimizer=self.optimizer[num_model], 
                    interaction_strength=self.interaction_strength[num_model],
                    weights_init=self.weights[num_model],
                    activation='Relu', batch_size=self.batch[num_model],
                    random_state=42)]
        return model
     
    def fitting(self, X, model, y = None):
        '''
        Training of the model.

        Parameters
        ----------
        X : Dataframe
            Train Set of images.
            
        model : function object
            BCM model.
            
        y : List
            Train Set of Labels. The default is None.

        Returns
        -------
        Trained model.

        '''
        
        if y is not None:
            model_fit = model[0].fit(X,y) #Fitting the model in a supervised way
        else:
            model_fit = model[0].fit(X) # Fitting the model, 
                                          # without the knowledge of the labels
        return [model_fit]