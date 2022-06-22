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
    
    def prevision(self, X, y, fitted_model):
        '''
        Function to make predictions on the basis of the fitted model.

        Parameters
        ----------

        X : Dataframe
            Test or validation set of images.

        y : List
            Test or validation set of labels.

        fitted_model : function object
            Fitted model.

        Returns
        -------
        The array containing the predictions.

        '''
        #Usage of the BCM predict function
        prediction = fitted_model[0].predict(X.values, y = np.zeros_like(y))
            
        return prediction
    
    def best_neuron(self, fitted_model, prevision):
        '''
        Method to extract the best labels, once the model has been trained and the 
        predictions have been made.

        Parameters
        ----------
        fitted_model : function object
            Fitted model.
        
        prevision : array
            Array containing the predictions of the model.

        Returns
        -------
        execution : array
            Labels corresponding to the best neuron.

        '''
        #Extraction of the labels
        execution = [fitted_model[0].weights[np.argmax(x)][28*28:].argmax() 
                 for x in prevision]
        
        return execution
    
    def top_ten(self, fitted_model, prevision):
        '''
        Function to extract the information of the ten neurons with the highest
        response.

        Parameters
        ----------
        fitted_model : function object
            Fitted Model.
        prevision : array
            Array containing the predictions of the model.

        Returns
        -------
        top_10 : array
            Array made of arrays, containing the ten best labels for each y.

        '''
        top_10 = []
        labels = fitted_model[0].weights[:,28*28:].argmax(axis=1) # Labels of the
                                                                  # 100 neurons 
        for x in prevision:
             
            # Union of prevision score and neuron
            sorting = sorted(zip(x, labels),\
                                key=lambda x : x[0], reverse=True)[:10]
             
            sorting = [x[1] for x in sorting]
            
            # Counting how many of the ten neurons give the same label as result
            counter_lab = (Counter(sorting).most_common())
            top_10.append(counter_lab)
        return top_10
   
    def checking_y_size(self, y_test, clf):
        '''
        Function to control the size of the y test set, particularly useful when
        the validation size is set in the classification function.

        Parameters
        ----------
        y_test : List
            Y test set used to check the dimension of the set of labels used in
            classification function.
        clf : Function object
            The classification function.

        Returns
        -------
        y_test : List
            Set of labels used in classification function.

        '''
        #Checking the size of the labels used
        if len(y_test) != len(clf[0]):
            print('length of y_test', (len(y_test)), 'is different from that used',\
                  'in the classification function:', len(clf[0]),\
                  '. The last has been considered.')
                
            y_test = clf[0]   
            
        else:
            y_test = y_test
        
        return y_test
        
    def checking_y_labels(self, clf, ten_label = False):
        '''
        Function to control if the set of labels comprises just the result from
        best neuron or from the ten best neurons.

        Parameters
        ----------
        clf : function
            Classification function.
        
        ten_label : Bool, optional
            The default is False.

        Returns
        -------
        y_to_test : List
            Set of labels to use in the study of metrics.

        '''
        if ten_label == False:
            
            y_to_test = clf[1]
        
        else:
            #Extrapolation of the first result from the ten labels for each of
            # image to predict
            y_to_test = [clf[2][x][0][0] for x in range (len(clf[0]))]
        
        return y_to_test
    
    
    def checking_batch(self,num_model, y_train):
        '''
        Control function to check the batch, particularly useful when the
        validation set is used.

        Parameters
        ----------
        i : int
            Number to select the corresponding size of the batch from the batch
            array.
        y_train : List
            Set of labels used in the classification function.

        Returns
        -------
        Int
            Returns the dimension of the batch, according to the dimension of
            the label set used.

        '''
        #CHecking the dimension of the batch. If it is higher than the dimensions
        #of the labels, it will be set equal to the dimension of the labels.
        
        if self.batch[num_model] > len(y_train):
            print('The dimension of the batch',self.batch[num_model],'is much higher '
                  'than that of the samples',len(y_train),'.\
                  This last has been considered.')
            self.batch[num_model] = len(y_train)
        else:
            self.batch[num_model] = self.batch[num_model]
        return self.batch[num_model]
    
    def neurons_configuration(self, fitted_model):
        '''
        Function to generate the image containing the configuration of the 
        neurons.
        
        Parameters
        ----------
        fitted_model : function object
            Fitted model.
        
        '''
        view_weights(fitted_model[0].weights, dims=(28,28))
    
    def neuron_graphs(self, x_train):
        '''
        General function to show the configurations of the neurons.
        The result can be one or more images, according to the 
        number_attempts' value.

        Parameters
        ----------
        x_train : Dataframe
            Train set.
        '''
        models = []
        fits = []
                
        for i in range (self.number_attempts):
            model = self.modellization(i) # Use of model function
            models.append(model) 
            fit = self.fitting(x_train, models[i]) # Use of fitting function
            fits.append(fit)
            graphs = self.neurons_configuration(fits[i]) # Use of neurons_configuration function
