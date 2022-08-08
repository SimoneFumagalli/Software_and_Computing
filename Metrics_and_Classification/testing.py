# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:26:25 2022

@author: Simone Fumagalli
"""

from sklearn.datasets import fetch_openml
import numpy as np
from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
from sklearn.model_selection import train_test_split
import itertools
import os
import sys

path = os.getcwd()
filepath = os.path.dirname
sys.path.append(path)

import Classification
import Validation

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

model = BCM(outputs=100, num_epochs=1, optimizer=Adam(lr=3e-2), 
            interaction_strength=0.,
            weights_init=GlorotNormal(),
            activation='Relu', batch_size=60000)

#Reshaping of the dataset
X_norm, y_transform = Classification.Variable_Reshape(X, y)
#Splitting of the dataset
x_train, x_test, y_train, y_test = \
train_test_split(X_norm, y_transform, test_size=1./8)

def test_checking_batch_size():
    assert isinstance(model.batch_size, int)
    
    model.batch_size = 70000
    check_batch = Classification.checking_batch_size(model, y_train)
    assert model.batch_size == len(y_train)
    
    model.batch_size = 60000
    check_batch = Classification.checking_batch_size(model, y_train)
    assert check_batch == model.batch_size


def test_Reshape():
    '''
    Function to test the Variable_Reshape function
    
    Given:
        X_norm: Dataframe of X input reshaped
        X: Dataframe of X input
        y_transformed: list of labels reshaped
    Expected:
        X_norm has the same type of the input X
        y_transformed has the form of array
    '''
    assert isinstance(X_norm, type(X)) #checking the type of x_norm
    assert isinstance(y_transform, np.ndarray) #checking the type of y_transformed
    
def test_clf():
    classifier = Classification.clf(model, x_train, x_test, y_train, y_test, False)
    fitted_model, prediction = classifier
    
    assert fitted_model == model
    assert len(prediction) == len(y_test)
    for i in range (len(y_test)):
        assert len(prediction[i]) == model.outputs
    
def test_top_ten():
    classifier = Classification.clf(model, x_train, x_test, y_train, y_test)
    top_ten_labels = Classification.top_ten(classifier)
    fitted_model, prediction = classifier
    
    labels = [fitted_model.weights[np.argmax(x)][28*28:].argmax() 
              for x in prediction]
    
    assert len(labels) == len(prediction)
    assert [isinstance(labels[i], int) for i in labels]
    
    #Checking the general length of the top_ten_labels
    assert len(top_ten_labels) == len(y_test)
    
    #Checking the length of each prevision using ten neurons
    for i in range(len(top_ten_labels)):
        assert (np.sum(top_ten_labels[i],0)[1]) == 10

def test_Metrics():
    classifier = Classification.clf(model, x_train, x_test, y_train, y_test)
    metric_single_label = Classification.Metrics(classifier, y_test, False)
    accuracy_single_label, dictionary_single_label = metric_single_label
    metric_top_ten_label = Classification.Metrics(classifier, y_test, False)
    accuracy_top_ten_label, dictionary_top_ten_label = metric_top_ten_label
    
    assert isinstance(accuracy_single_label, int) == \
        isinstance(accuracy_top_ten_label, int)
    assert isinstance(dictionary_single_label, str) == \
        isinstance(dictionary_top_ten_label, str)

def test_checking_number_training():
    clf_times =[4, 6, 8]
    n_splits = 6
    x_train_val, x_test_val, y_train_val, y_test_val = Validation.val_sets(x_train, y_train, n_splits)
    n_split = getattr(Validation.val_sets, 'n_splits')
    assert n_split == n_splits
    for i in range(len(clf_times)):
        if clf_times[i] > n_split:
            assert Exception("The number of times to operate the classification"
                            " must be lower or equal to "
                            "the number of splitting")
            return None
        else:
            return clf_times      

def test_val_sets():
    n_splits = 4
    x_train_val, x_test_val, y_train_val, y_test_val = Validation.val_sets(x_train, y_train, n_splits)
    assert len(x_train_val) == len(x_test_val) == len(y_train_val) \
                                                      == len(y_test_val) == 4
    
    for i in range(n_splits):
        assert len(x_train_val[i]) == len(y_train_val[i])    
        assert  len(x_test_val[i])== len(y_test_val[i])

def test_val_classification():
    n_splits = 4
    clf_times = 3
    validation_sets = Validation.val_sets(x_train, y_train, n_splits)
    classifiers = Validation.val_classification(model, validation_sets, clf_times)
        
    assert len(classifiers) == clf_times