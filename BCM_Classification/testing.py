# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:26:25 2022

@author: Simone Fumagalli
"""

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
from sklearn.model_selection import train_test_split
import os
import sys
import pytest

path = os.getcwd()
filepath = os.path.dirname(os.path.abspath('Classification.py'))
sys.path.append(path)

import Classification
import Validation

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

@pytest.fixture
def modelling():
    model = BCM(outputs=100, num_epochs=1, optimizer=Adam(lr=3e-2), 
                interaction_strength=0.,
                weights_init=GlorotNormal(),
                activation='Relu', batch_size=60000)
    return model

#Reshaping of the dataset
X_norm, y_transform = Classification.Variable_Reshape(X, y)
#Splitting of the dataset
x_train, x_test, y_train, y_test = \
train_test_split(X_norm, y_transform, test_size=1./8)

def test_checking_batch_size(modelling):
    '''
    Function to test the checking_batch_size function.

    Given:
        model: BCM model
        y_train: list of labels used for the training
    Expected:
        When model.batch_size higher than the length of y_train, modl.batch_size
        is reduced to be equal to the length of y_train
        If model.batch_size lower than the length of y_train, it doesn't change.

    '''
    assert isinstance(modelling.batch_size, int)
    
    modelling.batch_size = 70000
    check_batch = Classification.checking_batch_size(modelling, y_train)
    assert modelling.batch_size == len(y_train)
    
    modelling.batch_size = 60000
    check_batch = Classification.checking_batch_size(modelling, y_train)
    assert check_batch == modelling.batch_size


def test_Reshape():
    '''
    Function to test the Variable_Reshape function
    
    Given:
        X_1: List of int
        
        y_transformed: list of labels reshaped
    Expected:
        X_norm has the same type of the input X
        y_transformed has the form of array
    '''

    X_1 = np.arange(0,20, dtype=float)
    y_series = pd.Series(np.arange(0,10,dtype=int))
    X_norm, y_transform = Classification.Variable_Reshape(X_1, y_series)
    assert all(X_norm == X_1*(1./255))
    
    for i in range(10):
        assert np.where(y_transform[i] == 1.) == y_series[i]    
        
def test_clf():
    '''
    Function to the clf function that implements the BCM functions fit and predict.
    
    Given:
        model: BCM model
        x_train: Dataframe of the images used to train the model
        x_test: Dataframe of the images on which the prediction is made
        y_train: List of labels used to train the model
        y_test: List of labels that should be predicted from the model
    Expected:
        Fitted_model is a ndarray composed by the weights of the model
        The prediction is performed for each of the y_test labels and so their 
        dimensions should correspond
        Each prediction is composed by the result of the predict function for all the
        neuron and so the length of each prediction must correspond to the model.outputs
    '''
    classifier = Classification.clf(model, x_train, x_test, y_train, y_test, False)
    fitted_model, prediction = classifier
    
    assert isinstance(fitted_model, np.ndarray)
    assert len(prediction) == len(y_test)
    for i in range (len(y_test)):
        assert len(prediction[i]) == model.outputs
    
def test_top_ten():
    '''
    Function to test the top_ten_labels function.

    Given:
        top_ten_labels: array of predicted labels using the survey of 
                        the best ten neurons
        labels: extraction of the labels considering the neuron with the higher score
    
    Expected:
        The length of the labels should be equal to that of the predictions, or also
        to the size of the test set
        The type of the labels should be only int
        The length of top_ten_labels array should be equal to the size of test set
        The total number of votes for each label should be equal to ten
        
    
    '''
    classifier = Classification.clf(model, x_train, x_test, y_train, y_test)
    top_ten_labels = Classification.top_ten(classifier)
    fitted_model, prediction = classifier
    
    labels = [fitted_model[np.argmax(x)][28*28:].argmax() 
              for x in prediction]
    
    assert len(labels) == len(prediction)
    assert [isinstance(labels[i], int) for i in labels]
    
    #Checking the general length of the top_ten_labels
    assert len(top_ten_labels) == len(y_test)
    
    #Checking the length of each prevision using ten neurons
    for i in range(len(top_ten_labels)):
        assert (np.sum(top_ten_labels[i],0)[1]) == 10

def test_Metrics():
    '''
    Function to test the Metrics function.

    Given:
        accuracy_single_label: measured accuracy with labels obtained using 
                               just the best neuron
        dictionary_single_label: output of the classification_report function 
                                 using just the best neuron
        accuracy_top_ten_label: measured accuracy with labels obtained using 
                                the survey of the top ten neurons
        dictionary_top_ten_label: output of the classification_report function
                                  using the survey of the top ten neurons
    Expected:
        The accuracy_single_label and accuracy_top_ten_label should have the same
        type, int
        The dictionary_single_label and dictionary_top_ten_label should havet the
        same type, str
    '''
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
    '''
    Function to test the checking_number_training function.

    Given:
        n_splits: int indicating how many times the training set is splitted 
                  in different validation sets.
        clf_times: array of int.
    
    Expected:
        if clf_times is higher than the n_splits the exception error should occur.
        if clf_times is lower or equal to n_splits the returned value should be 
        the clf_times value.
    '''
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
    '''
    Function to test the val_sets function.
    
    Given:
        x_train_val: array of validation set of images used for the training
        x_test_val: array of validation set of images used for the predictions
        y_train: array of validation set of labels used for the training
        y_test: array of validation set of labels that must be predicted
        n_splits: int indicating the times in which the training set is splitted
    Expected:
        The length of different sets should be equal to the number of splitting
        The length of different x_train_val and y_train_val sets should be equal
        The length of different x_test_val and y_test_val sets should be equal        
    '''
    n_splits = 4
    x_train_val, x_test_val, y_train_val, y_test_val = Validation.val_sets(x_train, y_train, n_splits)
    assert len(x_train_val) == len(x_test_val) == len(y_train_val) \
                                                      == len(y_test_val) == 4
    
    for i in range(n_splits):
        assert len(x_train_val[i]) == len(y_train_val[i])    
        assert  len(x_test_val[i])== len(y_test_val[i])

def test_val_classification():
    '''
    Function to test the val_classification function

    Given:
        clf_times: int, number of times to operate the clf function
        classifiers: val_classification function outputs
    Expected:
        The number of times the classification is performed should be equal to
        the clf_times
    '''
    n_splits = 4
    clf_times = 3
    validation_sets = Validation.val_sets(x_train, y_train, n_splits)
    classifiers = Validation.val_classification(model, validation_sets, clf_times)
        
    assert len(classifiers) == clf_times
    
def test_val_metrics():
    '''
    Function to test the val_metrics function.
    
    Given:
        n_splits: int indicating the times in which the training set is splitted
        clf_times: int, number of times to operate the clf function
        validation_sets: output of val_sets function
        val_classifier: output of val_classification function
        validation_metric: output of val_metric function
    Expected:
        The length of validation_sets variable should be equal to clf_times
        For each classification, the metrics is performed and it should return the
        accuracy and classification__report. The length of the validation_metric
        of each classification so should be equal to two
        The first element of each metrics should be equal to a float type
    '''
    n_splits = 8
    clf_times = 4
    validation_sets= Validation.val_sets(x_train, y_train, n_splits)
    val_classifiers = Validation.val_classification(model, validation_sets, clf_times)
    validation_metric = Validation.val_metrics(val_classifiers, validation_sets)
    assert len(validation_metric) == clf_times
    for i in range(clf_times):
        assert len(validation_metric[i]) == 2
        assert isinstance(validation_metric[i][0], float)