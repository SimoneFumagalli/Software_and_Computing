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
from Utils import Testing_utils

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

@pytest.fixture
def modelling():
    model = BCM(outputs=100, num_epochs=1, optimizer=Adam(lr=3e-2), 
                interaction_strength=0.,
                weights_init=GlorotNormal(),
                activation='Relu', batch_size=60000)
    return model

@pytest.fixture
def Variables():
    #Reshaping of the dataset
    X_norm, y_transform = Classification.Variable_Reshape(X, y)
    #Splitting of the dataset
    x_train, x_test, y_train, y_test = \
    train_test_split(X_norm, y_transform, test_size=1./8)
    return x_train,x_test,y_train,y_test

def test_checking_batch_size(modelling, Variables):
    '''
    Function to test the checking_batch_size function.

    Given:
        model: BCM model
        y_train: list of labels used for the training
    Expected:
        When model.batch_size is higher than the length of y_train, model.batch_size
        is reduced to be equal to the length of y_train
        If model.batch_size lower than the length of y_train, it doesn't change.

    '''
    assert isinstance(modelling.batch_size, int)
    y_train = Variables[2]
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
        y_transform: list of labels reshaped
    Expected:
        By using the function Variable_Reshape, the every number of 
        variable X_norm should correspond to every number of X_1 divided by 255. 
        The 1 element of the arrays present in y_transform should be in the position
        corresponding to the numbers of y_series.
    '''

    X_1 = np.arange(0,20, dtype=float)
    y_series = pd.Series(np.arange(0,10,dtype=int))
    X_norm, y_transform = Classification.Variable_Reshape(X_1, y_series)
    assert all(X_norm == X_1*(1./255))
    
    for i in range(10):
        assert np.where(y_transform[i] == 1.) == y_series[i]    
        
def test_clf(modelling,Variables):
    '''
    Function to test the clf function that implements the BCM functions fit and predict.
    
    Given:
        model: BCM model
        x_train: Dataframe of the images used to train the model
        x_test: Dataframe of the images on which the prediction is made
        y_train: List of labels used to train the model
        y_test: List of labels that should be predicted from the model
    Expected:
        Fitted_model should be composed by the same number of elements of the output
        of the model.
        Elements of Fitted_model should have the dimension given by the dimension of
        the elements of x_train plus that of the elements of y_train.
        The prediction is performed for each of the y_test labels and so their 
        dimensions should correspond
        Each prediction is composed by the result of the predict function for all the
        neuron and so the length of each prediction must correspond to the model.outputs
    '''
    x_train, x_test, y_train, y_test = Variables
    classifier = Classification.clf(modelling, x_train, x_test, y_train, y_test)
    fitted_model, prediction = classifier
    
    assert len(fitted_model) == modelling.outputs
    assert [len(fitted_model[i]) == x_train.shape[1] + y_train.shape[1] \
            for i in range (modelling.outputs)]
    
    assert len(prediction) == len(y_test)
    for i in range (len(y_test)):
        assert len(prediction[i]) == modelling.outputs
    
def test_top_ten(modelling, Variables):
    '''
    Function to test the top_ten_labels function.

    Given:
        top_ten_labels: array of predicted labels using the survey of 
                        the best ten neurons    
    Expected:
        The length of top_ten_labels array should be equal to the size of test set
        The total number of votes for each label should be equal to the shape of
        the elements of the test set.
        
    
    '''
    x_train, x_test, y_train, y_test = Variables
    classifier = Classification.clf(modelling, x_train, x_test, y_train, y_test)
    top_ten_labels = Classification.top_ten(classifier)
        
    
    #Checking the general length of the top_ten_labels
    assert len(top_ten_labels) == len(y_test)
    
    #Checking the length of each prevision using ten neurons
    for i in range(len(top_ten_labels)):
        assert (np.sum(top_ten_labels[i],0)[1]) == y_test.shape[1]

def test_plot_params(modelling, Variables):
    '''
    Function to test the Metrics function.

    Given:
        classifier: array containing the weights and the predictions.
        sorted_weights: weigths listed in descending order.
        sorting_prediction: the array of prediction,corresponding to the x_predict
                            position, is ordered in a descendent manner.
        sorting_labels: labels corresponding to the prediction array are ordered
                        descendently.
    Expected:
        nc is expected to be the highest values among all the weights present in
        the classifier.
        The prediction index used for the choice of the best label must correspond
        to the highest value of prediction.
        The label considered should be an index corresponding to the highest 
        value present in the array considered.
    '''
    x_train, x_test, y_train, y_test = Variables
    classifier = Classification.clf(modelling, x_train, x_test, y_train, y_test)
    fitted_model, prediction = classifier
    x_predict = 0
    
    nc, label, top_ten_label, best_result = Classification.plot_params(classifier, x_predict)
    #Testing the nc value to be the highest possible
    sorted_weights = Testing_utils.reshaping_weights(classifier)    
    assert nc == sorted_weights[0]

    sorting_prediction, sorting_label = Testing_utils.sorting(classifier, x_predict)
    #Testing the index of prediction corresponding to the highest value of the array
    #considered.
    assert np.argmax(prediction[x_predict]) == sorting_prediction.index[0]
    
    #Testing the descendent order of the label
    assert [sorting_label.values[i] >= sorting_label.values[i+1] for i in range(len(sorting_label) - 1)]
    assert label == sorting_label.index[0]
    
def test_Metrics(modelling, Variables):
    '''
    Function to test the Metrics function.

    Given:
        y_true: list of test labels.
        accuracy_single_label: measured accuracy with labels obtained using 
                               just the best neuron
        accuracy_top_ten_label: measured accuracy with labels obtained using 
                                the survey of the top ten neurons
    Expected:
        The accuracy_single_label and accuracy_top_ten_label should be equal to
        the value of accuracy obtained using the theoretical formula.
    '''
    x_train, x_test, y_train, y_test = Variables
    classifier = Classification.clf(modelling, x_train, x_test, y_train, y_test)
    y_true = y_test.argmax(axis=1)
    
    #Demonstrating that the output of the Metrics of the single labels
    #correspond to the accuracy formula
    accuracy_single = Classification.Metrics(classifier, y_test, False)[0]
    y_lab = Testing_utils.ylabels(classifier)
    accuracy_formula = len(np.where(y_lab==y_true)[0])/len(y_lab)
    
    assert accuracy_formula == accuracy_single
    
    #Demonstrating that the output of the Metrics of the multiple labels
    #correspond to the accuracy formula
    accuracy_top_ten_label = Classification.Metrics(classifier, y_test, True)[0]
    ten_result = Testing_utils.top_10_labels(classifier)
    accuracy_formula_10 = len(np.where(ten_result==y_true)[0])/len(ten_result)
    
    assert accuracy_formula_10 == accuracy_top_ten_label

def test_checking_number_training(Variables):
    '''
    Function to test the checking_number_training function.

    Given:
        n_splits: int indicating how many times the training set is splitted 
                  in different validation sets.
        clf_times: array of int.
    
    Expected:
        if clf_times is higher than the n_splits the exception error should occur.
    '''
    x_train, y_train = Variables[0], Variables[2]
    clf_times =[7, 8, 9]
    n_splits = 6
    x_train_val, x_test_val, y_train_val, y_test_val = Validation.val_sets(x_train, y_train, n_splits)
    n_split = getattr(Validation.val_sets, 'n_splits')
    assert n_split == len(x_train_val)
    
    for i in range(len(clf_times)):
        with pytest.raises(Exception):
            Validation.check_number_training(clf_times[i])


def test_val_sets(Variables):
    '''
    Function to test the val_sets function.
    
    Given:
        x_train_val: array of validation set of images used for the training
        x_test_val: array of validation set of images used for the predictions
        y_train: array of validation set of labels used for the training
        y_test: array of validation set of labels that must be predicted
        n_splits: int indicating the times in which the training set is splitted
    Expected:
        The length of different sets should be equal to the number of splitting.
        The created sets of x_train, x_test, y_train, y_test should have different
        elements.
        The composition x_train_val and x_test_val sets should be equal to
        set used for the splitting.
        The composition y_train_val and y_test_val sets should be equal to
        set used for the splitting.
    '''
    n_splits = 4
    x_train, y_train = Variables[0], Variables[2]
    x_train_val, x_test_val, y_train_val, y_test_val = Validation.val_sets(x_train, y_train, n_splits)
    
    
    x_train_ = x_train.values.reshape(61250,784)
    x_t_list = Testing_utils.listing_and_sorting(x_train_)
    y_t_list = Testing_utils.listing_and_sorting(y_train)
    
    assert len(x_train_val) == len(x_test_val) == len(y_train_val) \
                                                      == len(y_test_val) == n_splits
    for j in range(n_splits-1):
        assert np.array_equal(x_train_val[j], x_train_val[j+1]) == False
        assert np.array_equal(x_test_val[j], x_test_val[j+1]) == False
        assert np.array_equal(y_train_val[j], y_train_val[j+1]) == False
        assert np.array_equal(x_test_val[j], y_test_val[j+1]) == False
        
    for i in range(n_splits):
        x_val = np.concatenate((x_train_val[i],x_test_val[i]))
        x_val_list = Testing_utils.listing_and_sorting(x_val)
        y_val = np.concatenate((y_train_val[i],y_test_val[i]))
        y_val_list = Testing_utils.listing_and_sorting(y_val)
        
        assert x_t_list == x_val_list
        assert y_t_list == y_val_list
        
def test_val_classification(modelling, Variables):
    '''
    Function to test the val_classification function

    Given:
        clf_times: int, number of times to operate the clf function
        x_train: set splitted to obtained the validation train and test set
        y_train: set splitted to obtained the validation train and test set

    Expected:
        The number of times the classification is performed should be equal to
        the clf_times
        Each fitted_model created should have a number of elements equal to 
        the outputs parameter.
        Each of these elements should have a dimension equal to the sum the
        dimension of train sets.
        The length of the predictions should have the same dimensions of the test
        labels.
        Each element of the predictions should be equal have a number of elements
        equal to the outputs parameter.
    '''
    n_splits = 4
    clf_times = 3
    x_train, y_train = Variables[0], Variables[2]
    validation_sets = Validation.val_sets(x_train, y_train, n_splits)
    
    x_train_val, x_test_val, y_train_val, y_test_val = validation_sets
    x_train_partial, x_test_partial, y_train_partial,y_test_partial = \
        Testing_utils.length_var(validation_sets, clf_times)
        
    val_classifiers = Validation.val_classification(modelling, validation_sets, clf_times)    
    assert len(val_classifiers) == clf_times
    
    fitted_models, predictions = [],[]
    for i in range(clf_times):
        fitted_models.append(val_classifiers[i][0]) 
        predictions.append(val_classifiers[i][1])
        assert len(fitted_models[i]) == modelling.outputs
        assert [len(fitted_models[i][j]) == x_train_partial[i].shape[1] + y_train_partial[i].shape[1] \
                                        for j in range (modelling.outputs)]

        assert len(predictions[i]) == len(y_test_partial[i])
        for k in range (len(predictions[i])):
            assert len(predictions[i][k]) == modelling.outputs
    
def test_val_metrics(modelling, Variables):
    '''
    Function to test the val_metrics function.
    
    Given:
        n_splits: int indicating the times in which the training set is splitted
        clf_times: int, number of times to operate the clf function
        validation_sets: output of val_sets function
        val_classifier: output of val_classification function
        val_single_label_metric: output of val_metric function usign label in a
                                single configuration
        val_ten_label_metric: output of val_metric function using label in a 
                              ten configuration
    Expected:
        The length of val_single_label_metric and val_ten_label_metric
        should be equal to clf_times.
        The expected output of the val_metric functions should correspond to the
        general formula of the accuracy.
    '''
    n_splits = 8
    clf_times = 4
    x_train, y_train = Variables[0], Variables[2]
    validation_sets= Validation.val_sets(x_train, y_train, n_splits)
    val_classifiers = Validation.val_classification(modelling, validation_sets, clf_times)
    val_single_label_metric = Validation.val_metrics(val_classifiers, validation_sets, False)
    val_ten_label_metric = Validation.val_metrics(val_classifiers, validation_sets, True)
    assert len(val_single_label_metric) == clf_times
    assert len(val_ten_label_metric) == clf_times
    
    #Testing the validation metrics using the single label configuration
    single_label_accuracies = []
    y_single_labs = []
    y_true = []
    for i in range(clf_times):
        single_label_accuracies.append(val_single_label_metric[i][0])
        
        y_single_labs.append(Testing_utils.ylabels(val_classifiers[i]))
        
        y_true.append(validation_sets[3][i].argmax(axis=1))
        
        accuracy_formula = len(np.where(y_single_labs[i]==y_true[i])[0])/len(y_single_labs[i])
        assert accuracy_formula == single_label_accuracies[i]
    
    #Testing the validation metrics using the ten label configuration
    ten_label_accuracies = []
    y_ten_labs = []
    for i in range(clf_times):
        ten_label_accuracies.append(val_ten_label_metric[i][0])
        
        y_ten_labs.append(Testing_utils.top_10_labels(val_classifiers[i]))
        
        accuracy_formula = len(np.where(y_ten_labs[i]==y_true[i])[0])/len(y_ten_labs[i])
        assert accuracy_formula == ten_label_accuracies[i]