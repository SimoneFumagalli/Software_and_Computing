# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:42:16 2022

@author: Simone Fumagalli
"""

import numpy as np
import pandas as pd
import Classification

def ylabels(classifier):
    '''
    Function to extract the labels information.

    Parameters
    ----------
    classifier : array
        Array containing the weights and prediction informations.

    Returns
    -------
    y_lab : list of array
        List containing the y labels in form of OneHOt Encoding.
    '''
    
    y_lab = [classifier[0][np.argmax(x)][28*28:].argmax() 
              for x in classifier[1]]
    return y_lab

def top_10_labels(classifier):
    '''
    Function to extract the best label obtained from the  the top_ten function.

    Parameters
    ----------
    classifier : array
        Array containing the weights and prediction informations.

    Returns
    -------
    ten_result : List
        List of labels.

    '''
    top_10 = Classification.top_ten(classifier)
    ten_result = [top_10[x][0][0] for x in range(len(top_10))]
    return ten_result

def reshaping_weights(classifier):
    '''
    Function to descendently sort the the weights.

    Parameters
    ----------
    classifier : array
        Array containing the weights and prediction informations.

    Returns
    -------
    sorting : Series
        Series in descending order.
    '''
    
    fitted_model = classifier[0]
    new_shape = len(fitted_model)*len(fitted_model[0])
    reshape_abs_weights = (np.abs(fitted_model)).reshape(new_shape)
    sorting = sorted(reshape_abs_weights, reverse = True)
    return sorting

def sorting(classifier, x_predict):
    '''
    Function to sort the prediction and the corresponding labels.

    Parameters
    ----------
    classifier : array
        Array containing the weights and prediction informations.
    x_predict : int
        Int about the image to study.

    Returns
    -------
    sorting_prediction : Series
        Sorted list of predictions.
    sorting_label : Series
        Sorted list of labels.
    '''
    
    fitted_model, prediction = classifier
    sorting_prediction = pd.Series(prediction[x_predict]).sort_values(ascending=False)
    sorting_label = pd.Series(fitted_model[sorting_prediction.index[0]][28*28:]).sort_values(ascending=False)
    return sorting_prediction, sorting_label

def listing_and_sorting(x):
    '''
    Function to make a list of x and to order it.

    Parameters
    ----------
    x : 
        Input variable.

    Returns
    -------
    x_t_sort : 
        Sorted list of the input.

    '''
    x_t_list = x.tolist()
    x_t_sort = x_t_list.sort()
    return x_t_sort