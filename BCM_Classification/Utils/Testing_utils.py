# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:42:16 2022

@author: Simone Fumagalli
"""

import numpy as np
import pandas as pd
import Classification

def ylabels(classifier):
    y_lab = [classifier[0][np.argmax(x)][28*28:].argmax() 
              for x in classifier[1]]
    return y_lab

def top_10_labels(classifier):
    top_10 = Classification.top_ten(classifier)
    ten_result = [top_10[x][0][0] for x in range(len(top_10))]
    return ten_result

def reshaping_weights(classifier):
    fitted_model = classifier[0]
    new_shape = len(fitted_model)*len(fitted_model[0])
    reshape_abs_weights = (np.abs(fitted_model)).reshape(new_shape)
    sorting = sorted(reshape_abs_weights, reverse = True)
    return sorting

def sorting(classifier, x_predict):
    fitted_model, prediction = classifier
    sorting_prediction = pd.Series(prediction[x_predict]).sort_values(ascending=False)
    sorting_label = pd.Series(fitted_model[sorting_prediction.index[0]][28*28:]).sort_values(ascending=False)
    return sorting_prediction, sorting_label

def listing_and_sorting(x):
    x_t_list = x.tolist()
    x_t_sort = x_t_list.sort()
    return x_t_sort