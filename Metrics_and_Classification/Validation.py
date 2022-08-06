# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:23:51 2022

@author: Simone Fumagalli
"""

from sklearn.model_selection import StratifiedKFold
import Metrics_and_Classification

def validation_sets(x_train,y_train, n_splits:int = 2):
    
    skf = StratifiedKFold(n_splits)

    x_train_val, x_test_val, y_train_val, y_test_val = [],[],[],[]
    
    for train_index, test_index in skf.split(x_train,y_train.argmax(axis=1)):
        x_train_, x_test_ = x_train.values[train_index], \
                                              x_train.values[test_index]
        y_train_, y_test_ = y_train[train_index], \
                                              y_train[test_index]
                                              
        x_train_val.append(x_train_)
        x_test_val.append(x_test_)
        y_train_val.append(y_train_)
        y_test_val.append(y_test_)
        
    return x_train_val, x_test_val, y_train_val, y_test_val