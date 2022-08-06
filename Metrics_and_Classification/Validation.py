# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:23:51 2022

@author: Simone Fumagalli
"""

from sklearn.model_selection import StratifiedKFold
import Classification

def __del__(self):
    print(" ")
    
def check_number_training(clf_times):
        n_splits = getattr(val_sets, 'n_splits')
        if clf_times > n_splits:
            raise Exception("The number of times to operate the classification"
                            " must be lower or equal to "
                            "the number of splitting")
            return None
        else:
            return clf_times  

def val_sets(x_train,y_train, n_splits:int = 2):
    setattr(val_sets, 'n_splits', n_splits)
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

def val_classification(model, validation_sets, clf_times: int):
    
    x_train, x_test, y_train, y_test = validation_sets
    classifiers = []
    
    clf_times = check_number_training(clf_times)
    
    for i in range(clf_times):
        classifier = Classification.clf(model, x_train[i], x_test[i],\
                                        y_train[i], y_test[i])
        classifiers.append(classifier)
        __del__(classifier)
    return classifiers

def val_metrics(val_classifiers, validation_sets, t_l = False):
    x_train, x_test, y_train, y_test = validation_sets
    validation_metrics = []
    for i in range (len(val_classifiers)):
        validation_metrics.append(Classification.Metrics(val_classifiers[i], y_test[i]))
    return validation_metrics


