# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:23:51 2022

@author: Simone Fumagalli
"""

from sklearn.model_selection import StratifiedKFold
from Metrics_and_Classification import Classification

def __del__(self):
    print(" ")
    
def check_number_training(clf_times: int):
    '''
    Function to control how many times the classification is performed on the
    validation set.

    Parameters
    ----------
    clf_times : int
        How many times the clf function is performed.

    Raises
    ------
    Exception
        Exception occurred when the number of clf_times is higher the number of
        splits.

    Returns
    -------
    clf_times : int
        How many times the clf function is performed.

    '''
    n_splits = getattr(val_sets, 'n_splits')
    if clf_times > n_splits:
        raise Exception("The number of times to operate the classification"
                        " must be lower or equal to "
                        "the number of splitting")
        return None
    else:
        return clf_times  

def val_sets(x_train,y_train, n_splits:int = 2):
    '''
    Generation of the validation set using the train one.

    Parameters
    ----------
    x_train : Dataframe
        Dataframe of the input x_train used for training the model.
    y_train : ndarray
        Array of input y used for training the model.
    n_splits : int, optional
        Number of times the train set is splitted in different validation
        sets. The default is 2.

    Returns
    -------
    x_train_val : Dataframe
        Dataframe of the validation set extracted from the input x_train.
        Used for training the model.
    x_test_val : Dataframe
        Dataframe of the validation set extracted from the input x_train. 
        Used for the prediction.
    y_train_val : ndarray
        Array of the validation set extracted from the y_train set.
        Used for training the model.
    y_test_val : ndarray
        Array of the validation set extracted from the y_train set.
        Used for the prediction.

    '''
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
    '''
    Function to operate the classification using the validation sets.

    Parameters
    ----------
    model : BCM model
        Model used for the classification..
    validation_sets : array
        Array containing the sets used for training and making the predictions.
    clf_times : int
        Number of times the classification is performed.

    Returns
    -------
    classifiers : array
        Array containing the fitted models and the predictions.

    '''
    x_train, x_test, y_train, y_test = validation_sets
    classifiers = []
    
    clf_times = check_number_training(clf_times)
    
    for i in range(clf_times):
        classifier = Classification.clf(model, x_train[i], x_test[i],\
                                        y_train[i], y_test[i])
        classifiers.append(classifier)
        __del__(classifier)
    return classifiers

def val_metrics(val_classifiers, validation_sets, ten_label_type = False):
    '''
    Function to visualize the metrics of each validation set used.

    Parameters
    ----------
    val_classifiers : array
        Array containing the fitted models and the predictions.
    validation_sets : array
        array containing the different validation sets.
    ten_label_type : bool, optional
        Used to choose the kind of label for the metrics. The default is False.

    Returns
    -------
    validation_metrics : array
        Array containing the information about the accuracy and the result of 
        classification_report functions.

    '''
    x_train, x_test, y_train, y_test = validation_sets
    validation_metrics = []
    for i in range (len(val_classifiers)):
        validation_metrics.append(Classification.Metrics(val_classifiers[i], y_test[i], ten_label_type))
    return validation_metrics


