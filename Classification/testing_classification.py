# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:42:26 2022

@author: Simone Fumagalli
"""
from Fai_qualcosa_di_strano import variables, fitting, predict, accuracy
from sklearn.datasets import fetch_openml
import numpy as np


X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

x_train = variables(X, y, 0)
y_train = variables(X, y, 1)
x_test = variables(X, y, 2)
y_test = variables(X, y, 3)

def test_variables():
    """
    It tests the dimensions of the variables obtained.

    """
    assert len(x_train) == len(X)*(7/8)
    assert len(y_train) == len(y)*(7/8)
    assert len(x_test) == len(X)*(1/8)
    assert len(y_test) == len(y)*(1/8)
        
model = fitting(x_train,y_train)

def test_model():
    """
    It tests the type of input introduced in the BCM model.
    """
    assert isinstance(model.outputs, int)
    assert isinstance(model.num_epochs, int)
    assert isinstance(model.batch_size, int)
    
def test_predict():
    """
    It tests the correct input of the x used for the prediction, controlling
    also that the predicted labels has the same lenght of the outputs considered
    in the BCM model.
    """
    x_predict = 0
    labels = predict(x_test,y_test,model, x_predict)
    if type(x_predict) == int and x_predict < len(y_test):
        assert type(x_predict) == int
        assert x_predict < len(y_test)
        assert len(labels) == model.outputs
        
    else:
        assert type(x_predict) == float or str or x_predict > len(y_test)

def test_accuracy():
    """
    It tests the types and lengths of the variables used in the accuracy function.
   
    """
    acc = accuracy(x_test,y_test,model)
    y_types = acc[0] #composed by the types of y_values, y_true, y_pred
    assert isinstance(y_types[0], list)
    assert isinstance(y_types[1], np.ndarray)
    assert isinstance(y_types[2], np.ndarray)
    y_length = acc[1] #composed by the lenghts of y_values, y_true, y_pred
    assert len(y_length[0]) == len(y_length[1]) == len(y_length[2])




        




