# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:42:26 2022

@author: Simone Fumagalli
"""
from BCM_classification import variables, fitting, predict, accuracy
from sklearn.datasets import fetch_openml
import pandas as pd

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

x_train = variables(X, y, 0)
y_train = variables(X, y, 1)
x_test = variables(X, y, 2)
y_test = variables(X, y, 3)

def test_variables():
    
    assert len(x_train) == len(X)*(7/8)
    assert len(y_train) == len(y)*(7/8)
    assert len(x_test) == len(X)*(1/8)
    assert len(y_test) == len(y)*(1/8)
        
model = fitting(x_train,y_train)

def test_model():
    
    assert isinstance(model.outputs, int)
    assert isinstance(model.num_epochs, int)
    assert isinstance(model.batch_size, int)

def test_predict():
        
        predict(x_test,y_test,model, x_predict)
        assert type(x_predict) == int
        assert len(labels) == model.outputs
        
        
    
        assert type(x_predict) == float or str


def test_accuracy():
    
    assert isinstance(y_values, list)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_values) == len(y_true) == len(y_test)



        





