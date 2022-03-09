# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:42:26 2022

@author: Simone Fumagalli
"""
import numpy as np
import pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

def variables(i):
    
    X_1=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    #Transformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    
    x_train, x_test, y_train, y_test = \
    train_test_split(X_1, y_transform, test_size=1./8, random_state=42)
    r = [x_train, y_train, x_test, y_test]
    return r[i], len(r[i])
    

def test_x_train():
    assert variables(0) == len(X)*(7/8)

def test_y_train():
    assert variables(1) == len(y)*(7/8)

def test_x_test():
    assert variables(2) == len(X)*(1/8)

def test_y_test():
    assert variables(3) == len(X)*(1/8)
    


