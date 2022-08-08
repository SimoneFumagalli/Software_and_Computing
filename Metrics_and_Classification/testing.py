# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:26:25 2022

@author: Simone Fumagalli
"""

from sklearn.datasets import fetch_openml
import numpy as np
from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
from sklearn.model_selection import train_test_split
import os
import sys

path = os.getcwd()
filepath = os.path.dirname
sys.path.append(path)

import Classification
import Validation

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

model = BCM(outputs=100, num_epochs=1, optimizer=Adam(lr=3e-2), 
            interaction_strength=0.,
            weights_init=GlorotNormal(),
            activation='Relu', batch_size=60000)

#Reshaping of the dataset
X_norm, y_transform = Classification.Variable_Reshape(X, y)
#Splitting of the dataset
x_train, x_test, y_train, y_test = \
train_test_split(X_norm, y_transform, test_size=1./8)

def test_Reshape():
    '''
    Function to test the Variable_Reshape function 
    '''
    assert type(X_norm) == type(X) #checking the type of x_norm
    assert isinstance(y_transform, np.ndarray) #checking the type of y_transformed
    
