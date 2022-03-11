# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:42:26 2022

@author: Simone Fumagalli
"""
from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
import numpy as np
import pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
X_1=X*(1./255) # Normalization of the X inputs in range [0,1]

#Transformation of the y vector
y_int = y.astype('int')
y_reshape = y_int.values.reshape(-1,1)
y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)

x_train, x_test, y_train, y_test = \
train_test_split(X_1, y_transform, test_size=1./8, random_state=42)

def test_variables():
    
    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, pd.Series), "y must be a Series type"
    assert X.shape[0] == y.shape[0], "they must have the same number of rows"
    assert len(x_train) == len(X)*(7/8)
    assert len(y_train) == len(y)*(7/8)
    assert len(x_test) == len(X)*(1/8)
    assert len(y_test) == len(y)*(1/8)
        

model = BCM(outputs=1000, num_epochs=10, optimizer=Adam(lr=4e-2), 
                        interaction_strength=0.,
                        weights_init=GlorotNormal(),
                        activation='Relu', batch_size=10000)
    
model.fit(x_train, y_train)

def test_model():
    assert isinstance(model.outputs, int)
    assert isinstance(model.num_epochs, int)
    assert isinstance(model.batch_size, int)

def test_predict():
    x_predict=0
    if type(x_predict) == int and x_predict < len(y_test):
        
        predict = model.predict(x_test.values[x_predict].reshape(1, -1), \
                                y=np.zeros_like(y_test[x_predict].reshape(1, -1)))
    
        # select the neuron connection with the highest response
        highest_response = model.weights[np.argmax(predict)][:28*28].reshape(28, 28)
        
        # collecting the predicted labels
        labels = model.weights[:, 28*28:].argmax(axis=1)
        
        #sorting the label with the highest response
        
        nc = np.amax(np.abs(model.weights))
        
        
        assert type(x_predict) == int
        assert type(nc) == np.float64
        assert len(labels) == model.outputs
        
        
    else:
        assert type(x_predict) == float or str


def test_accuracy():
    testing = model.predict(x_test, y_test)

    y_values = [model.weights[np.argmax(x)][28*28:].argmax() for x in testing]

    y_true = y_test.argmax(axis=1)
    y_pred = np.asarray(y_values)
    
    assert isinstance(y_values, list)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_values) == len(y_true) == len(y_test)



        





