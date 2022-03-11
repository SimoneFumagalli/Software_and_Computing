# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:12:21 2022

@author: Simone Fumagalli
"""

from BCM_classification import variables, fitting, predict, accuracy
from sklearn.datasets import fetch_openml

#Importing the MNIST Dataset
X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

#Splitting of the MNIST Dataset
x_train = variables(X, y, 0)
y_train = variables(X, y, 1)
x_test = variables(X, y, 2)
y_test = variables(X, y, 3)

#Training of the model usign the train variables
model = fitting(x_train, y_train)

#Prediction of the image 0 using the test variables
predict(x_test,y_test,model,x_predict=0)

# Testing the accuracy of the prediction
accuracy(x_test,y_test,model)




        
