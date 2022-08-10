# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:30:24 2022

@author: Simone Fumagalli
"""

from plasticity.model import BCM
from plasticity.model.optimizer import Adam, SGD, Adamax, Momentum
from plasticity.model.weights import Uniform, GlorotUniform, HeNormal, Normal,GlorotNormal
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


import Classification
import Validation

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

X_norm, y_transform = Classification.Variable_Reshape(X,y)

x_train, x_test, y_train, y_test = \
train_test_split(X_norm, y_transform, test_size=1./8)

model = BCM(outputs=1000, num_epochs=20, optimizer=Adam(lr=4e-2), 
            interaction_strength=0.,
            weights_init=Normal(),
            activation='Relu', batch_size=4000)
# Classification
classifier = Classification.clf(model, x_train, x_test, y_train, y_test)

# Plot of the classifications
for i in range (4):
    br =Classification.plot_best_result(x_test, y_test, classifier, i)

# Metrics
single_label_metric = Classification.Metrics(classifier, y_test,False)
ten_label_metric = Classification.Metrics(classifier,y_test, True)

# Validation
val_set = Validation.val_sets(x_train, y_train, 8)
val_clf = Validation.val_classification(model, val_set, 4)
val_single_label_metric = Validation.val_metrics(val_clf, val_set, False)
val_ten_label_metric = Validation.val_metrics(val_clf, val_set, True)
