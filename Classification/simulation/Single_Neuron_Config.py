# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:30:44 2022

@author: Simone Fumagalli
"""

from sklearn.datasets import fetch_openml

from plasticity.model.optimizer import Adam, Adamax, SGD, Momentum,\
                                       NesterovMomentum, Adagrad, \
                                       Adadelta, RMSprop

from plasticity.model.weights import GlorotNormal, HeNormal, Uniform, LecunUniform,\
                                     GlorotUniform, LecunNormal, HeUniform,\
                                     Orthogonal, TruncatedNormal

from BCM_classification import Classification

X, y = fetch_openml(name='Fashion-MNIST', version=1,  data_id=None, return_X_y=True)

out = [100,100,100,100,100,100,100,100]
epoc = [12000,5000,5000,5000,30,10,10,10]
bat = [61250,61250,61250,61250,1700,1000,1000,1000]
wei = [GlorotUniform(),GlorotNormal(),GlorotNormal(),GlorotNormal(),GlorotNormal(),GlorotNormal(),GlorotNormal(),GlorotNormal()]
num = 8
opti = [SGD(lr=1e-1), Adamax(), Momentum(),\
       NesterovMomentum(lr=1e-2),SGD(lr=3e-2),Adagrad(lr=1e-2),Adadelta(lr=1e-2),\
       RMSprop(lr=1e2)]
strenght = [-0.0005,-0.001,-0.005,-0.005,-0.005,-0.001,-0.001,-0.001]
clas = Classification(out, bat, wei, num, opti, strenght, epoc)

x_norm, y_resh = clas.Variable_Reshape(X, y)
x_train,x_test,y_train,y_test = clas.Variable_Split(x_norm, y_resh, 1./8)

model = clas.modellization(0)
model_fit = clas.fitting(x_train, model)
neur = clas.neurons_configuration(model_fit)