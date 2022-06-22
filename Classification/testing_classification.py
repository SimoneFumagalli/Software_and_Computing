# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:26:25 2022

@author: Simone Fumagalli
"""

from sklearn.datasets import fetch_openml
import numpy as np

from plasticity.model.optimizer import Adam, Adamax, SGD, Momentum,\
                                       NesterovMomentum, Adagrad, Adadelta,\
                                       RMSprop
from plasticity.model.weights import GlorotNormal, HeNormal, Uniform,\
                                     LecunUniform, GlorotUniform, LecunNormal,\
                                     HeUniform, Orthogonal, TruncatedNormal


from untitled16 import Classia

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

out = [100,100]
epoc = [1,1,5000,5000,30,10,10,10]
bat = [30000,60000,61250,61250,1700,1000,1000,1000]
wei = [GlorotUniform(),GlorotNormal(),GlorotNormal(),GlorotNormal(),\
       GlorotNormal(),GlorotNormal(),GlorotNormal(),GlorotNormal()]
num = 2
opti = [Adam(lr=1e-3), Adamax(), Momentum(),\
        NesterovMomentum(lr=1e-2),SGD(lr=3e-2),Adagrad(lr=1e-2),Adadelta(lr=1e-2),\
        RMSprop(lr=1e-2)]
strength = [-0.001,-0.001,-0.005,-0.005,-0.005,-0.001,-0.001,-0.001]

go = Classia(out, bat, wei, num, opti, strength, epoc)

#Reshaping of the dataset
x_norm, y_transformed = go.Variable_Reshape(X, y)
#Splitting of the dataset
x_train, x_test, y_train, y_test = go.Variable_Split(x_norm, y_transformed, 1./8)

def test_Reshape():
    '''
    Function to test the Variable_Reshape function 
    '''
    assert type(x_norm) == type(X) #checking the type of x_norm
    assert isinstance(y_transformed, np.ndarray) #checking the type of y_transformed
    

def test_Split():    
    '''
    Function to test the Variable_Splitting function
    '''
    #Checking the length of the variables
    assert len(x_train) == len(X)*(7/8)
    assert len(y_train) == len(y)*(7/8)
    assert len(x_test) == len(X)*(1/8)
    assert len(y_test) == len(y)*(1/8)

def test_modellization():
    '''
    Function to test the modellization function
    '''
    model = go.modellization(0)
    
    #Checking that the variables of modellization function equal the
    #element of the arrays of output, batch and so on.
    
    assert model[0].outputs == out[0]
    assert model[0].num_epochs == epoc[0]
    assert model[0].batch_size == bat[0]
    assert model[0].weights_init == wei[0]
    assert model[0].optimizer == opti[0]
    assert model[0].interaction_strength == strength[0]

def test_fitting():
    '''
    Function to test the fitting funtion.
    '''
    
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)

    model_a = go.modellization(0)
    model_b = go.modellization(1)
    
    fitt_a = go.fitting(x_train, model_a, y_train)
    fitt_b = go.fitting(x_train, model_b)
    
    #Checking the length of the labels of the neurons
    assert len(fitt_a[0].weights[:,28*28:].argmax(axis=1)) == out[0]
    #Checking the difference between the two models
    assert fitt_a[0] != fitt_b[0]
    assert fitt_b[0] == model_b[0]

def test_prevision():
    '''
    Function to test the prevision function.
    '''
    out = [100]
    epoc = [1]
    bat = [30000]
    wei = [GlorotUniform()]
    num = 1
    opti = [Adam(lr=1e-3)]
    strength = [-0.001]
    go = Classia(out, bat, wei, num, opti, strength, epoc)
   
    model = go.modellization(0)
    
    fitt = go.fitting(x_train, model, y_train)
    
    prediction = go.prevision(x_test, y_test, fitt)
    
    #Checking the output of the prediction function to be equal to the y_test
    assert len(prediction) == len(y_test)
    
    #Checking the length of output of each prevision
    for i in range(len(prediction)):
        assert len(prediction[i]) == out[0]

def test_best_neuron():
    '''
    Function to test the best_neuron function.
    '''
    out = [100]
    epoc = [1]
    bat = [30000]
    wei = [GlorotUniform()]
    num = 1
    opti = [Adam(lr=1e-3)]
    strength = [-0.001]
    go = Classia(out, bat, wei, num, opti, strength, epoc)

    model = go.modellization(0)
    
    fitt = go.fitting(x_train, model, y_train)
    
    prediction = go.prevision(x_test, y_test, fitt)
    
    labels = go.best_neuron(fitt, prediction)
    
    #Checking the length of the labels
    assert len(labels) == len(y_test)

def test_top_ten():
    '''
    Function to test the top_ten function.
    '''
    out = [100]
    epoc = [1]
    bat = [30000]
    wei = [GlorotUniform()]
    num = 1
    opti = [Adam(lr=1e-3)]
    strength = [-0.001]
    go = Classia(out, bat, wei, num, opti, strength, epoc)

    model = go.modellization(0)
    
    fitt = go.fitting(x_train, model, y_train)
    
    prediction = go.prevision(x_test, y_test, fitt)
    
    top_ten_labels = go.top_ten(fitt, prediction)
    
    #Checking the general length of the top_ten_labels
    assert len(top_ten_labels) == len(y_test)
    
    #Checking the length of each prevision using ten neurons
    for i in range(len(top_ten_labels)):
        assert (np.sum(top_ten_labels[i],0)[1]) == 10

def test_checking_y_size():
    '''
    Function to test the checking_y_size function
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    clf_0 = go.clf(0, x_train, x_test, y_train, y_test)
    clf_1 = go.clf(1, x_train, x_test, y_train, y_test, 1./8)

    check_0 = go.checking_y_size(y_test, clf_0)
    check_1 = go.checking_y_size(y_test, clf_1)
    
    #Checking the length of tue output of the two functions
    assert check_0.all() == clf_0[0].all() == y_test.all()
    assert check_1.all() == clf_1[0].all()

def test_checking_y_labels():
    '''
    Function to test the checking_y_labels function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    clf_0 = go.clf(0, x_train, x_test, y_train, y_test)
    clf_1 = go.clf(1, x_train, x_test, y_train, y_test, 1./8)
    
    check_0_f = go.checking_y_labels(clf_0, False)
    check_0_t = go.checking_y_labels(clf_0, True)
    
    check_1_f = go.checking_y_labels(clf_1, False)
    check_1_t = go.checking_y_labels(clf_1, True)
    
    #Checking the kind of labels used
    assert check_0_f == clf_0[1]
    assert check_1_f == clf_1[1]
    
    #Checking the length of the different outputs
    assert len(check_0_f) == len(check_0_t) == len(clf_0[1]) == len(y_test)
    assert len(check_1_f) == len(check_1_t) == len(clf_1[1])    
    
def test_checking_batch():
    '''
    Function to test the checking_batch function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,70000]
    wei = [GlorotUniform(),GlorotNormal(),]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    check_0 = go.checking_batch(0, y_train)
    check_1 = go.checking_batch(1, y_train)
    
    #Checking the batch size, with and without validation splitting
    assert check_0 == bat[0]
    assert check_1 == len(y_train)
    
def test_clf():
    '''
    Function to test the clf function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    clf_0 = go.clf(0, x_train, x_test, y_train, y_test)
    clf_1 = go.clf(1, x_train, x_test, y_train, y_test, 1./8)
    
    #Checking the two functions to have the same number of outputs
    assert len(clf_0) == len(clf_1)
    
    #Checking the dimensions of the labels to be different
    assert len(clf_0[0]) != len(clf_1[0])
    
    #Checking the difference between the models used
    assert clf_0[3] != clf_1[3]
    
    #Checking the length of the outputs of the first model
    assert len(clf_0[0]) == len(clf_0[1]) == len(clf_0[2]) \
           == len(clf_0[4])== len(y_test)
    
    #Checking the lenght of the outputs of the second model
    assert len(clf_1[1]) == len(clf_1[2]) == len(clf_1[4]) \
           == len(clf_1[0])
    
    #Checking the length of each set of ten labels
    for j in range(len(clf_0[2])):
            assert (np.sum(clf_0[2][j],0)[1]) == 10
    
    for j in range(len(clf_1[2])):
            assert (np.sum(clf_1[2][j],0)[1]) == 10


def test_single_Metric():
    '''
    Function to test the single_Metric function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 1
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    clf_0 = go.clf(0, x_train, x_test, y_train, y_test)
    clf_1 = go.clf(1, x_train, x_test, y_train, y_test, 1./8)
    
    s_p_0_f = go.single_Metric(0, y_test, clf_0, False)
    s_p_0_t = go.single_Metric(0, y_test, clf_0, True)
    
    s_p_1_f = go.single_Metric(1, y_test, clf_1, False)
    s_p_1_t = go.single_Metric(1, y_test, clf_1, True)
    
    #Checking the type of each of single performance output
    assert isinstance(s_p_0_f[1],str)
    assert isinstance(s_p_0_t[1],str)
    assert isinstance(s_p_1_f[1],str)
    assert isinstance(s_p_1_t[1],str)
    
    #Checking the type of each of single performance output
    assert isinstance(s_p_0_f[0][0],float)
    assert isinstance(s_p_0_t[0][0],float)
    assert isinstance(s_p_1_f[0][0],float)
    assert isinstance(s_p_1_t[0][0],float)

def test_multiple_clf():
    '''
    Function to test the classification of multiple models
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    multi_0 = go.multiple_clf(x_train, x_test, y_train, y_test)
    multi_1 = go.multiple_clf(x_train, x_test, y_train, y_test,1./8)
    
    assert len(multi_0) == len(multi_1)
    
    for i in range(num):
        assert len(multi_0[i][0]) != len(multi_1[i][0])
        assert multi_0[i][3] != multi_1[i][3]
        assert len(multi_0[i][0]) == len(multi_0[i][1]) == len(multi_0[i][2]) \
               == len(multi_0[i][4])== len(y_test)
        assert len(multi_1[i][0]) == len(multi_1[i][1]) == len(multi_1[i][2]) \
               == len(multi_1[i][4])
        
        for j in range(len(multi_0[i][2])):
                assert (np.sum(multi_0[i][2][j],0)[1]) == 10
        
        for j in range(len(multi_1[i][2])):
                assert (np.sum(multi_1[i][2][j],0)[1]) == 10

        
def test_Metrics():
    '''
    Function to test the Metrics function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 2
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]

    go = Classia(out, bat, wei, num, opti, strength, epoc)
    
    multiple_clf_0 = go.multiple_clf(x_train, x_test, y_train, y_test)
    multiple_clf_1 = go.multiple_clf(x_train, x_test, y_train, y_test, 1./8)
    
    metric_0 = go.Metrics(y_test, multiple_clf_0, ten_label = False)
    metric_1 = go.Metrics(y_test, multiple_clf_1, ten_label = True)
    
    #Checking the type of the outputs
    for i in range(num):
        assert isinstance(metric_0[i][0],float)
        assert isinstance(metric_1[i][0],float)
    
def test_best_result():
    '''
    Function to test the best_result function.
    '''
    out = [100,100]
    epoc = [1,1]
    bat = [30000,60000]
    wei = [GlorotUniform(),GlorotNormal()]
    num = 1
    opti = [Adam(lr=1e-3), Adamax()]
    strength = [-0.001,-0.001]
    
    go = Classia(out, bat, wei, num, opti, strength, epoc)

    clf_0 = go.clf(0, x_train, x_test, y_train, y_test)
    clf_1 = go.clf(1, x_train, x_test, y_train, y_test, 1./8)
    
    b_r_0 = [go.best_result(x_test, y_test, clf_0, i) \
           for i in range(10)]
    
    b_r_1 = [go.best_result(x_test, y_test, clf_1, i) \
           for i in range(10)]
    
    #Checking the type of the output of the best_result functions
    for j in range (len(b_r_0)):
        assert isinstance(b_r_0[j][0], np.int64)
        assert isinstance(b_r_0[j][1], list)
        assert (np.sum(b_r_0[j][1],0)[1]) == 10
    
    for j in range (len(b_r_1)):
        assert isinstance(b_r_1[j][0], np.int64)
        assert isinstance(b_r_1[j][1], list)
        assert (np.sum(b_r_1[j][1],0)[1]) == 10
