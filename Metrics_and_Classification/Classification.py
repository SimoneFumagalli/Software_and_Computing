# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:24:17 2022

@author: Simone Fumagalli
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:09:41 2022

@author: Simone Fumagalli
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pylab as plt
from collections import Counter
from plasticity.utils import view_weights

def __del__(self):
    print(" ")

def checking_batch_size(model, y_train):
    '''
    Function to control the batch_size parameter of the BCM model.

    Parameters
    ----------
    model : BCM model
        Model used for the classification.
    y_train : array
        Ndarray containing the list of labels used for training the model.

    Returns
    -------
    int
        The value of the batch_size parameter.

    '''
    if model.batch_size > len(y_train):
        print('The dimension of the batch',model.batch_size,'is much higher '
              'than that of the samples',len(y_train),'.\
              This last has been considered.')
        model.batch_size = len(y_train)
    else:
        model.batch_size = model.batch_size
    return model.batch_size

def Variable_Reshape(X,y):
    '''
    Function to reshape the input variables X and y

    Parameters
    ----------
    X : Dataframe
        Dataframe of the pixels composing the images.
    y : Series
        Series containing the value of the labels associated to the images.

    Returns
    -------
    X_norm : Dataframe
        Dataframe of the X input reshaped.
    y_transform : ndarray
        Ndarray containing the reshaped y using the OneHot encoding.

    '''
    X_norm=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    # Transformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    return X_norm, y_transform

def clf(model, x_train, x_test, y_train, y_test, plot_view_weights = False):
    '''
    Classification function in which are implemented the fit and predict function
    of the plasticity package.

    Parameters
    ----------
    model : BCM model
        Model used for the classification.
    x_train : Dataframe
        Dataframe of the input X used for training the model.
    x_test : Dataframe
        Dataframe of the input X to make the predictions.
    y_train : ndarray
        Array of input y used for training the model.
    y_test : ndarray
        Array of input y to make the predictions.
    view_weights: bool, optional
        Term indicating if the function is used to classify or to show the
        configuration of the neuron.
    Returns
    -------
    fitted_model : BCM model
        Trained model.
    prediction : Array
        Array containing the predictions of the model.
    None.
    '''
    if plot_view_weights == False:
        #Checking the batch size
        model.batch_size = checking_batch_size(model,y_train)
        # Use of fitting function
        fitted_model = model.fit(x_train,y_train) 
        # Use of prevision function
        prediction = model.predict(x_test, y = np.zeros_like(y_test))
        return fitted_model, prediction
    
    else:
        fitted_model = model.fit(x_train)
        view_weights(fitted_model.weights, dims=(28,28))
        return None
    
def top_ten(classifier):
    '''
    Function to select the ten best neurons and their labels for each image to
    predict.

    Parameters
    ----------
    classifier : array
        Array containing the fitted model and the predictions.

    Returns
    -------
    top_10 : array
        Array containing the labels of the top ten neurons for each prediction.

    '''
    top_10 = []
    fitted_model, prediction = classifier
    labels = fitted_model.weights[:,28*28:].argmax(axis=1) # Labels of the 100 neurons
    
    for x in prediction:
         
        # Union of prevision score and neuron
        sorting = sorted(zip(x, labels),\
                         key=lambda x : x[0], reverse=True)[:10]
        sorting = [x[1] for x in sorting]    
        # Counting how many of the ten neurons give the same label as result
        counter_lab = (Counter(sorting).most_common())
        top_10.append(counter_lab)
    
    return top_10


def plot_best_result(x_test, y_test, classifier, x_predict:int):
    '''
    Function to plot the result of the classification.

    Parameters
    ----------
    x_test : Dataframe
        Dataframe of the input X to make the predictions.
    y_test : ndarray
        Array of input y to make the predictions.
    classifier : array
        Array containing the fitted model and the predictions.
    resulting_labels : array
        Array containing two arrays: one of the labels obtained using the info
        from just the best neuron and the other is composed by the labels obtained
        using the information from the top ten neurons.
    x_predict : int
        Int corresponding to the image to which the function will show the result
        of the classification.

    Returns
    -------
    None.

    '''
    fitted_model, prediction = classifier
    nc = np.amax(np.abs(fitted_model.weights))
    label = [fitted_model.weights[np.argmax(x)][28*28:].argmax() 
              for x in prediction][x_predict]
    top_ten_label = top_ten(classifier)[x_predict]
    best_result = fitted_model.weights[np.argmax(prediction[x_predict])][:28*28].reshape(28, 28)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    ax1.set_title('Image from the Fashion-MNIST dataset: {:d}'.format(y_test[x_predict].argmax()))
    ax1.imshow(x_test.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
    
    ax2.set_title('Prediction using BCM: {:d}'.format(label))
    ax2.imshow(best_result, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')
    
    fig.text(0.5, 0.24, 'Ten best neuron result: {}'.format(top_ten_label),
             horizontalalignment='center', fontsize = 14)
    
    return None

def Metrics(classifier, y_test, ten_label_type:bool = False):
    '''
    Function to show the metrics of the classification using the 
    classification_report function.

    Parameters
    ----------
    classifier : array
        Array containing the fitted model and the predictions.
    y_test : ndarray
        Array of input y to make the predictions.
    ten_label_type : bool, optional
        Set True if want to consider the ten labels. The default is False.

    Returns
    -------
    accuracy_values : float
        Value of the accuracy result.
    performance : dict
        Dict containing the values obtained from the classification_report function.

    '''
    fitted_model, prediction = classifier
    if ten_label_type == True:
        ten_labels = top_ten(classifier)
        y_labels = [ten_labels[x][0][0] for x in range(len(ten_labels))]
    else:
        y_labels = [fitted_model.weights[np.argmax(x)][28*28:].argmax() 
                  for x in prediction]
    y_test = y_test.argmax(axis=1)
    
    performance = classification_report(y_test, y_labels, 
                                         zero_division = 0,
                                         output_dict = True)
    print(classification_report(y_test, y_labels,
                                   zero_division = 0))
    
    accuracy_values = performance['accuracy']
    
    return accuracy_values, performance

