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
from sklearn.model_selection import StratifiedKFold

def __del__(self):
    print(" ")

def Variable_Reshape(X,y):
    X_norm=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    # Transformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    return X_norm, y_transform

def clf(model, x_train, x_test, y_train, y_test):
    
    #Checking the batch size
    #model.batch_size = checking_batch(model,y_train)
    # Use of fitting function
    fitted_model = model.fit(x_train,y_train) 
    # Use of prevision function
    prediction = model.predict(x_test, y = np.zeros_like(y_test))
    
    return fitted_model, prediction

def top_ten(classifier):
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

def resulting_labels(classifier):
    fitted_model, prediction = classifier
    labels = [fitted_model.weights[np.argmax(x)][28*28:].argmax() 
             for x in prediction]
    

    # Use of top_ten function
    top_10_array = top_ten(classifier)
    
    return labels, top_10_array

def plot_best_result(x_test, y_test, classifier, resulting_labels, x_predict):
    fitted_model, prediction = classifier
    nc = np.amax(np.abs(fitted_model.weights))
    label = resulting_labels[0][x_predict]
    top_ten_label = resulting_labels[1][x_predict]
    best_result = fitted_model.weights[np.argmax(prediction[x_predict])][:28*28].reshape(28, 28)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    ax1.set_title('Image from the Fashion-MNIST dataset: {:d}'.format(y_test[x_predict].argmax()))
    ax1.imshow(x_test.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
    
    ax2.set_title('Prediction using BCM: {:d}'.format(label))
    ax2.imshow(best_result, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')
    
    fig.text(0.5, 0.24, 'Ten best neuron result: {}'.format(top_ten_label),
             horizontalalignment='center', fontsize = 14)
    
    return None


    