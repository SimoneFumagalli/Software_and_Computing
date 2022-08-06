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

