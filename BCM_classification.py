from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal, Normal
import numpy as np
import pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
    
X*=1./255 # Normalization of the X inputs in range [0,1]

