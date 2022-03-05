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

#Transformation of the y vector
y_int = y.astype('int')
y_reshape = y_int.values.reshape(-1,1)
y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)

x_train, x_test, y_train, y_test = train_test_split(X, y_transform, test_size=1./8, random_state=42)

#FITTING
model = BCM(outputs=1000, num_epochs=10, optimizer=Adam(lr=4e-2), interaction_strength=0.,
            weights_init=GlorotNormal(),
            activation='Relu', batch_size=10000)
model.fit(x_train, y_train)

