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

#PREDICTION

def predict(X,y,x_predict):
    predict = model.predict(x_test.values[x_predict].reshape(1, -1), y=np.zeros_like(y_test[x_predict].reshape(1, -1)))

# select the neuron connection   with the highest response
    highest_response = model.weights[np.argmax(predict)][:28*28].reshape(28, 28)

    nc = np.amax(np.abs(model.weights))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    ax1.set_title('Image from the MNIST dataset:{:d}'.format(y_test[x_predict].argmax()))
    ax1.imshow(x_test.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')

    ax2.set_title('Prediction using BCM')
    ax2.imshow(highest_response, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')

predict(x_test, y_test, x_predict=0)