from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotNormal
import numpy as np
import pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

X_1=X*(1./255) # Normalization of the X inputs in range [0,1]

#Transformation of the y vector
y_int = y.astype('int')
y_reshape = y_int.values.reshape(-1,1)
y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)

x_train, x_test, y_train, y_test = \
train_test_split(X_1, y_transform, test_size=1./8, random_state=42)


model = BCM(outputs=1000, num_epochs=10, optimizer=Adam(lr=4e-2), 
                interaction_strength=0.,
                weights_init=GlorotNormal(),
                activation='Relu', batch_size=10000)
    
model.fit(x_train, y_train)
   

#PREDICTION

def predict(X,y,x_predict):
    
    if type(x_predict) == int and x_predict < len(y_test):
        predict = model.predict(x_test.values[x_predict].reshape(1, -1), \
                                y=np.zeros_like(y_test[x_predict].reshape(1, -1)))
    
        # select the neuron connection with the highest response
        highest_response = model.weights[np.argmax(predict)][:28*28].reshape(28, 28)
        
        # collecting the predicted labels
        labels = model.weights[:, 28*28:].argmax(axis=1)
        
        #sorting the label with the highest response
        predicted_label = sorted(zip(predict.ravel(), labels), \
                                 key=lambda x : x[0], reverse=True)[:1]
            
        predicted_label = [x[1] for x in predicted_label]
        
        nc = np.amax(np.abs(model.weights))
        
        #plotting

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
        ax1.set_title('Image from the MNIST dataset:{:d}'.format(y_test[x_predict].argmax()))
        ax1.imshow(x_test.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
    
        ax2.set_title('Prediction using BCM:{:d}'.format(predicted_label[0]))
        ax2.imshow(highest_response, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')
    
    else:
        print("Error: x_predict must be an integer number lower than", len(y_test)-1,\
          ". Please, enter a valid number next time")
predict(x_test, y_test, x_predict=0)


######## EVALUATION OF THE ACCURACY

def testing(x,y):

    testing = model.predict(x_test, y_test)

    y_values = [model.weights[np.argmax(x)][28*28:].argmax() for x in testing]

    y_true = y_test.argmax(axis=1)
    y_pred = np.asarray(y_values)

    print('Prediction Accuracy on the test set: {:.3f}'.format(accuracy_score(y_true, y_pred)))

testing(x_test, y_test)