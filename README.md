# Classification and Inpainting using BCM

* [Theory](#theory)
* [Classification](#classification)
* [References](#references)


## Theory

The model at the basis of this work is called BCM (Bienenstock, Cooper and Munro) theory that refers to the synaptic modification first proposed in 1982.
The BCM theory inserts inside the synaptic plasticity field, that is a process in which the synapses change their efficacy according to their previous acitivities.
Starting from this notion, Donald Hebb proposed a form of synaptic plasticity driven by the pre- and postsynaptic activity.
He postulated that a repeated and persistent stimulation of a postsynaptic cell from a presynaptic ones increases its strength, known as Hebb's Rule.

This theory of synaptic plasticity is based on three postulates:

* The change in synaptic weights  <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}dw_i/dt" title="https://latex.codecogs.com/gif.image?\small \dpi{110}dw_i/dt" />  is proportional to presynaptic activity <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}x_i" title="https://latex.codecogs.com/gif.image?\small \dpi{110}x_i" />;
* This change is also proportional to a non-monotonic function <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}\phi" title="https://latex.codecogs.com/gif.image?\small \dpi{110}\phi" /> of the postsynaptic activity <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}y" title="https://latex.codecogs.com/gif.image?\small \dpi{110}y" />. It has two different behaviours according to the postsynaptic activity: it decreses for low y and increases for higher <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}y" title="https://latex.codecogs.com/gif.image?\small \dpi{110}y" />;
* The modification threshold, indicated with <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta" title="https://latex.codecogs.com/png.image?\dpi{110}\theta" /> and corresponding to a variation higher than zero, is itself a superlinear function of the history of postsynaptic activity <img src="https://latex.codecogs.com/gif.image?\small&space;\dpi{110}y" title="https://latex.codecogs.com/gif.image?\small \dpi{110}y" />.

More generally, BCM model proposes a sliding threshold for long-term potentiation (LTP) and long-term depression (LTD) induction, stating that synaptic plasticity is stabilized by a dynamic adaptation of the time-averaged post-synaptic activity.

### Mathematical Formulation of BCM

The original BCM equation is defined by:

<img src="https://latex.codecogs.com/png.image?\dpi{110}y&space;=&space;\sum_{i}w_ix_i&space;\\\\\indent&space;\frac{\mathrm{d}&space;w_i}{\mathrm{d}&space;t}&space;=&space;y(y-\theta)x_i-\epsilon&space;w_i,&space;\\\\\indent&space;\theta&space;=&space;E[y/y_0]&space;" title="https://latex.codecogs.com/png.image?\dpi{110}y = \sum_{i}w_ix_i \\\\\indent \frac{\mathrm{d} w_i}{\mathrm{d} t} = y(y-\theta)x_i-\epsilon w_i, \\\\\indent \theta = E[y/y_0], " />

while a more recent formula has been drawn from Law and Cooper in 1994:

<img src="https://latex.codecogs.com/png.image?\dpi{110}y&space;=&space;\sigma\Biggl(\sum_{i}w_ix_i\Biggl),&space;\\\\\indent&space;\frac{\mathrm{d}w_i}{\mathrm{d}&space;t}&space;=&space;\frac{y(y&space;-&space;\theta)x_i}{\theta},&space;\\\\\indent&space;\theta&space;=&space;E[y^{2}]&space;" title="https://latex.codecogs.com/png.image?\dpi{110}y = \sigma\Biggl(\sum_{i}w_ix_i\Biggl), \\\\\indent \frac{\mathrm{d}w_i}{\mathrm{d} t} = \frac{y(y - \theta)x_i}{\theta}, \\\\\indent \theta = E[y^{2}] " />

For further details, look at this site [here](http://scholarpedia.org/article/BCM).

## Classification

In order to work with the BCM model, the plasticity package must be downloaded at this [link](https://github.com/Nico-Curti/plasticity), created by Nico Curti et al. on GitHub. At this link, information about the download of the plasticity package, implementation and parameters value can be found.

Once the installation of the Python version of the Plasticity package has been performed, download also the Classification folder, in which are contained the BCM_Classification, testing and simulation files.

In the next paragraphs, just an explanation of the code is shown. If interested in how to use these scripts and to see some outputs, go to `Simulation`. 

### BCM_Classification

This script contains the functions useful to classify the MNIST dataset.
* At first, the `variable` function has been constructed in order to generate the inputs, X and y, and to splitting them into two sets of different length, the train and test. The X corresponds to the image with the number drawn, while the y represents the label corresponding to the image.

```python
def variables(X,y, i):
    X_1=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    #Transformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    
    #Splitting of the variables in training and test sets
    x_train, x_test, y_train, y_test = \
    train_test_split(X_1, y_transform, test_size=1./8, random_state=42)
    variable = [x_train, y_train, x_test, y_test]
    return variable[i]
```

* The `fitting` function implements the BCM model and it is trained using the train set previously created.

```python
def fitting(X,y):
    model = BCM(outputs=100, num_epochs=10, optimizer=Adam(lr=1e-3), 
                interaction_strength=0.,
                weights_init=HeNormal(),
                activation='Relu', batch_size=100)
    
    model.fit(X,y) #training of the model
    return model
```

According to the parameters chosen, the `fitting` will take some times to reach the end. The value's parameters used here are just an example.

* The `predict` function is constructed to control if the input parameter, x_predict, is chosen correctly. Indeed, this parameter must respect some conditions since it represents the image for which the model should represent its prediction. After the prediction, the highest response has been recorded and coupled to the best label, representing the number corresponding to the image. The, the plot showing the raw and predicting images is printed, reporting also the already known and predicted labels.

```python
def predict(X, y, model, x_predict):
    if type(x_predict) == int and x_predict < len(y):
        predict = model.predict(X.values[x_predict].reshape(1, -1), \
                                y=np.zeros_like(y[x_predict].reshape(1, -1)))
    
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
        ax1.set_title('Image from the MNIST dataset:{:d}'.format(y[x_predict].argmax()))
        ax1.imshow(X.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
    
        ax2.set_title('Prediction using BCM:{:d}'.format(predicted_label[0]))
        ax2.imshow(highest_response, cmap='bwr', vmin=-nc, vmax=nc); ax2.axis('off')
        return len(labels)
    else:
        print("Error: x_predict must be an integer number lower than", len(y)-1,\
          ". Please, enter a valid number next time")
```

* In the last part of this script, a function for `testing`  the accuracy of the prediction has been defined.

```python

def accuracy(X,y,model):
    accuracy = model.predict(X, y)
    
    # prediction of the labels
    y_values = [model.weights[np.argmax(x)][28*28:].argmax() for x in accuracy]

    y_true = y.argmax(axis=1)
    
    y_pred = np.asarray(y_values)
    
    length = [len(y_values), len(y_true), len(y_pred)]
    print('Prediction Accuracy on the test set: {:.3f}'.format(accuracy_score(y_true, y_pred)))
    return length
```


For the predict and testing functions, the `return` elements are useful for testing purposes.

### Simulation
 The `simulation` script contains the main code of the project, containing the functions of the `BCM_Classification` script and showing the results obtained.
 At first, the `BCM_Classification` must be imported, together with the functions that it contains.
Even the MNIST dataset must be imported and so the `fetch_openml` is imported by `sklearn.datasets` 

```python
from BCM_classification import variables, fitting, predict, accuracy
from sklearn.datasets import fetch_openml
```

Then, the MNIST dataset must be imported and assigned to the variables X and y. To download the MNIST dataset, some time can be required.
After, the train and test datasets are generated by recalling the `variables` function.

```python
X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
x_train = variables(X, y, 0)
y_train = variables(X, y, 1)
x_test = variables(X, y, 2)
y_test = variables(X, y, 3)
```

At this point, the model is created and fitted by usign the `fitting` function and the `x_train`, `y_train` parameters.

```python
model = fitting(x_train, y_train)
```

<img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Fitting_Epochs.png" 
height = "350" width = "350">

Now, once the model has been trained, the prediction and the accuracy test can be performed by using the `x_test` and `y_test` parameters. With `x_predict` is indicated the image that the model must be able to predict.  

```python
predict(x_test,y_test,model,x_predict=0)
accuracy(x_test,y_test,model)
```

In order to perform the prediction simulation, the following model has been used:

```python
model = BCM(outputs=1000, num_epochs=10, optimizer=Adam(lr=4e-2), 
                interaction_strength=0.,
                weights_init=GlorotNormal(),
                activation='Relu', batch_size=10000)
```

which means that the `GlorotNormal` function must be imported in `BCM_Classification`.

By using this setting, the output of predict and testing functions are the following:
\\
<img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Plot_raw_predict.png" 
height = "300" width = "500">
<img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Accuracy_test.png" 
height = "50" width = "300">
















## References

* Blais, B. S., & Cooper, L. (2008). BCM theory. Scholarpedia, 3(3), 1570. (http://scholarpedia.org/article/BCM)

* Jedlicka, P. (2002). Synaptic plasticity, metaplasticity and BCM theory. Bratislavské lekárske listy, 103(4/5), 137-143.

* https://github.com/Nico-Curti/plasticity