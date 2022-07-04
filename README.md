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

This script contains the `class` and the functions useful to classify the Fashion-MNIST dataset obtained through fetch_openml().
Also, since the BCM functions are incompatible with the `cross_val_score` and similar methods, the Class contains the functions useful to validate the data.
The inputs of the `class` must be array, except for the number of models required for the validation.
The inputs are the following:
- output: it represents the number of neurons;
- batch: it represents the size of the batch;
- weights: it represents the weights_init function of plasticity package;
- number_attempts: it represents the number of models to consider and also the length of the other arrays;
- optimizer: it represents the optimizer;
- interaction_strength: it represents the interaction_strength parameter of plasticity package;
- number_epoch: it represents the number of epoch for the training phase.

* At first, the `Variable_Reshape` function has been constructed in order to reshape the inputs, X and y. The X corresponds to the image with the number drawn, while the y represents the label corresponding to the image.

```python
def Variable_Reshape(self,X,y):
    '''
    Function to reshape the images and the labels that enter in the model.        

    Parameters
    ----------
    X : Dataframe
        Dataframe containing informations about the pixels of images.
        
    y : Series
        List containing the labels related to the image.

    Returns
    -------
    X_norm : Dataframe
        Normalized version of the X set.
    y_transform : List
        Categorical form of the labels.

    '''
    X_norm=X*(1./255) # Normalization of the X inputs in range [0,1]
    
    # TRansformation of the y vector
    y_int = y.astype('int')
    y_reshape = y_int.values.reshape(-1,1)
    y_transform = OneHotEncoder(sparse=False).fit_transform(y_reshape)
    return X_norm, y_transform
```
* The `Variable_Split` function uses the train_test_split to generate the train and test sets.

```python
def Variable_Split(self, X, y, fraction: float):
    '''
    Function to split the input variables X and y in train and test set.        

    Parameters
    ----------
    X : Dataframe
        Dataframe containing informations about the pixels of images.
        
    y : List
        List containing the labels related to the image.
        
    fraction : float
        Fraction representing the ratio between the train and test dataset.

    Returns
    -------
    The four variables x_train, y_train, x_test, y_test composing the train and
    test sets.

    '''
    # Splitting of the variables in train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=fraction,\
                                                            random_state=42)  
    
    return x_train,\
            x_test,\
            y_train,\
            y_test
```
* The `modelizzation` function is used to generate the model, according to the element `i` of the arrays choosen.
```python
def modellization(self, i):
    '''
    Creation of the model using the parameters of the Classification class.
    
    Parameters
    ----------
    i : int
        Parameter corresponding to position of the element of each model 
        considered.
    Returns
    -------
    Construction of the model using the parameters of the class.

    '''
    
    #Construction of the BCM model
    model = [BCM(outputs= self.output[i], 
                num_epochs=self.number_epoch[i], 
                optimizer=self.optimizer[i], 
                interaction_strength=self.interaction_strength[i],
                weights_init=self.weights[i],
                activation='Relu', batch_size=self.batch[i],
                random_state=42)]
    return model
```
The value's parameters used here are just an example.

* The `fitting` function uses the model previously created to train the neurons.

```python
def fitting(self, X, model, y = None):
'''
    Training of the model.

    Parameters
    ----------
    X : Dataframe
        Train Set of images.
        
    model : function object
        BCM model.
        
    y : List
        Train Set of Labels. The default is None.

    Returns
    -------
    Trained model.

    '''
    
    if y is not None:
        model_fit = model[0].fit(X,y) #Fitting the model in a supervised way
    else:
        model_fit = model[0].fit(X) # Fitting the model, 
                                        # without the knowledge of the labels
    return [model_fit]
```

According to the parameters chosen, the `fitting` will take some times to reach the end.

* The `prevision` function uses the predict function of the BCM package to make predictions about the test set, once the model has been fitted.

```python
def prevision(self, X, y, fitted_model):
    '''
    Function to make predictions on the basis of the fitted model.

    Parameters
    ----------

    X : Dataframe
        Test or validation set of images.

    y : List
        Test or validation set of labels.

    fitted_model : function object
        Fitted model.

    Returns
    -------
    The array containing the predictions.

    '''
    #Usage of the BCM predict function
    prediction = fitted_model[0].predict(X.values, y = np.zeros_like(y))
        
    return prediction
```

* The `best_neuron` extrapolates the label information coupling the `.weights` function of the BCM package and the outputs from the `prevision` function.

```python
def best_neuron(self, fitted_model, prevision):
    '''
    Method to extract the best labels, once the model has been trained and the 
    predictions have been made.

    Parameters
    ----------
    fitted_model : function object
        Fitted model.
    
    prevision : array
        Array containing the predictions of the model.

    Returns
    -------
    execution : array
        Labels corresponding to the best neuron.

    '''
    #Extraction of the labels
    execution = [fitted_model[0].weights[np.argmax(x)][28*28:].argmax() 
                for x in prevision]
    
    return execution
```

* The `top_ten` function has the same role of the `best_neuron` but in this case the output is represented by the ten neurons with the best result for each image.

```python
def top_ten(self, fitted_model, prevision):
        '''
        Function to extract the information of the ten neurons with the highest
        response.

        Parameters
        ----------
        fitted_model : function object
            Fitted Model.
        prevision : array
            Array containing the predictions of the model.

        Returns
        -------
        top_10 : array
            Array made of arrays, containing the ten best labels for each y.

        '''
        top_10 = []
        labels = fitted_model[0].weights[:,28*28:].argmax(axis=1) # Labels of the
                 # 100 neurons 
        for x in prevision:
             
            # Union of prevision score and neuron
            sorting = sorted(zip(x, labels),\
                                key=lambda x : x[0], reverse=True)[:10]
             
            sorting = [x[1] for x in sorting]
            
            # Counting how many of the ten neurons give the same label as result
            counter_lab = (Counter(sorting).most_common())
            top_10.append(counter_lab)
        return top_10
```
* `checking_y_size` function is used to control the dimension of size of the y used in the classification function since the validation size can change the size of the train set. It's just considered the y size since it's variability can cause problems in the `single_performance` function.

```python
def checking_y_size(self, y_test, clf):
    '''
    Function to control the size of the y test set, particularly useful when
    the validation size is set in the classification function.

    Parameters
    ----------
    y_test : List
        Y test set used to check the dimension of the set of labels used in
        classification function.
    clf : Function object
        The classification function.

    Returns
    -------
    y_test : List
        Set of labels used in classification function.

    '''
    #Checking the size of the labels used
    if len(y_test) != len(clf[0]):
        print('length of y_test', (len(y_test)), 'is different from that used',\
                'in the classification function:', len(clf[0]),\
                '. The last has been considered.')
            
        y_test = clf[0]   
        
    else:
        y_test = y_test
    
    return y_test
```

* `checking_y_labels` is the function used to control the form of the labels used, either one or in the form of ten labels, necessary to study the accuracy.

```python
def checking_y_labels(self, clf, ten_label = False):
    '''
    Function to control if the set of labels comprises just the result from
    best neuron or from the ten best neurons.

    Parameters
    ----------
    clf : function
        Classification function.
    
    ten_label : Bool, optional
        The default is False.

    Returns
    -------
    y_to_test : List
        Set of labels to use in the study of metrics.

    '''
    if ten_label == False:
        
        y_to_test = clf[1]
    
    else:
        #Extrapolation of the first result from the ten labels for each of
        # image to predict
        y_to_test = [clf[2][x][0][0] for x in range (len(clf[0]))]
    
    return y_to_test
```

* `checking_batch` is the function to control the size of the batch, always due to the presence of the possibility to validate the model. In that case, the size of the X and y sets are reduced and so the batch must be control to not be higher than the new size of the two parameters.

```python
def checking_batch(self,i, y_train):
    '''
    Control function to check the batch, particularly useful when the
    validation set is used.

    Parameters
    ----------
    i : int
        Number to select the corresponding size of the batch from the batch
        array.
    y_train : List
        Set of labels used in the classification function.

    Returns
    -------
    Int
        Returns the dimension of the batch, according to the dimension of
        the label set used.

    '''
    #CHecking the dimension of the batch. If it is higher than the dimensions
    #of the labels, it will be set equal to the dimension of the labels.
    
    if self.batch[i] > len(y_train):
        print('The dimension of the batch',self.batch[i],'is much higher '
                'than that of the samples',len(y_train),'.\
                This last has been considered.')
        self.batch[i] = len(y_train)
    else:
        self.batch[i] = self.batch[i]
    return self.batch[i]
```

* `neurons_configuration` is the function to observe the acquired form of the neurons, but for just one model. In order to observe more models'configurations, the `neurons_graphs` has been written.

```python
def neurons_configuration(self, fitted_model):
    '''
    Function to generate the image containing the configuration of the 
    neurons.
    
    Parameters
    ----------
    fitted_model : function object
        Fitted model.
    
    '''
    view_weights(fitted_model[0].weights, dims=(28,28))
```

```python
def neuron_graphs(self, x_train):
    '''
    General function to show the configurations of the neurons.
    The result can be one or more images, according to the 
    number_attempts' value.

    Parameters
    ----------
    x_train : Dataframe
        Train set.
    '''
    models = []
    fits = []
            
    for i in range (self.number_attempts):
        model = self.modellization(i) # Use of model function
        models.append(model) 
        fit = self.fitting(x_train, models[i]) # Use of fitting function
        fits.append(fit)
        graphs = self.neurons_configuration(fits[i]) # Use of neurons_configuration function
```

* The `clf` function is used to make the classification and prediction of a single model, in order to reduce so the number of function to implement in the script. An example can be seen in the `Simulation` paragraph.
```python
def clf(self, i, x_train, x_test, y_train, y_test, validation_size: float = None):
    '''
    General function to make the classification.

    Parameters
    ----------
    i : Int
        Number corresponding to the element of the arrays of output, batch
        and so on.
    x_train : Dataframe
        Train set of images.
    x_test : Dataframe
        Test set of images.
    y_train : List
        Train set of labels.
    y_test : List
        Test set of labels.
    validation_size : float, optional
        Float used for further splitting of the set. The default is None.

    Returns
    -------
    y_test_ : List
        Set of labels used, either the test or the validation.
    best_neuron : List
        Set of labels corresponding to the best results for each image.
    top_10_array : List
        Set of ten labels corresponding to the best ten neurons.
    fitted_model : function object
        Fitted Model.
    predicting : array
        Array containing the predictions of the model.

    '''
    if validation_size is not None:    
        # Splitting of the train set to obtain the validation set
        x_train_, x_test_, y_train_, y_test_ = \
            self.Variable_Split(x_train, y_train, validation_size)
    else:
        x_train_, x_test_, y_train_, y_test_ = x_train, x_test,y_train, y_test
    
    #Checking the batch size.
    self.batch[i] = self.checking_batch(i, y_train_)
    
    # Use of model function
    model = self.modellization(i) 
    
    # Use of fitting function
    fitted_model = self.fitting(x_train_, model, y_train_) 
    
    # Use of prevision function
    predicting = self.prevision(x_test_, y_test_, fitted_model)
    
    # Use of labels function
    best_neuron = self.best_neuron(fitted_model, predicting)
    
    # Use of top_ten function
    top_10_array = self.top_ten(fitted_model, predicting)
    
    return y_test_, best_neuron, top_10_array, fitted_model, predicting
```

* The `single_Metric` function is used to study the metrics of a single model, after the implementation of the `clf` function.
By using the validation option in `clf`, it's so possible to validate the single model.

```python
def single_Metric(self, i, y_test, clf, ten_label = False):
    '''
    Function to study the metrics of a single model.

    Parameters
    ----------
    i : Int
        Number corresponding to the model under study.
    y_test : List
        Set of labels.
    clf : function
        Classification function.
    ten_label : bool, optional
        The default is False.

    Returns
    -------
    accuracy_values: List
        List of floats corresponding to the accuracies measured.
    Classification report: str
        List of complete classification report.

    '''
    models = []
    accuracy_values = []
    
    #Checking the size of the test size
    y_test = self.checking_y_size(y_test, clf)
    
    #Extrapolation of the labels from the hot-encoding vector
    y_test = y_test.argmax(axis=1)
    
    #Checking the kind of labels, the single one or the ten result
    y_to_test = self.checking_y_labels(clf, ten_label)
    models.append(self.modellization(i))
    
    print(classification_report(y_test, y_to_test,
                            zero_division = 0))
    
    
    performance = classification_report(y_test, y_to_test, 
                                            zero_division = 0,
                                            output_dict = True)
    
    accuracy_values.append(performance['accuracy'])
    
    
    return accuracy_values, classification_report(y_test, y_to_test, zero_division = 0)
```
* The `multiple_clf` uses the `clf` function on a certain number of models.

```python
def multiple_clf(self, x_train, x_test, y_train, y_test, v_s = None):
        '''
        Function to classify multiple models.

        Parameters
        ----------
        x_train : Dataframe
            Train set of images.
        x_test : Dataframe
            Test set of images.
        y_train : List
            Train set of labels.
        y_test : List
            Test set of labels.
        v_s : float, optional
            Fraction describing the further splitting of the train set if v_s
            is True. The default is None.

        Returns
        -------
        clas : array
            Array containing all the output of the clf function.
        '''
        clas = []
        
        #Cycle to operate the classification for each of the models under study
        for i in range(self.number_attempts):
            classification = self.clf(i, x_train, x_test, y_train, y_test, v_s)
            clas.append(classification)
        return clas
```
* The `Metrics` function is used to study the perfomance of a list of parameters related to different models. Even in this case, it's possible to validate the models.

```python
def Metrics(self, y_test, multiple_clf, ten_label = False):
        '''
        Function used to give some informations about the model predictions
        performance, such as accuracy, precision, recall and so on.

        Parameters
        ----------
        y_test : List
            Test set of labels.
        
        multiple_clf: list
            LIst of output from the multiple classification function.
        ten_label : bool, optional
            The default is False.

        Returns
        -------
        accuracy_values : List
            List of values as output of single metric function applied to
            different models.
        '''
        models = []
        single_p = []
        accuracy_values = []
        
        #Cycle to operate the performance of each models
        for i in range(self.number_attempts):
            models.append(multiple_clf[i][3])
            perf = self.single_Metric(i, y_test, multiple_clf[i], ten_label)
            accuracy_values.append(perf[0])
        
        print('Maximum accuracy:', np.max(accuracy_values),'from the model:', \
              models[np.argmax(accuracy_values)])
            
        return accuracy_values
```

* The `best_result` function is used to shown the input image and the obtained result using a model. If one want to use the multiple classification function, insert the corresponding set of model, like clf[0] or clf[1] and so on.

```python
def best_result(self, x_test, y_test, clf, x_predict):
    '''
        Major function to show the model's best neuron graph.

        Parameters
        ----------
        x_test : Dataframe
            Dataframe of the test set.
        y_test : List
            Set of test labels.
        clf : function object.
            Array containing the different outputs of the classification function.
            If the multiple classification function is used, insert clf[num_model].
        x_predict : int
            Integer corresponding to the number of label to predict.

        Returns
        -------
        label : Int
            label of the best neuron.
        Ten_label: List
            labels of the ten best neuron.

        '''   
        nc = np.amax(np.abs(clf[3][0].weights))
        label = clf[1][x_predict]
        ten_label = clf[2][x_predict]
        best_result = clf[3][0].weights[np.argmax(clf[4][x_predict])]\
            [:28*28].reshape(28, 28)
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
        ax1.set_title('Image from the Fashion-MNIST dataset: {:d}'.format(y_test[x_predict].argmax()))
        ax1.imshow(x_test.values[x_predict].reshape(28, 28), cmap='gray'); ax1.axis('off')
        
        ax2.set_title('Prediction using BCM: {:d}'.format(label))
        ax2.imshow(best_result, vmin=-nc, vmax=nc); ax2.axis('off')
        
        fig.text(0.5, 0.24, 'Ten best neuron result: {}'.format(ten_label),
                 horizontalalignment='center', fontsize = 14)
        
        return label, ten_label
```

### Results
 The `results` folder contains the best results obtained using different parameters of the Plasticity package.
 The `multiple_clf`function has been used, with and without the validation set.
 At first, the `BCM_Classification` must be imported,
 together with the Fashion-Mnist and the `fetch_openml` from the `sklearn.datasets`.
 Then, the Fashion-MNIST dataset must be imported and assigned to the variables X and y.

 ```python
 from sklearn.datasets import fetch_openml

 from plasticity.model.optimizer import Adam, Adamax, SGD, Momentum, NesterovMomentum, Adagrad, Adadelta, RMSprop

 from plasticity.model.weights import GlorotNormal, HeNormal, Uniform, LecunUniform, GlorotUniform, LecunNormal, HeUniform, Orthogonal, TruncatedNormal

 from BCM_classification import Classification

 X, y = fetch_openml(name='Fashion-MNIST', version=1,  data_id=None, return_X_y=True)
 ```
 The parameters of the `class` then must be written.

 In general, the parameters must be written once.
 Once passing from a fitting model with both X and y to one with just the X (or the inverse passage), the parameters and the `class` must be re-assigned.

#### Visualization of the Neuron Configuration

 ```python
 out = [100,100,100,100]
 epoc = [250, 600, 700, 800]
 bat = [61250,61250,61250,61250]
 wei = [GlorotUniform(),GlorotNormal(),GlorotNormal(),GlorotNormal()]
 num = 4
 opti = [SGD(lr=1e-2), SGD(lr=3e-2),, SGD(lr=1e-2), SGD(lr=1e-2)]
 strenght = [1.5,-0.09, 1, 1]
 clas = Classification(out, bat, wei, num, opti, strenght, epoc)
 ```
 By using the `Variable_Reshape` and the `Variable_Split`, X and y are first reshaped and then they are splitted in the train and test set.
 
 ```python
 x_norm, y_resh = clas.Variable_Reshape(X, y)
 x_train,x_test,y_train,y_test = clas.Variable_Split(x_norm, y_resh, 1./8)
 ```
 Then the `neuron_graphs` function must be called and, depending on the `number_attempts`, a certain number of output will be shown. Every single result will be equal to that in case just `neuron_configuration` is used.
 
 ```python
 ne = clas.neuron_graphs(x_train)
 ```

 #ALMENO 4 IMMAGINI DI NEURONI CONFIGURATI
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Fitting_Epochs.png" 
 height = "350" width = "350">

 #### Classification and Performance
 At this point, the parameters and the model must be re-assigned.

 ```python
 out = [1000,1000,1000,1000]
 epoc = [250, 600, 700, 800]
 bat = [61250,61250,61250,61250]
 wei = [GlorotUniform(),GlorotNormal(),GlorotNormal(),GlorotNormal()]
 num = 4
 opti = [SGD(lr=1e-2), SGD(lr=3e-2),, SGD(lr=1e-2), SGD(lr=1e-2)]
 strenght = [1.5,-0.09, 1, 1]
 clas = Classification(out, bat, wei, num, opti, strenght, epoc)
 ```
 
 Now, the `multiple_clf` function can be used, with and without the validation set.

 ```python
 #VALIDATION
 multiple_clas = clas.multiple_clf(x_train, x_test, y_train, y_test,1./8)

 #WITHOUT VALIDATION
 multiple_clas = clas.multiple_clf(x_train, x_test, y_train, y_test)
 ```
 [Note: even the `clf` function can be used with and without the validation set. In that specific case, the number of model that one want to consider must be specified. Also, the `single_Metric` must be used.]
 
 The `Metric` function then can be used, obtaining the performance of the different models, indicating the best result.
 
 ```python
 metrics = clas.Metrics(y_test, clas)
 ```
 #Risultati Metric with Validation Set
 <div align = "center">
 <table cellspacing="2" cellpadding="2" width="400" border="0">
 <tbody>
 <tr><td valign="top">
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/results/Metric_Validation_Set/Metric_best_neur_part_1.png" 
 height = "300" width = "300"> </td>
 <td valign="top">
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/results/Metric_Validation_Set/Metric_best_neur_part_2.png" 
 height = "200" width = "200"> <td>
 </tr></body></table>
 </div>
 #Risultati Metric with Test Set
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Fitting_Epochs.png" 
 height = "350" width = "350">

It is also possible to study the performance of the classification by considering the ten best neurons.

 ```python
 metrics = clas.Metrics(y_test, clas, ten_labels = True)
 ```

 In the images below, it's possible to observe decreased results, due probably to the not so much accurate configuration of the neurons (for the result with the validation set, look at the images in the folder).

 #Risultati Metric with Test Set
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Fitting_Epochs.png" 
 height = "350" width = "350">

 The best result between the considered models is:
 ```python
 output = 1000
 epoch = 800
 batch_size = 61250
 weight = GlorotUniform()
 optimizer = SGD(lr=1e-2)
 interaction_strength = 1
 ```
 Some examples are shown below.

 #Immagini ottenute
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Classification/Simulation_images/Fitting_Epochs.png" 
 height = "350" width = "350">

 
 ### Testing

 In the last file, `testing`, the parameters are controlled to be in the right form such as required by the functions. 














 ## References

 * Blais, B. S., & Cooper, L. (2008). BCM theory. Scholarpedia, 3(3), 1570. (http://scholarpedia.org/article/BCM)

 * Jedlicka, P. (2002). Synaptic plasticity, metaplasticity and BCM theory. Bratislavské lekárske listy, 103(4/5), 137-143.

* https://github.com/Nico-Curti/plasticity