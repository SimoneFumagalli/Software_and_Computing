# Example

In this section an Example on how to use the Classification and Validation scripts is shown. Some results are contained in the relative folders.

At the beginning the are several functions to import:

```python
 from plasticity.model import BCM
 from plasticity.model.optimizer import Adam, Adamax, SGD, Momentum,\
                                       NesterovMomentum, Adagrad, Adadelta,\
                                       RMSprop
 from plasticity.model.weights import Normal, GlorotNormal, HeNormal, Uniform,\
                                     LecunUniform, GlorotUniform, LecunNormal,\
                                     HeUniform, Orthogonal, TruncatedNormal
 from sklearn.model_selection import train_test_split
 from sklearn.datasets import fetch_openml
 import os
 import sys

 path = os.getcwd()
 filepath = os.path.dirname
 sys.path.append(path)

import Classification
import Validation

```

Then, the dataset must be downloaded, transformed and splitted in train and test sets. In this example, the MNIST dataset has been considered:

```python
X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

X_norm, y_transform = Classification.Variable_Reshape(X,y)

x_train, x_test, y_train, y_test = \
train_test_split(X_norm, y_transform, test_size=1./8)
```

Then, the model needs to be constructed:

```python

model = BCM(outputs=1000, num_epochs=20, optimizer=Adam(lr=4e-2), 
            interaction_strength=0.,
            weights_init=Normal(),
            activation='Relu', batch_size=4000)
```

Now, in order to obtain the configuration of the fitted neurons the `plot_view_weights` parameter of the `clf` function needs to be set on `True`.

```python
classifier = Classification.clf(model, x_train, x_test, y_train, y_test, True)
```
<div align = "center">
<img src=https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/BCM_Classification/Example/Classification/Plot_view_weights.png height = "600" width = "600">
</div>

In order to observe the comparison between the input images and the classification best results, the `plot_view_weights` must be `False` and the `plot_best_result` function must be invoked.
```python
classifier = Classification.clf(model, x_train, x_test, y_train, y_test)

for i in range (4):
    br =Classification.plot_best_result(x_test, y_test, classifier, i)
```
In this case, four images will be shown.

 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/BCM_Classification/Example/Classification/Figure_1.png" 
 height = "800" width = "800">
<img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/BCM_Classification/Example/Classification/Figure_2.png" 
 height = "800" width = "800">

<div align="center">
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Metrics_and_Classification/Example/MNIST/Classification/Figure_3.png" 
 height = "800" width = "800">
 <img src="https://github.com/SimoneFumagalli/Software_and_Computing/blob/main/Metrics_and_Classification/Example/MNIST/Classification/Figure_4.png" 
 height = "800" width = "800">
</div>

The study of the metrics related to the classification and visualize the response:
```python
single_label_metric = Classification.Metrics(classifier, y_test,False)
ten_label_metric = Classification.Metrics(classifier,y_test, True)
```

Where the results take the form:
Single Label Metric                                              | Ten Label Metric           
```python                                                        |         
               precision     recall      f1-score     support    |                 precision      recall     f1-score     support
        0        0.78         0.94         0.85        887       |        0          0.80         0.95         0.87        887
        1        0.82         0.76         0.78        970       |        1          0.66         0.98         0.79        970
        2        0.85         0.67         0.75        916       |        2          0.93         0.63         0.75        916
        3        0.75         0.68         0.72        910       |        3          0.84         0.79         0.82        910 
        4        0.85         0.80         0.82        836       |        4          0.83         0.71         0.77        836
        5        0.69         0.82         0.75        756       |        5          0.77         0.75         0.76        756
        6        0.71         0.76         0.74        888       |        6          0.80         0.95         0.87        888
        7        0.69         0.61         0.65        908       |        7          0.76         0.74         0.75        908
        8        0.74         0.68         0.71        828       |        8          0.85         0.49         0.62        828
        9        0.67         0.81         0.71        851       |        9          0.67         0.72         0.69        851
                                                                 |
accuracy                                   0.75       8750       |  accuracy                                   0.78       8750
macro avg        0.75         0.75         0.75       8750       |  macro avg        0.79         0.77         0.77       8750
weighted avg     0.76         0.75         0.75       8750       |  weighted avg     0.79         0.78         0.77       8750

```

In order to validate the data, the Validation functions can be used. With `n_splits = 8`, the `x_train` and `y_train` sets will be split in 8 train and validation sets, all with the same dimensions. With `clf_times = 4`, just four of these sets will be used to make the classification and to study the metrics. The metrics take the same form of that already explained.

```python
val_set = Validation.val_sets(x_train, y_train, 8)
val_clf = Validation.val_classification(model, val_set, 4)
val_single_label_metric = Validation.val_metrics(val_clf, val_set, False)
val_ten_label_metric = Validation.val_metrics(val_clf, val_set, True)
```