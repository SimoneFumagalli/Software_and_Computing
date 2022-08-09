# BCM classification

* [The scripts](#scripts)
* [How to Install](#Howtoinstall)
* [How to Use](#howtouse)
* [References](#references)

## Scripts

This repository contains four scripts:
* The Classification script contains functions to operate automatically a classification of a model starting from the BCM package. In addition, a function to study the metric has been implemented;
* The Validation script contains the functions to operate a validation of the model constructed;
* The testing file contains the tests made on the Classification and Validation by using the pytest function.
* The example scripts contains how to use the functions. In this folder, some results are shown.

## How to Install

In order to work with these scripts, first the BCM package must be installed following the guide at this [link](https://github.com/Nico-Curti/plasticity). To know better the BCM package and how to use it, reference to this link.
Then the present repository can be downloaded:
```python
git clone https://github.com/SimoneFumagalli/Software_and_Computing
```
## How to Use
In this section the steps to use the Classification and Validation are shown, with the example scripts observable [here] LINK DEGLI ESEMPI.

* The following modules must be imported:
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
* Then, the set of images and labels must be imported by using for example the ``fetch_openml`` function. These inputs now need to be reshaped: the X is rescaled and the y is transformed using the OneHotEncoding.
After the reshaping, the dataset needs to be separated in the train and test sets, using the ``train_test_split`` function.
Now, the model can be constructed.

### Classification
* In order to make the classification, the ``Classification.clf`` must be used. The result will depend on the ```plot_view_weights```:
if it is set ```True```, the result will be the plot of the configured neurons; if ```False``` it will be composed by the weights of the fitted model and the predictions made.
Example of the `plot_view_weights = True`:

* In the end, after the classification, it's possible to observe which neuron has been chosen to be the best one by invoking the ```plot_best_result``` and specifying which image by using the ```x_predict``` function.
Instead, to study the metrics of the model, the ```Metrics```function must be called, specifying if the result is based on the best neuron response (```ten_label_type = False```) or if it is based on the vote of the ten neurons with the highest response (```ten_label_type = True```).
Example of the `plot_best_result`, followed by the example of the metric with `ten_label_type = False`:

### Validation
* To validate the used model, the `Validation.py` script can be used. At first, the `val_sets` function must be written where thanks to the `StratifiedkFold` function different train and test sets are generated using the `train` and `test` set obtained woth the `train_test_split`used before. The `n_splits` parameter indicates the number of different sets.
* Now, the classification can be performed using `val_classification` which takes as parameter the `model`, the `val_sets` variable and the number of classifications to perform `clf_times`.
* The metrics can be studied using `val_metrics` function that takes `val_classification` and  `val_sets` parameter.

## References

 * Blais, B. S., & Cooper, L. (2008). BCM theory. Scholarpedia, 3(3), 1570. (http://scholarpedia.org/article/BCM)

 * Jedlicka, P. (2002). Synaptic plasticity, metaplasticity and BCM theory. Bratislavské lekárske listy, 103(4/5), 137-143.

* https://github.com/Nico-Curti/plasticity