# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:42:16 2022

@author: Simone Fumagalli
"""

import numpy as np
import Classification

def ylabels(classifier):
    y_lab = [classifier[0][np.argmax(x)][28*28:].argmax() 
              for x in classifier[1]]
    return y_lab

def top_10_labels(classifier):
    top_10 = Classification.top_ten(classifier)
    ten_result = [top_10[x][0][0] for x in range(len(top_10))]
    return ten_result