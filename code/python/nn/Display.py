#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 05:23:44 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""

import numpy as np
import matplotlib.pyplot as plt

def showDigit(mnist_data_line, w=28, h=28):
    # mnist_data_line = digit followed by image data
    all_values = mnist_data_line.split(',')
    # convert list to float array -> image of wxh
    image_array = np.asfarray(all_values[1:]).reshape((h, w))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.title('Label is "%s"' % all_values[0], fontsize=25)
    plt.show()
    