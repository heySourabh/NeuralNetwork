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

def plot(x, y, title, xlabel, ylabel):
    plt.plot(x, y, "-o", lw=3)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=15)
    ax = plt.axes()
    plt.grid()
    plt.tight_layout()


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "Palatino"
    plot([0.01, 0.1, 0.2, 0.3, 0.6, 0.9], 
         [0.9111, 0.95028, 0.9512, 0.94686, 0.90808, 0.86796], 
         "Change of Performance of the \nNeural Network with Learning Rate",
         "learning rate", "performance")
    