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

if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "Palatino"
    
    xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    
    plt.plot(xdata,
            [0.940766666666667, 0.9572, 0.962833333333333, 0.967933333333333, 
             0.967666666666667, 0.969466666666667, 0.970433333333333, 0.9702, 
             0.969366666666667, 0.969666666666667, 0.968266666666667], 
            "-o", lw=3, label="learn rate = 0.05")
    plt.plot(xdata,
            [0.9471, 0.9599, 0.963633333333334, 0.965933333333333, 0.9654, 
             0.9657, 0.966133333333333, 0.967333333333333, 0.9648, 0.9662, 
             0.964066666666667], 
            "-s", lw=3, label="learn rate = 0.1")
    plt.plot(xdata,
            [0.949766666666667, 0.957, 0.962066666666667, 0.962466666666667, 
             0.960966666666667, 0.959333333333333, 0.9595, 0.9613, 0.9594, 
             0.9585, 0.9551], 
            "-^", lw=3, label="learn rate = 0.2")
    plt.plot(xdata,
            [0.9435, 0.952133333333333, 0.952033333333333, 0.9508, 0.9541, 
             0.951133333333333, 0.9493, 0.948266666666667, 0.947833333333333, 
             0.946, 0.944666666666667], 
            "-v", lw=3, label="learn rate = 0.3")
    
    plt.xticks(range(0, xdata[-1] + 1, 2))
    plt.title("Change of Performance of the Neural Network with \nLearning rate and Epochs")
    plt.xlabel("number of epochs", fontsize=12)
    plt.ylabel("performance", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    # plt.margins(0,0)
    plt.savefig("performance_lr_epochs.png", bbox_inches = 'tight', dpi=320)
    