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
    
    xdata = [5, 10, 20, 40, 100, 200, 500]
    
    plt.plot(xdata,
            [0.7128, 0.8944, 0.9327, 0.9541, 0.9685, 0.975, 0.9763], 
            "-o", lw=3, label="learn rate = 0.05")
    plt.plot(xdata,
            [0.8382, 0.9044, 0.9255, 0.9537, 0.9662, 0.9726, 0.977], 
            "-s", lw=3, label="learn rate = 0.1")
    
    #plt.xticks(range(0, xdata[-1] + 1, 50))
    plt.title("Change of Performance of the Neural Network with \nLearning rate and number of hidden nodes (Epochs = 7)")
    plt.xlabel("number of hidden nodes", fontsize=12)
    plt.ylabel("performance", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.margins(0.05, 0.1)
    plt.savefig("performance_lr_hiddennodes.png", bbox_inches = 'tight', dpi=320)
    