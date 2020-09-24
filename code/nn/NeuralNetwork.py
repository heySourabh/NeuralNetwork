#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:02:58 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""

import numpy as np

# Neural network class definition
class NeuralNetwork:
    
    # initialize the neural network
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = numInputNodes
        self.hnodes = numHiddenNodes
        self.onodes = numOutputNodes
        
        # learning rate
        self.lr = learningRate
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is 
        # from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc.
        self.wih = np.random.normal(0.0, 1.0 / np.sqrt(self.inodes),
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, 1.0 / np.sqrt(self.hnodes), 
                                    (self.onodes, self.hnodes))
        pass
    
    # train the neural network
    def train(self):
        pass
    
    # query the neural network
    def query(self):
        pass
