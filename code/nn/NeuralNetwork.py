#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:02:58 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""


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
        pass
    
    # train the neural network
    def train(self):
        pass
    
    # query the neural network
    def query(self):
        pass
