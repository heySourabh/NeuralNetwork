#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:02:58 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""

import numpy as np
import scipy.special

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
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert lists to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        
        # hidden layer error is the output_errors, split by weights 
        # and recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for links between the hidden and output layers
        self.who += self.lr * \
            np.dot(output_errors * final_outputs * (1.0 - final_outputs),
                   hidden_outputs.T)
        
        # update the weights for links between the input and hidden layers
        self.wih += self.lr * \
            np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                   inputs.T)
        
        pass
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


if __name__ == "__main__":
    nn = NeuralNetwork(3, 3, 3, 0.3)
    print(nn.query([1, 2, 3]))
    print("test")