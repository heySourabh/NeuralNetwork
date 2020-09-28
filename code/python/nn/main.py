#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:15:22 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""
import NeuralNetwork as nn
import Display as display
import numpy as np

def main():
    # number of input, hidden and output nodes
    input_nodes = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    
    # learning rate
    learning_rate = 0.3
    
    # create instance of neural network
    neuralNetwork = nn.NeuralNetwork(input_nodes, hidden_nodes, 
                         output_nodes, learning_rate)
    
    # load the MNIST training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    # plot MNIST data
    # display.showDigit(training_data_list[1])
    
    # train the neural network: 
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        # create target output values (0.99 for desired label, 0.01 otherwise)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the desired label for this record
        targets[int(all_values[0])] = 0.99
        neuralNetwork.train(inputs, targets)
        

if __name__ == "__main__":
    main()
    