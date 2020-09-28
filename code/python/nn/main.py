#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:15:22 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""
import NeuralNetwork as nn
import Display as display

def main():
    # number of input, hidden and output nodes
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    
    # learning rate
    learning_rate = 0.3
    
    # create instance of neural network
    neuralNetwork = nn.NeuralNetwork(input_nodes, hidden_nodes, 
                         output_nodes, learning_rate)
    
    # read MNIST data
    data_file = open("mnist_dataset/mnist_train_100.csv")
    data_list = data_file.readlines()
    data_file.close()
    
    # plot MNIST data
    display.showDigit(data_list[1])
    

if __name__ == "__main__":
    main()
    