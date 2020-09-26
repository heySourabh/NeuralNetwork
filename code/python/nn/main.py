#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:15:22 2020

@author: Sourabh Bhat ( https://spbhat.in/ )
"""
import NeuralNetwork as nn

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
    
    print(neuralNetwork.query([5, 1, -8]))
    

if __name__ == "__main__":
    main()
    