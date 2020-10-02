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
    learning_rate = 0.2
    
    # create instance of neural network
    neuralNetwork = nn.NeuralNetwork(input_nodes, hidden_nodes, 
                         output_nodes, learning_rate)
    
    print("Training...")
    # load the MNIST training data CSV file into a list
    training_data_file = open("../../../mnist_dataset/mnist_train.csv", 'r')
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
    
    print("Testing...")
    # load the MNIST test data CSV file into a list
    test_data_file = open("../../../mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    # test the neural network
    # scorecard for how well the network performs, initially empty
    scorecard = []
    
    # go through all the records in the test data set
    for record in test_data_list:
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # print(correct_label, "correct label")
        # scale and shift the inputs
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        # query the network
        outputs = neuralNetwork.query(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # print(label, "network's answer\n")
        # append correct or incorrect to list
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
    
    # print(scorecard)
    
    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("performance =", scorecard_array.sum() / scorecard_array.size)

if __name__ == "__main__":
    main()
    