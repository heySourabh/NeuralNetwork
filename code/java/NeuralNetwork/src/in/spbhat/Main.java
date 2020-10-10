/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import in.spbhat.util.MNISTData;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        int inputNodes = 3;
        int hiddenNodes = 3;
        int outputNodes = 3;
        double learningRate = 0.3;

        var neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        Scanner mnistScanner = new Scanner(new File("mnist_dataset/mnist_test_10.csv"));
        new MNISTData(mnistScanner.nextLine()).display(10);
    }
}
