/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

public class Main {
    public static void main(String[] args) {
        int inputNodes = 3;
        int hiddenNodes = 3;
        int outputNodes = 3;
        double learningRate = 0.3;

        var neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        System.out.println(neuralNetwork);
    }
}
