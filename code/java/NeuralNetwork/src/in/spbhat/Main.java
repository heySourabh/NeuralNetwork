/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import in.spbhat.util.MNISTData;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static in.spbhat.util.DoubleArray.apply;
import static in.spbhat.util.DoubleArray.newFilledArray;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        int inputNodes = 28 * 28;
        int hiddenNodes = 100;
        int outputNodes = 10;
        double learningRate = 0.3;

        var neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        System.out.println("Training network");
        try (Scanner trainDataFile = new Scanner(new File("mnist_dataset/mnist_train_100.csv"))) {
            while (trainDataFile.hasNextLine()) {
                MNISTData trainData = new MNISTData(trainDataFile.nextLine());
                double[] inputs = apply(trainData.getData1D(), e -> e * 0.99 + 0.01);
                double[] targets = newFilledArray(outputNodes, 0.01);
                targets[trainData.getLabel()] = 0.99;
                neuralNetwork.train(inputs, targets);
            }
        }

        System.out.println("Testing network");
        List<Integer> scorecard = new ArrayList<>();
        try (Scanner testDataFile = new Scanner(new File("mnist_dataset/mnist_test_10.csv"))) {
            while (testDataFile.hasNextLine()) {
                MNISTData testData = new MNISTData(testDataFile.nextLine());
                double[] inputs = apply(testData.getData1D(), e -> e * 0.99 + 0.01);
                double[] outputs = neuralNetwork.query(inputs);
                int networkLabel = getLabel(outputs);
                int correctLabel = testData.getLabel();
                scorecard.add(networkLabel == correctLabel ? 1 : 0);
                System.out.println("Correct label = " + correctLabel);
                System.out.println("Network's label = " + networkLabel);
            }
        }
        System.out.println(scorecard);
    }

    private static int getLabel(double[] output) {
        int maxIndex = 0;
        double maxOutput = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxOutput) {
                maxIndex = i;
                maxOutput = output[i];
            }
        }
        return maxIndex;
    }
}
