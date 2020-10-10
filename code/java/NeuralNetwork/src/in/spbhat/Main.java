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
        long timeStart = System.currentTimeMillis();
        double performance = networkPerformance(200, 0.1, 5,
                new File("../../../mnist_dataset/mnist_train.csv"),
                new File("../../../mnist_dataset/mnist_test.csv"));
        System.out.println("Performance = " + performance);
        long timeStop = System.currentTimeMillis();
        System.out.println("Time taken = " + (timeStop - timeStart) / 1000.0 + " seconds");
    }

    private static double networkPerformance(int hiddenNodes, double learningRate, int epochs,
                                             File trainDataFile, File testDataFile) throws FileNotFoundException {
        int inputNodes = 28 * 28;
        int outputNodes = 10;

        var neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        System.out.println("Training network...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            try (Scanner trainDataScanner = new Scanner(trainDataFile)) {
                while (trainDataScanner.hasNextLine()) {
                    MNISTData trainData = new MNISTData(trainDataScanner.nextLine());
                    double[] inputs = apply(trainData.getData1D(), e -> e * 0.99 + 0.01);
                    double[] targets = newFilledArray(outputNodes, 0.01);
                    targets[trainData.getLabel()] = 0.99;
                    neuralNetwork.train(inputs, targets);
                }
            }
        }

        System.out.println("Testing network...");
        List<Integer> scorecard = new ArrayList<>();
        try (Scanner testDataScanner = new Scanner(testDataFile)) {
            while (testDataScanner.hasNextLine()) {
                MNISTData testData = new MNISTData(testDataScanner.nextLine());
                double[] inputs = apply(testData.getData1D(), e -> e * 0.99 + 0.01);
                double[] outputs = neuralNetwork.query(inputs);
                int networkLabel = getLabel(outputs);
                int correctLabel = testData.getLabel();
                scorecard.add(networkLabel == correctLabel ? 1 : 0);
            }
        }

        return scorecard.stream()
                .mapToDouble(i -> i)
                .sum() / scorecard.size();
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
