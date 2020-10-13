/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import in.spbhat.util.MNISTData;
import org.ojalgo.ann.ArtificialNeuralNetwork;
import org.ojalgo.ann.NetworkInvoker;
import org.ojalgo.ann.NetworkTrainer;
import org.ojalgo.structure.Access1D;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import static in.spbhat.util.DoubleArray.apply;
import static in.spbhat.util.DoubleArray.newFilledArray;
import static org.ojalgo.ann.ArtificialNeuralNetwork.Error.HALF_SQUARED_DIFFERENCE;

public class OjAlgoNetwork {
    public static void main(String[] args) throws IOException {
        long timeStart = System.currentTimeMillis();
        int inputNodes = 28 * 28;
        int hiddenNodes = 200;
        int outputNodes = 10;
        double learningRate = 0.1;
        int epochs = 5;
        NetworkTrainer trainer = ArtificialNeuralNetwork
                .builder(inputNodes)
                .layer(hiddenNodes, ArtificialNeuralNetwork.Activator.SIGMOID)
                .layer(outputNodes, ArtificialNeuralNetwork.Activator.SIGMOID)
                .get().newTrainer()
                .error(HALF_SQUARED_DIFFERENCE)
                .rate(learningRate);

        // training
        System.out.println("Training network...");
        File trainDataFile = new File("../../../mnist_dataset/mnist_train.csv");
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.print("Epoch:" + (epoch + 1) + "/" + epochs + "\r");
            Files.lines(trainDataFile.toPath()).map(MNISTData::new).forEach(trainData -> {
                double[] inputs = apply(trainData.getData1D(), e -> e * 0.99 + 0.01);
                double[] targets = newFilledArray(outputNodes, 0.01);
                targets[trainData.getLabel()] = 0.99;
                Access1D<Double> input = Access1D.wrap(inputs);
                Access1D<Double> output = Access1D.wrap(targets);
                trainer.train(input, output);
            });
        }

        // testing
        final NetworkInvoker neuralNetwork = trainer.get().newInvoker();
        System.out.println("Testing network...");
        File testDataFile = new File("../../../mnist_dataset/mnist_test.csv");
        List<Integer> scorecard = new ArrayList<>();
        Files.lines(testDataFile.toPath()).map(MNISTData::new).forEach(testData -> {
            double[] inputs = apply(testData.getData1D(), e -> e * 0.99 + 0.01);
            final Access1D<Double> input = Access1D.wrap(inputs);
            double[] outputs = neuralNetwork.invoke(input).toRawCopy1D();
            int networkLabel = getLabel(outputs);
            int correctLabel = testData.getLabel();
            scorecard.add(networkLabel == correctLabel ? 1 : 0);
        });

        System.out.println("Performance = " +
                scorecard.stream()
                        .mapToDouble(i -> i)
                        .sum() / scorecard.size());

        long timeStop = System.currentTimeMillis();
        System.out.println("Time taken = " + (timeStop - timeStart) / 1000.0 + " seconds");
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
