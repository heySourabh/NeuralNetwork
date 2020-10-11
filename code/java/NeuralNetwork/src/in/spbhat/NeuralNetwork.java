/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import java.util.Random;

import static in.spbhat.util.DoubleArray.apply;
import static in.spbhat.util.DoubleArray.subtract;
import static in.spbhat.util.DoubleMatrix.*;
import static java.lang.Math.exp;

public class NeuralNetwork {
    private static final Random rng = new Random();

    private final int iNodes;
    private final int hNodes;
    private final int oNodes;
    private final double learningRate;
    private double[][] wih;
    private double[][] who;

    public NeuralNetwork(int numInputNodes, int numHiddenNodes, int numOutputNodes, double learningRate) {
        this.iNodes = numInputNodes;
        this.hNodes = numHiddenNodes;
        this.oNodes = numOutputNodes;
        this.learningRate = learningRate;

        // link weight matrices, wih and who
        // weights inside the arrays are wij, where link is from node i to node j in the next layer
        this.wih = createRandom(hNodes, iNodes, 0.0, 1.0 / Math.sqrt(iNodes), rng);
        this.who = createRandom(oNodes, hNodes, 0.0, 1.0 / Math.sqrt(hNodes), rng);
    }

    public void train(double[] inputs, double[] targets) {
        double[] hiddenInputs = multiply(wih, inputs);
        double[] hiddenOutputs = activationFunction(hiddenInputs);
        double[] finalInputs = multiply(who, hiddenOutputs);
        double[] finalOutputs = activationFunction(finalInputs);

        double[] outputErrors = subtract(targets, finalOutputs);
        double[] hiddenErrors = multiply(transpose(who), outputErrors);

        who = add(who, computeWeightsChange(hiddenOutputs, finalOutputs, outputErrors));
        wih = add(wih, computeWeightsChange(inputs, hiddenOutputs, hiddenErrors));
    }

    public double[] query(double[] inputs) {
        double[] hiddenInputs = multiply(wih, inputs);
        double[] hiddenOutputs = apply(hiddenInputs, this::sigmoid);
        double[] finalInputs = multiply(who, hiddenOutputs);

        return apply(finalInputs, this::sigmoid);
    }

    private double[][] computeWeightsChange(double[] inputs, double[] outputs, double[] outputErrors) {
        int numOutputs = outputs.length;

        double[] outputOp = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            outputOp[i] = learningRate * outputErrors[i] * outputs[i] * (1.0 - outputs[i]);
        }

        return outerProduct(outputOp, inputs);
    }

    private double[][] outerProduct(double[] v1, double[] v2) {
        int rows = v1.length;
        int cols = v2.length;

        double[][] matrix = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = v1[i] * v2[j];
            }
        }

        return matrix;
    }

    private double[] activationFunction(double[] x) {
        double[] sigmoid = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            sigmoid[i] = sigmoid(x[i]);
        }
        return sigmoid;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "iNodes=" + iNodes +
                ", hNodes=" + hNodes +
                ", oNodes=" + oNodes +
                ", learningRate=" + learningRate +
                '}';
    }
}
