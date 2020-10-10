/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import in.spbhat.util.DoubleMatrix;

import java.util.Random;

import static in.spbhat.util.DoubleArray.apply;
import static in.spbhat.util.DoubleArray.multiply;
import static in.spbhat.util.DoubleArray.subtract;
import static in.spbhat.util.DoubleArray.*;
import static in.spbhat.util.DoubleMatrix.add;
import static in.spbhat.util.DoubleMatrix.multiply;
import static in.spbhat.util.DoubleMatrix.*;
import static java.lang.Math.exp;

public class NeuralNetwork {
    private static final Random rng = new Random(31416);

    private final int iNodes;
    private final int hNodes;
    private final int oNodes;
    private final double learningRate;
    private final double[][] wih;
    private final double[][] who;

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
        double[] hiddenOutputs = apply(hiddenInputs, this::sigmoid);
        double[] finalInputs = multiply(who, hiddenOutputs);
        double[] finalOutputs = apply(finalInputs, this::sigmoid);

        double[] outputErrors = subtract(targets, finalOutputs);
        double[] hiddenErrors = multiply(transpose(who), outputErrors);

        double[] ones = newFilledArray(finalOutputs.length, 1.0);
        double[] oneMinusFinalOutputs = subtract(ones, finalOutputs);
        double[][] op1 = {multiply(multiply(outputErrors, finalOutputs), oneMinusFinalOutputs)};
        double[][] op2 = {hiddenOutputs};
        double[][] changeWeights = multiply(multiply(transpose(op1), op2), learningRate);
        DoubleMatrix.copy(add(who, changeWeights), who);

        ones = newFilledArray(hiddenOutputs.length, 1.0);
        double[] oneMinusHiddenOutputs = subtract(ones, hiddenOutputs);
        op1 = new double[][]{multiply(multiply(hiddenErrors, hiddenOutputs), oneMinusHiddenOutputs)};
        op2 = new double[][]{inputs};
        changeWeights = multiply(multiply(transpose(op1), op2), learningRate);
        DoubleMatrix.copy(add(wih, changeWeights), wih);

        System.out.println(stringify(who));
    }

    public double[] query(double[] inputs) {
        double[] hiddenInputs = multiply(wih, inputs);
        double[] hiddenOutputs = apply(hiddenInputs, this::sigmoid);
        double[] finalInputs = multiply(who, hiddenOutputs);

        return apply(finalInputs, this::sigmoid);
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
