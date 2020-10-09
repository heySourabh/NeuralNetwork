/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

import java.util.Random;

import static in.spbhat.util.DoubleArray.apply;
import static in.spbhat.util.DoubleMatrix.createRandom;
import static in.spbhat.util.DoubleMatrix.multiply;
import static java.lang.Math.exp;

public class NeuralNetwork {
    private static final Random rng = new Random(123);

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

    public void train() {
    }

    public double[] query(double[] inputs) {
        double[] hiddenInputs = multiply(wih, inputs);
        double[] hiddenOutputs = apply(hiddenInputs, this::sigmoid);
        double[] finalInputs = multiply(who, hiddenOutputs);

        return apply(finalInputs, this::sigmoid);
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

    private double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
}
