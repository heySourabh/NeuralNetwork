/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat;

public class NeuralNetwork {
    private final int iNodes;
    private final int hNodes;
    private final int oNodes;
    private final double learningRate;

    public NeuralNetwork(int numInputNodes, int numHiddenNodes, int numOutputNodes, double learningRate) {
        this.iNodes = numInputNodes;
        this.hNodes = numHiddenNodes;
        this.oNodes = numOutputNodes;
        this.learningRate = learningRate;
    }

    public void train() {
    }

    public double[] query() {
        return new double[0];
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
