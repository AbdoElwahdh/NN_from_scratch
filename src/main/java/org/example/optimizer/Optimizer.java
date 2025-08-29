package org.example.optimizer;


public interface Optimizer {
    void updateWeights(double[][] weights, double[][] gradients);
    void updateBiases(double[] biases, double[] gradients);
}
