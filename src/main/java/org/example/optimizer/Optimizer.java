package org.example.optimizer;


/**
 * Optimizer updates parameters given gradients.
 */
public interface Optimizer {
    void updateWeights(double[][] weights, double[][] gradients);
    void updateBiases(double[] biases, double[] gradients);
}
