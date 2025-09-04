package org.example.activations;

/**
 * Sigmoid activation: 1 / (1 + e^-x).
 */
public class Sigmoid implements ActivationFunction {
    
    /**
     * Applies sigmoid element-wise.
     */
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }
        return output;
    }
    
    /**
     * Computes sigmoid'(x) = s(x) * (1 - s(x)) element-wise.
     */
    @Override
    public double[] derivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-input[i]));
            output[i] = sigmoid * (1 - sigmoid);
        }
        return output;
    }
}