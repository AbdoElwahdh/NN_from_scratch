package org.example.activations;

/**
 * Softmax activation producing a probability distribution over classes.
 */
public class Softmax implements ActivationFunction {
    
    /**
     * Applies numerically-stable softmax.
     */
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        double max = input[0];
        for (double value : input) {
            if (value > max) max = value;
        }
        
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }
        
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }
        
        return output;
    }
    
    /**
     * Returns diagonal approximation of softmax Jacobian (for simplicity).
     * For training, combined gradient with cross-entropy is used instead.
     */
    @Override
    public double[] derivative(double[] input) {
        double[] softmax = activate(input);
        double[] derivative = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            derivative[i] = softmax[i] * (1 - softmax[i]);
        }
        return derivative;
    }
}