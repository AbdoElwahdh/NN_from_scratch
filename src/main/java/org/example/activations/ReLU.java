package org.example.activations;

/**
 * Rectified Linear Unit activation: max(0, x).
 */
public class ReLU implements ActivationFunction {
    
    /**
     * Returns max(0, x) element-wise.
     */
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
    
    /**
     * Returns 1 for x>0, otherwise 0, element-wise.
     */
    @Override
    public double[] derivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? 1.0 : 0.0;
        }
        return output;
    }
}
