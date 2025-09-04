package org.example.mathematics;

public class Gradient {
    
    /**
     * Outer product error (out) x input (in) for dense layer weight gradients.
     */
    public static double[][] calculateWeightGradient(double[] input, double[] error) {
        double[][] gradient = new double[error.length][input.length];
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++) {
                gradient[i][j] = error[i] * input[j];
            }
        }
        return gradient;
    }
    
    /**
     * Bias gradient equals the error itself for dense layers.
     */
    public static double[] calculateBiasGradient(double[] error) {
        return error.clone();
    }
}