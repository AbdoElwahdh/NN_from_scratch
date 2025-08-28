package org.example.mathematics;

public class Gradient {
    
    public static double[][] calculateWeightGradient(double[] input, double[] error) {
        double[][] gradient = new double[error.length][input.length];
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < input.length; j++) {
                gradient[i][j] = error[i] * input[j];
            }
        }
        return gradient;
    }
    
    public static double[] calculateBiasGradient(double[] error) {
        return error.clone();
    }
}