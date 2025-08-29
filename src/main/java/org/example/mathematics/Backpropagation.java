package org.example.mathematics;


public class Backpropagation {
    
    public static double[] calculateOutputError(double[] output, double[] target) {
        double[] error = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            error[i] = output[i] - target[i];
        }
        return error;
    }
    
    public static double[] calculateHiddenError(double[][] weights, double[] nextError, double[] derivative) {
        double[] error = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            for (int i = 0; i < weights.length; i++) {
                error[j] += weights[i][j] * nextError[i] * derivative[i];
            }
        }
        return error;
    }
}