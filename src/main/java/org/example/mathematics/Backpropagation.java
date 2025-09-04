package org.example.mathematics;


/**
 * Gradients for output and hidden layers used during backpropagation.
 */
public class Backpropagation {
    
    /**
     * For cross-entropy with softmax, this equals (y_hat - y).
     */
    public static double[] calculateOutputError(double[] output, double[] target) {
        double[] error = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            error[i] = output[i] - target[i];
        }
        return error;
    }
    
    /**
     * Propagates error to previous layer using W^T * nextError element-wise times activation derivative.
     */
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