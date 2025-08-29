package org.example.loss;

public class LossFunction {
    
    public static double crossEntropy(double[] predictions, double[] targets) {
        double loss = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            loss += targets[i] * Math.log(predictions[i] + 1e-15);
        }
        return -loss;
    }
    
    public static double meanSquaredError(double[] predictions, double[] targets) {
        double loss = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            loss += Math.pow(predictions[i] - targets[i], 2);
        }
        return loss / predictions.length;
    }
}
