package org.example.optimizer;


public class GradientDescent implements Optimizer {
    private double learningRate;
    
    public GradientDescent(double learningRate) {
        this.learningRate = learningRate;
    }
    
    @Override
    public void updateWeights(double[][] weights, double[][] gradients) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * gradients[i][j];
            }
        }
    }
    
    @Override
    public void updateBiases(double[] biases, double[] gradients) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * gradients[i];
        }
    }
}
