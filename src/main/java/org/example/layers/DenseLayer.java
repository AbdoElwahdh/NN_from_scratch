package org.example.layers;

import org.example.activations.ActivationFunction;
import org.example.mathematics.MatrixOperations;

public class DenseLayer extends Layer {
    private double[][] weights;
    private double[] biases;
    
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        super(inputSize, outputSize, activation);
        initializeWeights();
    }
    
    private void initializeWeights() {
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        
        // Xavier initialization
        double std = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = std * Math.random() - std / 2;
            }
            biases[i] = 0.01;
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        double[] z = MatrixOperations.multiply(weights, input);
        for (int i = 0; i < z.length; i++) {
            z[i] += biases[i];
        }
        return activation.activate(z);
    }
    
    @Override
    public double[][] getWeights() {
        return weights;
    }
    
    @Override
    public double[] getBiases() {
        return biases;
    }
    
    @Override
    public void updateWeights(double[][] weightGradients, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
        }
    }
    
    @Override
    public void updateBiases(double[] biasGradients, double learningRate) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * biasGradients[i];
        }
    }
}