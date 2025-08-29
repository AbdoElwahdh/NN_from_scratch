package org.example.layers;

import org.example.activations.ActivationFunction;
import org.example.mathematics.MatrixOperations;

public abstract class Layer {
    protected int inputSize;
    protected int outputSize;
    protected ActivationFunction activation;
    
    public Layer(int inputSize, int outputSize, ActivationFunction activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
    }
    
    public abstract double[] forward(double[] input);
    public abstract double[][] getWeights();
    public abstract double[] getBiases();
    public abstract void updateWeights(double[][] weightGradients, double learningRate);
    public abstract void updateBiases(double[] biasGradients, double learningRate);
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    public ActivationFunction getActivation() {
        return activation;
    }
}