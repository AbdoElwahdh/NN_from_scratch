package org.example.layers;

import org.example.activations.ActivationFunction;
import org.example.mathematics.MatrixOperations;

/**
 * Base abstraction for network layers providing sizes, activation and parameter access.
 */
public abstract class Layer {
    protected int inputSize;
    protected int outputSize;
    protected ActivationFunction activation;
    
    public Layer(int inputSize, int outputSize, ActivationFunction activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
    }
    
    /** Performs forward pass given an input vector. */
    public abstract double[] forward(double[] input);
    /** Returns weight matrix (out x in). */
    public abstract double[][] getWeights();
    /** Returns bias vector (out). */
    public abstract double[] getBiases();
    /** Applies SGD-style weight update (used when no optimizer is set). */
    public abstract void updateWeights(double[][] weightGradients, double learningRate);
    /** Applies SGD-style bias update (used when no optimizer is set). */
    public abstract void updateBiases(double[] biasGradients, double learningRate);
    
    /** Input dimensionality for this layer. */
    public int getInputSize() {
        return inputSize;
    }
    
    /** Output dimensionality for this layer. */
    public int getOutputSize() {
        return outputSize;
    }
    
    /** Activation function used by this layer. */
    public ActivationFunction getActivation() {
        return activation;
    }
}