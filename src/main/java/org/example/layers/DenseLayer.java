package org.example.layers;

import org.example.activations.ActivationFunction;
import org.example.mathematics.MatrixOperations;

public class DenseLayer extends Layer {
    private double[][] weights;
    private double[] biases;

    /**
     * Fully-connected layer with weights (out x in) and biases (out).
     */
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        super(inputSize, outputSize, activation);
        initializeWeights();
    }

    /**
     * Initializes weights with He init and small positive biases.
     */
    private void initializeWeights() {
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];

        // He initialization for ReLU-like activations; reasonable default for others
        double std = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                // Sample from a normal-ish via Box-Muller lite using Math.random()
                double u1 = Math.random();
                double u2 = Math.random();
                double z = Math.sqrt(-2.0 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
                weights[i][j] = std * z;
            }
            biases[i] = 0.01;
        }
    }

    /**
     * z = W x + b, then apply activation.
     */
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

    // ==================================================================
    // === الدوال الجديدة التي تمت إضافتها لتحميل النموذج ===
    // ==================================================================

    /**
     * Sets the weights for this layer. Used when loading a pre-trained model.
     * @param weights The weights to set.
     */
    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    /**
     * Sets the biases for this layer. Used when loading a pre-trained model.
     * @param biases The biases to set.
     */
    public void setBiases(double[] biases) {
        this.biases = biases;
    }
    // ==================================================================

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
