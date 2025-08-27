package org.example.neuralnet;

import org.example.neuralnet.math.Matrix;

import java.util.function.Function;

/**
 * Represents a simple feedforward neural network with one hidden layer.
 */
public class NeuralNetwork {

    private final int inputNodes;
    private final int hiddenNodes;
    private final int outputNodes;

    private Matrix weightsInputToHidden;
    private Matrix weightsHiddenToOutput;

    private Matrix biasHidden;
    private Matrix biasOutput;

    // Activation function (we will use Sigmoid )
    // We pass it as a function to make our network more flexible in the future.
    private final Function<Double, Double> activationFunction = x -> 1 / (1 + Math.exp(-x));

    /**
     * Creates a new Neural Network with a specified architecture.
     *
     * @param inputNodes  The number of nodes in the input layer.
     * @param hiddenNodes The number of nodes in the hidden layer.
     * @param outputNodes The number of nodes in the output layer.
     */
    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Initialize weights with random values
        this.weightsInputToHidden = new Matrix(this.hiddenNodes, this.inputNodes);
        this.weightsInputToHidden.randomize();

        this.weightsHiddenToOutput = new Matrix(this.outputNodes, this.hiddenNodes);
        this.weightsHiddenToOutput.randomize();

        // Initialize biases with random values
        this.biasHidden = new Matrix(this.hiddenNodes, 1);
        this.biasHidden.randomize();

        this.biasOutput = new Matrix(this.outputNodes, 1);
        this.biasOutput.randomize();
    }
    /**
     * Performs the feedforward process.
     * Takes an input array and returns the network's prediction.
     *
     * @param inputArray The input data as a 1D array.
     * @return A 1D array representing the network's output.
     */
    public double[] feedForward(double[] inputArray) {
        // Step 1: Convert input array to a Matrix
        Matrix inputs = Matrix.fromArray(inputArray);

        // Step 2: Calculate the hidden layer outputs
        // Formula: hidden_outputs = activation_function( (W_ih * I) + B_h )
        Matrix hidden = this.weightsInputToHidden.multiply(inputs);
        hidden = hidden.add(this.biasHidden);
        // Apply the activation function to every element
        hidden = hidden.map(this.activationFunction);

        // Step 3: Calculate the final output
        // Formula: final_outputs = activation_function( (W_ho * H) + B_o )
        Matrix output = this.weightsHiddenToOutput.multiply(hidden);
        output = output.add(this.biasOutput);
        // Apply the activation function to every element
        output = output.map(this.activationFunction);

        // Step 4: Convert the final output Matrix back to an array and return it
        return output.toArray();
    }

}
