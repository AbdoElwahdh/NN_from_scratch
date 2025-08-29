package org.example;

import org.example.layers.DenseLayer;
import org.example.layers.Layer;
import org.example.activations.*;
import org.example.mathematics.*;
import org.example.loss.LossFunction;
import org.example.optimizer.GradientDescent;
import org.example.optimizer.Optimizer;

import java.util.*;

public class NeuralNetwork {
    private List<Layer> layers;
    private Optimizer optimizer;
    private List<Double> trainingLoss;
    private List<Double> validationAccuracy;
    
    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.trainingLoss = new ArrayList<>();
        this.validationAccuracy = new ArrayList<>();
    }
    
    public void addLayer(Layer layer) {
        layers.add(layer);
    }
    
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }
    
    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    public void train(List<double[]> trainData, List<double[]> trainLabels, 
                     int epochs, int batchSize) {
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            
            for (int i = 0; i < trainData.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainData.size());
                
                // Process batch
                for (int j = i; j < end; j++) {
                    double[] input = trainData.get(j);
                    double[] target = trainLabels.get(j);
                    
                    // Forward pass
                    List<double[]> layerOutputs = new ArrayList<>();
                    double[] current = input;
                    layerOutputs.add(current);
                    
                    for (Layer layer : layers) {
                        current = layer.forward(current);
                        layerOutputs.add(current);
                    }
                    
                    // Calculate loss
                    double loss = LossFunction.crossEntropy(current, target);
                    epochLoss += loss;
                    
                    // Backward pass
                    double[] error = Backpropagation.calculateOutputError(current, target);
                    
                    // Backpropagate through layers
                    for (int k = layers.size() - 1; k >= 0; k--) {
                        Layer layer = layers.get(k);
                        double[] layerInput = layerOutputs.get(k);
                        double[] layerOutput = layerOutputs.get(k + 1);
                        
                        // Calculate gradients
                        double[][] weightGradients = Gradient.calculateWeightGradient(layerInput, error);
                        double[] biasGradients = Gradient.calculateBiasGradient(error);
                        
                        // Update weights and biases
                        layer.updateWeights(weightGradients, 0.01); // Using fixed learning rate for simplicity
                        layer.updateBiases(biasGradients, 0.01);
                        
                        // Calculate error for next layer
                        if (k > 0) {

                          // Calculate pre-activation values for derivative
                            double[] preActivation = MatrixOperations.multiply(layer.getWeights(), layerInput);
                            for (int b = 0; b < preActivation.length; b++) {
                                preActivation[b] += layer.getBiases()[b];
                            }
                            double[] derivative = layer.getActivation().derivative(preActivation);
                            // derivative has the same length as error (nextError)
                            if (derivative.length != error.length) {
                                System.err.println("Warning: derivative length (" + derivative.length + 
                                                 ") != error length (" + error.length + ")");
                            }
                            error = Backpropagation.calculateHiddenError(layer.getWeights(), error, derivative);
                        }
                    }
                }
            }
            
            // Record average loss for this epoch
            trainingLoss.add(epochLoss / trainData.size());
            System.out.printf("Epoch %d, Loss: %.4f\n", epoch + 1, epochLoss / trainData.size());
        }
    }
    
    public double evaluate(List<double[]> testData, List<double[]> testLabels) {
        int correct = 0;
        
        for (int i = 0; i < testData.size(); i++) {
            double[] prediction = forward(testData.get(i));
            double[] target = testLabels.get(i);
            
            int predictedClass = argmax(prediction);
            int actualClass = argmax(target);
            
            if (predictedClass == actualClass) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / testData.size();
        validationAccuracy.add(accuracy);
        return accuracy;
    }
    
    private int argmax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    public List<Double> getTrainingLoss() {
        return trainingLoss;
    }
    
    public List<Double> getValidationAccuracy() {
        return validationAccuracy;
    }
}