package org.example;

import org.example.activations.ReLU;
import org.example.activations.Softmax;
import org.example.data.DataLoader;
import org.example.layers.DenseLayer;
import org.example.layers.Layer;
import org.example.NeuralNetwork;


import java.util.List;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        try {
            // Load MNIST dataset

            String trainImages_path = "data/train-images-idx3-ubyte/train-images-idx3-ubyte"; 
            String trainLabels_path = "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte"; 
            String testImages_path  = "data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"; 
            String testLabels_path  = "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
            

            List<double[]> trainImages = DataLoader.loadMNISTImages(trainImages_path, 1000);
            List<Integer> trainLabels = DataLoader.loadMNISTLabels(trainLabels_path, 1000);
            List<double[]> testImages = DataLoader.loadMNISTImages(testImages_path, 200);
            List<Integer> testLabels = DataLoader.loadMNISTLabels(testLabels_path, 200);
            
            // print first training label and image
            // System.out.println("First training label: " + trainLabels.get(0));
            // System.out.println("First training image:");
            // DataLoader.printImage(trainImages.get(0));
            
            // Convert labels to one-hot encoding
            List<double[]> trainLabelsOneHot = new ArrayList<>();
            for (Integer label : trainLabels) {
                trainLabelsOneHot.add(DataLoader.oneHotEncode(label, 10));
            }
            
            List<double[]> testLabelsOneHot = new ArrayList<>();
            for (Integer label : testLabels) {
                testLabelsOneHot.add(DataLoader.oneHotEncode(label, 10));
            }
            
            // Create neural network
            NeuralNetwork nn = new NeuralNetwork();
            
            // Add layers
            Layer hiddenLayer = new DenseLayer(784, 128, new ReLU());
            Layer outputLayer = new DenseLayer(128, 10, new Softmax());
            
            nn.addLayer(hiddenLayer);
            nn.addLayer(outputLayer);
            
            // Train the network
            System.out.println("Training neural network...");
            nn.train(trainImages, trainLabelsOneHot, 10, 32);
            
            // Evaluate the network
            double accuracy = nn.evaluate(testImages, testLabelsOneHot);
            System.out.printf("Test accuracy: %.2f%%\n", accuracy * 100);
            
            // Print training history
            System.out.println("\nTraining history:");
            List<Double> lossHistory = nn.getTrainingLoss();
            for (int i = 0; i < lossHistory.size(); i++) {
                System.out.printf("Epoch %d: Loss = %.4f\n", i + 1, lossHistory.get(i));
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

