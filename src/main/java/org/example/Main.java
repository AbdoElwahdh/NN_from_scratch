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
            List<double[]> trainImages = DataLoader.loadMNISTImages("train-images-idx3-ubyte", 1000);
            List<Integer> trainLabels = DataLoader.loadMNISTLabels("train-labels-idx1-ubyte", 1000);
            List<double[]> testImages = DataLoader.loadMNISTImages("t10k-images-idx3-ubyte", 200);
            List<Integer> testLabels = DataLoader.loadMNISTLabels("t10k-labels-idx1-ubyte", 200);
            
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

// import java.util.*;
// import java.io.*;

// import org.example.activations.ReLU;
// import org.example.activations.Softmax;
// import org.example.data.DataLoader;
// import org.example.layers.DenseLayer;
// import org.example.layers.Layer;
// import org.example.NeuralNetwork;

// public class Main {
//     public static void main(String[] args) {
//         try {
//             // Download MNIST dataset if not exists
//             System.out.println("Checking for MNIST dataset...");
//             DataLoader.downloadMNIST();
            
//             // Load MNIST data
//             System.out.println("Loading training data...");
//             List<double[]> trainImages = DataLoader.loadMNISTImages("train-images-idx3-ubyte", 60000);
//             List<Integer> trainLabels = DataLoader.loadMNISTLabels("train-labels-idx1-ubyte", 60000);
            
//             System.out.println("Loading test data...");
//             List<double[]> testImages = DataLoader.loadMNISTImages("t10k-images-idx3-ubyte", 10000);
//             List<Integer> testLabels = DataLoader.loadMNISTLabels("t10k-labels-idx1-ubyte", 10000);
            
//             // Convert labels to one-hot encoding
//             List<double[]> trainLabelsOneHot = new ArrayList<>();
//             for (Integer label : trainLabels) {
//                 trainLabelsOneHot.add(DataLoader.oneHotEncode(label, 10));
//             }
            
//             List<double[]> testLabelsOneHot = new ArrayList<>();
//             for (Integer label : testLabels) {
//                 testLabelsOneHot.add(DataLoader.oneHotEncode(label, 10));
//             }
            
//             // Split training data into train/validation
//             List[] splitData = DataLoader.splitData(trainImages, trainLabelsOneHot, 0.8);
//             List<double[]> trainData = splitData[0];
//             List<double[]> trainDataLabels = splitData[1];
//             List<double[]> valData = splitData[2];
//             List<double[]> valDataLabels = splitData[3];
            
//             // Create neural network
//             System.out.println("Creating neural network...");
//             NeuralNetwork nn = new NeuralNetwork();
            
//             // Add layers
//             nn.addLayer(new DenseLayer(784, 128, new ReLU()));
//             nn.addLayer(new DenseLayer(128, 64, new ReLU()));
//             nn.addLayer(new DenseLayer(64, 10, new Softmax()));
            
//             // Train the network
//             System.out.println("Training neural network...");
//             nn.train(trainData, trainDataLabels, 10, 64);
            
//             // Evaluate on validation set
//             System.out.println("Evaluating on validation set...");
//             double valAccuracy = nn.evaluate(valData, valDataLabels);
//             System.out.printf("Validation accuracy: %.2f%%\n", valAccuracy * 100);
            
//             // Evaluate on test set
//             System.out.println("Evaluating on test set...");
//             double testAccuracy = nn.evaluate(testImages, testLabelsOneHot);
//             System.out.printf("Test accuracy: %.2f%%\n", testAccuracy * 100);
            
//             // Save training history to file
//             saveTrainingHistory(nn.getTrainingLoss(), nn.getValidationAccuracy());
            
//         } catch (Exception e) {
//             System.err.println("Error: " + e.getMessage());
//             e.printStackTrace();
//         }
//     }
    
//     /**
//      * Save training history to CSV file
//      */
//     private static void saveTrainingHistory(List<Double> lossHistory, List<Double> accuracyHistory) {
//         try (PrintWriter writer = new PrintWriter("training_history.csv")) {
//             writer.println("Epoch,Loss,Accuracy");
//             for (int i = 0; i < lossHistory.size(); i++) {
//                 double accuracy = i < accuracyHistory.size() ? accuracyHistory.get(i) : 0.0;
//                 writer.printf("%d,%.4f,%.4f\n", i + 1, lossHistory.get(i), accuracy);
//             }
//             System.out.println("Training history saved to training_history.csv");
//         } catch (IOException e) {
//             System.err.println("Error saving training history: " + e.getMessage());
//         }
//     }
// }