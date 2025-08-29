package org.example.data;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class DataLoader {

    public static List<double[]> loadMNISTImages(String filename, int maxImages) throws IOException {
        List<double[]> images = new ArrayList<>();
        byte[] data = Files.readAllBytes(Paths.get(filename));

        int offset = 16; // MNIST header size
        int imageSize = 28 * 28;

        int numImages = Math.min(maxImages, (data.length - offset) / imageSize);

        for (int i = 0; i < numImages; i++) {
            double[] image = new double[imageSize];
            for (int j = 0; j < imageSize; j++) {
                int pixel = data[offset + i * imageSize + j] & 0xFF;
                image[j] = pixel / 255.0; // Normalize to [0, 1]
            }
            images.add(image);
        }

        return images;
    }

    public static List<Integer> loadMNISTLabels(String filename, int maxLabels) throws IOException {
        List<Integer> labels = new ArrayList<>();
        byte[] data = Files.readAllBytes(Paths.get(filename));

        int offset = 8; // MNIST label header size

        int numLabels = Math.min(maxLabels, data.length - offset);

        for (int i = 0; i < numLabels; i++) {
            labels.add((int) data[offset + i]);
        }

        return labels;
    }
  
    public static double[] oneHotEncode(int label, int numClasses) {
        double[] encoded = new double[numClasses];
        encoded[label] = 1.0;
        return encoded;
    }

    // 🔹 Helper method to print first image as 28x28 matrix
    public static void printImage(double[] image) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double pixel = image[i * 28 + j];
                // Print pixel value with simple rounding (to avoid very long numbers)
                System.out.printf("%.2f ", pixel);
            }
            System.out.println();
        }
    }
    