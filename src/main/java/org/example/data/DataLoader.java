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
                // نطبع الرقم مع تقريب بسيط (عشان ميكونش طويل قوي)
                System.out.printf("%.2f ", pixel);
            }
            System.out.println();
        }
    }
    
}

// import java.io.*;
// import java.net.*;
// import java.util.*;
// import java.util.zip.GZIPInputStream;

// public class DataLoader {
    
//     // URLs for MNIST dataset
//     private static final String TRAIN_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
//     private static final String TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
//     private static final String TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
//     private static final String TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
    
//     /**
//      * Download and extract MNIST dataset
//      */
//     public static void downloadMNIST() {
//         String[] urls = {TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL};
        
//         for (String url : urls) {
//             try {
//                 String filename = url.substring(url.lastIndexOf("/") + 1);
//                 File file = new File(filename);
                
//                 if (!file.exists()) {
//                     System.out.println("Downloading: " + filename);
//                     downloadFile(url, filename);
//                     extractGZIP(filename);
//                 }
//             } catch (Exception e) {
//                 System.err.println("Error downloading " + url + ": " + e.getMessage());
//             }
//         }
//     }
    
//     /**
//      * Download a file from URL
//      */
//     private static void downloadFile(String urlString, String filename) throws IOException {
//         URL url = new URL(urlString);
//         try (InputStream in = url.openStream();
//              FileOutputStream out = new FileOutputStream(filename)) {
            
//             byte[] buffer = new byte[8192];
//             int bytesRead;
//             while ((bytesRead = in.read(buffer)) != -1) {
//                 out.write(buffer, 0, bytesRead);
//             }
//         }
//     }
    
//     /**
//      * Extract GZIP file
//      */
//     private static void extractGZIP(String gzipFile) throws IOException {
//         String outputFile = gzipFile.replace(".gz", "");
        
//         try (GZIPInputStream gzipIn = new GZIPInputStream(new FileInputStream(gzipFile));
//              FileOutputStream out = new FileOutputStream(outputFile)) {
            
//             byte[] buffer = new byte[8192];
//             int bytesRead;
//             while ((bytesRead = gzipIn.read(buffer)) != -1) {
//                 out.write(buffer, 0, bytesRead);
//             }
//         }
        
//         // Delete the compressed file after extraction
//         new File(gzipFile).delete();
//     }
    
//     /**
//      * Load MNIST images from file
//      */
//     public static List<double[]> loadMNISTImages(String filename, int maxImages) throws IOException {
//         List<double[]> images = new ArrayList<>();
        
//         try (DataInputStream dataStream = new DataInputStream(new FileInputStream(filename))) {
//             // Read MNIST header
//             int magicNumber = dataStream.readInt();
//             int numImages = dataStream.readInt();
//             int numRows = dataStream.readInt();
//             int numCols = dataStream.readInt();
            
//             numImages = Math.min(numImages, maxImages);
//             int imageSize = numRows * numCols;
            
//             System.out.println("Loading " + numImages + " images from " + filename);
            
//             for (int i = 0; i < numImages; i++) {
//                 double[] image = new double[imageSize];
//                 for (int j = 0; j < imageSize; j++) {
//                     int pixel = dataStream.readUnsignedByte();
//                     image[j] = pixel / 255.0; // Normalize to [0, 1]
//                 }
//                 images.add(image);
                
//                 // Show progress
//                 if ((i + 1) % 1000 == 0) {
//                     System.out.println("Loaded " + (i + 1) + " images...");
//                 }
//             }
//         }
        
//         return images;
//     }
    
//     /**
//      * Load MNIST labels from file
//      */
//     public static List<Integer> loadMNISTLabels(String filename, int maxLabels) throws IOException {
//         List<Integer> labels = new ArrayList<>();
        
//         try (DataInputStream dataStream = new DataInputStream(new FileInputStream(filename))) {
//             // Read MNIST header
//             int magicNumber = dataStream.readInt();
//             int numLabels = dataStream.readInt();
            
//             numLabels = Math.min(numLabels, maxLabels);
            
//             System.out.println("Loading " + numLabels + " labels from " + filename);
            
//             for (int i = 0; i < numLabels; i++) {
//                 int label = dataStream.readUnsignedByte();
//                 labels.add(label);
                
//                 // Show progress
//                 if ((i + 1) % 1000 == 0) {
//                     System.out.println("Loaded " + (i + 1) + " labels...");
//                 }
//             }
//         }
        
//         return labels;
//     }
    
//     /**
//      * Convert label to one-hot encoding
//      */
//     public static double[] oneHotEncode(int label, int numClasses) {
//         double[] encoded = new double[numClasses];
//         encoded[label] = 1.0;
//         return encoded;
//     }
    
//     /**
//      * Split data into training and validation sets
//      */
//     public static <T> List[] splitData(List<T> data, List<T> labels, double trainRatio) {
//         int totalSize = data.size();
//         int trainSize = (int) (totalSize * trainRatio);
        
//         List<T> trainData = new ArrayList<>(data.subList(0, trainSize));
//         List<T> trainLabels = new ArrayList<>(labels.subList(0, trainSize));
//         List<T> valData = new ArrayList<>(data.subList(trainSize, totalSize));
//         List<T> valLabels = new ArrayList<>(labels.subList(trainSize, totalSize));
        
//         return new List[]{trainData, trainLabels, valData, valLabels};
//     }
// }