package org.example.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.example.NeuralNetwork;
import org.example.activations.*;
import org.example.layers.DenseLayer;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.Optional;

@Service
public class NeuralNetworkService {

    private NeuralNetwork neuralNetwork;
    private final String ARTIFACTS_DIR = "artifacts";

    @PostConstruct
    public void init() throws IOException {
        System.out.println("Loading the neural network model...");
        Optional<File> latestModelFile = findLatestModelFile(ARTIFACTS_DIR);

        if (latestModelFile.isPresent()) {
            File modelFile = latestModelFile.get();
            System.out.println("Found model file: " + modelFile.getName());
            this.neuralNetwork = loadModel(modelFile);
            System.out.println("Model loaded successfully!");
        } else {
            throw new IOException("FATAL: No model file found in '" + ARTIFACTS_DIR + "' directory.");
        }
    }

    public int predict(double[] imageInput) {
        if (this.neuralNetwork == null) {
            throw new IllegalStateException("Neural network is not loaded.");
        }
        double[] predictions = this.neuralNetwork.forward(imageInput);
        int maxIndex = 0;
        for (int i = 1; i < predictions.length; i++) {
            if (predictions[i] > predictions[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private NeuralNetwork loadModel(File modelFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode rootNode = mapper.readTree(modelFile);
        NeuralNetwork nn = new NeuralNetwork();
        JsonNode layersNode = rootNode.path("architecture").path("layers");
        JsonNode paramsNode = rootNode.path("parameters");

        for (int i = 0; i < layersNode.size(); i++) {
            JsonNode layerInfo = layersNode.get(i);
            int inputSize = layerInfo.path("input_size").asInt();
            int outputSize = layerInfo.path("output_size").asInt();
            String activationName = layerInfo.path("activation").asText();
            ActivationFunction activation = getActivationFunction(activationName);
            DenseLayer layer = new DenseLayer(inputSize, outputSize, activation);

            JsonNode layerParams = paramsNode.get(i);
            double[][] weights = mapper.convertValue(layerParams.path("weights"), double[][].class);
            double[] biases = mapper.convertValue(layerParams.path("biases"), double[].class);

            layer.setWeights(weights);
            layer.setBiases(biases);
            nn.addLayer(layer);
        }
        return nn;
    }

    private ActivationFunction getActivationFunction(String name) {
        return switch (name) {
            case "ReLU" -> new ReLU();
            case "Softmax" -> new Softmax();
            case "Sigmoid" -> new Sigmoid();
            default -> throw new IllegalArgumentException("Unknown activation function: " + name);
        };
    }

    private Optional<File> findLatestModelFile(String dir) throws IOException {
        Path dirPath = Paths.get(dir);
        if (!Files.exists(dirPath) || !Files.isDirectory(dirPath)) return Optional.empty();
        return Files.list(dirPath)
                .filter(path -> path.toString().endsWith(".json"))
                .map(Path::toFile)
                .max(Comparator.comparingLong(File::lastModified));
    }
}
