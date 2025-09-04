package org.example;

import org.example.layers.Layer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Utilities for saving model architecture and parameters to a JSON file.
 */
public class ModelIO {
    public static void save(NeuralNetwork nn, String filePath) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        // architecture
        sb.append("  \"architecture\": {\n");
        List<Layer> layers = nn.getLayers();
        sb.append("    \"num_layers\": ").append(layers.size()).append(",\n");
        sb.append("    \"layers\": [\n");
        for (int idx = 0; idx < layers.size(); idx++) {
            Layer layer = layers.get(idx);
            sb.append("      {");
            sb.append("\"type\": \"dense\",");
            sb.append("\"input_size\": ").append(layer.getInputSize()).append(",");
            sb.append("\"output_size\": ").append(layer.getOutputSize()).append(",");
            sb.append("\"activation\": \"").append(layer.getActivation().getClass().getSimpleName()).append("\"");
            sb.append("}");
            if (idx < layers.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ]\n");
        sb.append("  },\n");

        // weights
        sb.append("  \"parameters\": [\n");
        for (int idx = 0; idx < layers.size(); idx++) {
            Layer layer = layers.get(idx);
            double[][] w = layer.getWeights();
            double[] b = layer.getBiases();
            sb.append("    {\n");
            sb.append("      \"weights\": ").append(serialize2D(w)).append(",\n");
            sb.append("      \"biases\": ").append(serialize1D(b)).append("\n");
            sb.append("    }");
            if (idx < layers.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");

        ensureParentDirectory(filePath);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(sb.toString());
        }
    }

    private static String serialize1D(double[] arr) {
        if (arr == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(Double.toString(arr[i]));
            if (i < arr.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String serialize2D(double[][] arr) {
        if (arr == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(serialize1D(arr[i]));
            if (i < arr.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    private static void ensureParentDirectory(String filePath) throws IOException {
        Path p = Paths.get(filePath).toAbsolutePath();
        Path parent = p.getParent();
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent);
        }
    }
}


