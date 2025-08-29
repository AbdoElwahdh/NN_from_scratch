package org.example.activations;


public class Sigmoid implements ActivationFunction {
    
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }
        return output;
    }
    
    @Override
    public double[] derivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-input[i]));
            output[i] = sigmoid * (1 - sigmoid);
        }
        return output;
    }
}