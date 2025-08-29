package org.example.activations;


public class ReLU implements ActivationFunction {
    
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
    
    @Override
    public double[] derivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? 1.0 : 0.0;
        }
        return output;
    }
}
