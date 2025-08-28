package org.example.activations;

public interface ActivationFunction {
    double[] activate(double[] input);
    double[] derivative(double[] input);
}
