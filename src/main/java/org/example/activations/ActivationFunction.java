package org.example.activations;

/**
 * Defines the activation contract used by layers to transform pre-activations
 * and to provide derivatives for backpropagation.
 */
public interface ActivationFunction {
    /**
     * Applies the activation function element-wise on the given vector.
     * @param input pre-activation values
     * @return activated values
     */
    double[] activate(double[] input);

    /**
     * Computes the derivative of the activation function w.r.t. its input, element-wise.
     * @param input pre-activation values
     * @return derivative values
     */
    double[] derivative(double[] input);
}
