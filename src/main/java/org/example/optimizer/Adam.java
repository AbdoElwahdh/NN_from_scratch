package org.example.optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Adam optimizer with per-parameter first (m) and second (v) moments.
 */
public class Adam implements Optimizer {
	private final double learningRate;
	private final double beta1;
	private final double beta2;
	private final double epsilon;

	private long timestep;

	private final Map<double[][], double[][]> mWeights;
	private final Map<double[][], double[][]> vWeights;
	private final Map<double[], double[]> mBiases;
	private final Map<double[], double[]> vBiases;

	public Adam(double learningRate, double beta1, double beta2, double epsilon) {
		this.learningRate = learningRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
		this.timestep = 0L;
		this.mWeights = new IdentityHashMap<>();
		this.vWeights = new IdentityHashMap<>();
		this.mBiases = new IdentityHashMap<>();
		this.vBiases = new IdentityHashMap<>();
	}

	@Override
	/**
	 * Applies Adam update to a weight matrix.
	 */
	public void updateWeights(double[][] weights, double[][] gradients) {
		if (weights == null || gradients == null) return;
		this.timestep++;
		double[][] m = mWeights.computeIfAbsent(weights, k -> new double[weights.length][weights[0].length]);
		double[][] v = vWeights.computeIfAbsent(weights, k -> new double[weights.length][weights[0].length]);

		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				double g = gradients[i][j];
				m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;
				v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g;
				double mHat = m[i][j] / (1.0 - Math.pow(beta1, timestep));
				double vHat = v[i][j] / (1.0 - Math.pow(beta2, timestep));
				weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
			}
		}
	}

	@Override
	/**
	 * Applies Adam update to a bias vector.
	 */
	public void updateBiases(double[] biases, double[] gradients) {
		if (biases == null || gradients == null) return;
		this.timestep++;
		double[] m = mBiases.computeIfAbsent(biases, k -> new double[biases.length]);
		double[] v = vBiases.computeIfAbsent(biases, k -> new double[biases.length]);

		for (int i = 0; i < biases.length; i++) {
			double g = gradients[i];
			m[i] = beta1 * m[i] + (1.0 - beta1) * g;
			v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
			double mHat = m[i] / (1.0 - Math.pow(beta1, timestep));
			double vHat = v[i] / (1.0 - Math.pow(beta2, timestep));
			biases[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
		}
	}
}
