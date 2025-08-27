package org.example.neuralnet.math;

import java.util.Random;
import java.util.function.Function;

/**
 * A class representing a 2D Matrix, which is the fundamental data structure for our neural network.
 * It contains the data and the basic operations needed for calculations.
 */
public class Matrix {

    private final int rows;
    private final int cols;
    private final double[][] data;
    private static final Random random = new Random();

    /**
     * Creates a new Matrix with the given number of rows and columns.
     * Initializes all elements to zero.
     *
     * @param rows The number of rows.
     * @param cols The number of columns.
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    /**
     * Fills the matrix with random values between -1 and 1.
     * This is used for initializing the weights of the neural network.
     */
    public void randomize() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Random values between -1 and 1
                this.data[i][j] = random.nextDouble() * 2 - 1;
            }
        }
    }

    /**
     * Performs matrix addition.
     *
     * @param other The matrix to add.
     * @return A new Matrix object that is the sum of this matrix and the other.
     * @throws IllegalArgumentException if the matrices do not have the same dimensions.
     */
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for addition.");
        }

        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    /**
     * Performs matrix multiplication (dot product).
     *
     * @param other The matrix to multiply by.
     * @return A new Matrix object that is the product of this matrix and the other.
     * @throws IllegalArgumentException if the number of columns in this matrix does not match the number of rows in the other.
     */
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix A's columns must match Matrix B's rows for multiplication.");
        }

        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * Creates a new Matrix from a 1D array.
     * The resulting matrix will be a column vector (n rows, 1 column).
     *
     * @param array The input array.
     * @return A new Matrix object.
     */
    public static Matrix fromArray(double[] array) {
        Matrix result = new Matrix(array.length, 1);
        for (int i = 0; i < array.length; i++) {
            result.data[i][0] = array[i];
        }
        return result;
    }
    /**
     * Applies a function to every element of the matrix.
     *
     * @param func The function to apply.
     * @return A new Matrix with the function applied to each element.
     */
    public Matrix map(Function<Double, Double> func) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                double originalValue = this.data[i][j];
                result.data[i][j] = func.apply(originalValue);
            }
        }
        return result;
    }
    /**
     * Converts a single-column matrix back into a 1D array.
     *
     * @return A 1D double array.
     * @throws IllegalStateException if the matrix has more than one column.
     */
    public double[] toArray() {
        if (this.cols != 1) {
            throw new IllegalStateException("Cannot convert a matrix with more than one column to an array.");
        }
        double[] result = new double[this.rows];
        for (int i = 0; i < this.rows; i++) {
            result[i] = this.data[i][0];
        }
        return result;
    }



    /**
     * Prints the matrix to the console for debugging purposes.
     */
    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(this.data[i][j] + " ");
            }
            System.out.println();
        }
    }
}
