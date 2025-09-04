# Neural Network from Scratch in Java

A simple implementation of a neural network from scratch in Java, designed to classify handwritten digits using the MNIST dataset.

## Project Overview

This project implements a basic neural network with the following features:
- **Dense Layers**: Fully connected neural network layers
- **Activation Functions**: ReLU, Sigmoid and Softmax activation functions
- **Backpropagation**: Gradient-based learning algorithm
- **MNIST Dataset**: Handwritten digit classification
- **Artifacts Saving**: Saves architecture + learned weights/biases to JSON under `artifacts/`
- **Maven or Plain javac Run**: Build and run via Maven or Java CLI

## Project Structure

```
src/main/java/org/example/
├── activations/          # Activation functions
│   ├── ActivationFunction.java
│   ├── ReLU.java
│   ├── Sigmoid.java
│   └── Softmax.java
├── data/                 # Data loading and preprocessing
│   └── DataLoader.java
├── layers/               # Neural network layers
│   ├── DenseLayer.java
│   └── Layer.java
├── loss/                 # Loss functions
│   └── LossFunction.java
├── mathematics/          # Mathematical operations
│   ├── Backpropagation.java
│   ├── Gradient.java
│   └── MatrixOperations.java
├── optimizer/            # Optimization algorithms
│   ├── Adam.java
│   ├── GradientDescent.java
│   └── Optimizer.java
├── Main.java            # Main application entry point
└── NeuralNetwork.java   # Neural network implementation
```

## Features

### Core Components
- **Neural Network**: Main network class that manages layers and training
- **Layers**: Dense layers with configurable input/output dimensions
- **Activation Functions**: ReLU, Sigmoid, and Softmax implementations
- **Optimizers**: Gradient Descent and Adam optimization algorithms
- **Data Loading**: MNIST dataset loader with preprocessing

### Mathematical Operations
- Matrix operations for forward and backward propagation
- Gradient computation for backpropagation
- Loss function calculations

## Requirements

- Java 20 or higher
- Maven 3.6 or higher (optional if you use plain `javac`/`java`)
- MNIST dataset files (included in data/ folder)

## Run

1) Clone the repository:
```bash
git clone <repository-url>
cd NN_from_scratch_java
```

2) Option A — Run with Maven:
```bash
mvn -q compile
mvn -q exec:java -Dexec.mainClass="org.example.Main"
```

2) Option B — Run with javac/java (Windows PowerShell):
```powershell
mkdir -Force target\classes
$files = Get-ChildItem -Recurse -Filter *.java src\main\java | ForEach-Object { $_.FullName }
javac -encoding UTF-8 -d target\classes $files
java -cp target\classes org.example.Main
```

## Usage

The main application loads the MNIST dataset and demonstrates:
- Data loading and preprocessing
- Neural network creation with dense layers
- Training with backpropagation
- Model evaluation and accuracy calculation
- Model saving to JSON under `artifacts/`

## Configuration

You can modify the neural network architecture in `Main.java`:
- Change layer dimensions
- Adjust learning rate
- Modify number of epochs
- Change batch size

Model save path (default):
```java
long ts = System.currentTimeMillis();
String outPath = String.format("artifacts/model-%d.json", ts);
ModelIO.save(nn, outPath);
```
Change `artifacts/` or filename as you prefer. The directory is created automatically.

## Data

The project includes the MNIST dataset files:
- `train-images-idx3-ubyte`: Training images
- `train-labels-idx1-ubyte`: Training labels
- `t10k-images-idx3-ubyte`: Test images
- `t10k-labels-idx1-ubyte`: Test labels

## Building

```bash
# Compile (Maven)
mvn -q compile

# Package
mvn -q package

# Clean
mvn -q clean
```

## Contributing

Feel free to contribute by:
- Adding new activation functions
- Implementing additional optimizers
- Improving the neural network architecture
- Adding more datasets support

## License

This project is open source and available under the MIT License.
