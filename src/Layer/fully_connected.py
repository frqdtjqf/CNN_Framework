from src.Layer.base import Layer
import numpy as np

class FullyConnected(Layer):
    def __init__(self, num_input_neurons: int, num_neurons: int, weights: list = None, biases: list = None):
        self.num_input_neurons = num_input_neurons
        self.num_neurons = num_neurons
        if weights is None:
            self.weights = np.random.randn(num_neurons, num_input_neurons) * 0.01
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.zeros(num_neurons)
        else:
            self.biases = biases
    def forward(self, input: np.ndarray) -> np.ndarray:
        # Input speichern für Backward
        self.input_cache = input

        # Numpy Arrays für saubere Berechnungen
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

        # Forward Pass
        output = input @ self.weights.T + self.biases  # Shape: (batch, num_neurons)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Grad w.r.t input
        grad_input = grad_output @ self.weights

        # Grad w.r.t weights
        self.grad_weights = grad_output.T @ self.input_cache

        # Grad w.r.t biases
        self.grad_biases = np.sum(grad_output, axis=0)

        return grad_input
    
    def update_params(self, learning_rate: float):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

fc = FullyConnected(num_input_neurons=4, num_neurons=3)

x = np.random.randn(2,4)  # batch_size=2
out = fc.forward(x)

grad = np.ones_like(out)
dx = fc.backward(grad)

print("Input:\n", x)
print("Output:\n", out)
print("Grad Input:\n", dx)