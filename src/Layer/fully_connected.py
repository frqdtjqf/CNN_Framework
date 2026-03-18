from src.Layer.base import Layer
import numpy as np

class FullyConnected(Layer):
    def __init__(self, num_input_neurons: int, num_neurons: int, weights: list = None, biases: list = None):
        self.num_input_neurons = num_input_neurons
        self.num_neurons = num_neurons
        if weights is None:
            self.weights = [[0.0 for _ in range(num_input_neurons)] for _ in range(num_neurons)]
        else:
            self.weights = weights
        if biases is None:
            self.biases = [0.0 for _ in range(num_neurons)]
        else:
            self.biases = biases

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass