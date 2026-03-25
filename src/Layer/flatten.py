from src.Layer.base import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        self.input_cache = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input_cache = input
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.input_cache.shape)