from src.Layer.base import Layer
from src.Layer.activation_functions import BaseActivation, ReLU

class Activation(Layer):
    def __init__(self, activation: BaseActivation=ReLU()):
        self.activation_function = activation.activation
        self.activation_derivative = activation.derivative

        self.input_cache = None

    def forward(self, input):
        self.input_cache = input
        return self.activation_function(input)

    def backward(self, grad_output):
        return grad_output * self.activation_derivative(self.input_cache)