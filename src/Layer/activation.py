from src.Layer.base import Layer

class Activation(Layer):
    def __init__(self, activation_function: callable):
        self.activation_function = activation_function

    def forward(self, input):
        return self.activation_function(input)

    def backward(self, grad_output):
        pass