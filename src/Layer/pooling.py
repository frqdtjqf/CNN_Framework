from src.Layer.base import Layer

class Pooling(Layer):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass