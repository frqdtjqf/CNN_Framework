from src.Layer.base import Layer

class Convolution(Layer):
    def __init__(self, filter_size: int, num_filters: int, stride: int=1, padding: int=0):

        # initialize the layer with given parameters
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update_params(self, learning_rate):
        pass