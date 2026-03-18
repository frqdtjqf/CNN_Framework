# Base class for layers in a cnn. All layers are based on this class
# update_params is only for layers with parameters

class Layer:
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update_params(self, learning_rate):
        pass