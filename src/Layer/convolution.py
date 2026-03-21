from src.Layer.base import Layer
import numpy as np

class Convolution(Layer):
    def __init__(self, filter_size: int, num_filters: int, in_channels: int, stride: int=1, padding: int=0):

        # initialize the layer with given parameters
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.filters = None
        self.biases = None

        self.grad_filters = None
        self.grad_biases = None

        self.input_padded_cache = None

        self._initialize_filters_biases()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        input: numpy array of shape (batch_size, in_channels, height, width) - the input feature maps to be convolved with the filters
        returns: output of the convolution layer, shape (batch_size, num_filters, out_height, out_width), for each filter computed feature maps
        """

        batch_size, _, in_height, in_width = input.shape

        input_padded = self._apply_zero_padding(input)
        output = self._get_output_tensor(in_height, in_width, batch_size)

        self.input_padded_cache = input_padded #store padded input for backward pass

        output = self._convolve(input_padded, output)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: numpy array of shape (batch_size, num_filters, out_height, out_width) - the gradient of the loss with respect to the output of the convolution layer
        returns: the gradient of the loss with respect to the input of the convolution layer, shape (batch_size, in_channels, height, width)
        computes the gradients with respect to the filters, biases, and input using the chain rule of calculus and stores them for parameter updates
        """
        grad_input_padded = np.zeros_like(self.input_padded_cache)

        self.grad_filters = np.zeros_like(self.filters)
        self.grad_biases = np.zeros_like(self.biases)

        grad_input = self._compute_gradients(grad_output, grad_input_padded)

        return grad_input

    def update_params(self, learning_rate: float):
        """
        learning_rate: the learning rate for updating the weights and biases
        updates the weights and biases using the computed gradients from the backward pass"""
        self.filters -= learning_rate * self.grad_filters
        self.biases -= learning_rate * self.grad_biases

    def _remove_padding(self, padded: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            return padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return padded

    def _initialize_filters_biases(self):
        self.filters = np.random.randn(
            self.num_filters, self.in_channels, self.filter_size, self.filter_size
        )
        self.biases = np.zeros(self.num_filters)

    def _apply_zero_padding(self, input: np.ndarray) -> np.ndarray:
        """
        input: numpy array of shape (batch_size, in_channels, height, width) - the input feature maps to be convolved with the filters
        returns: input with zero padding applied, shape (batch_size, in_channels, height + 2*padding, width + 2*padding)
        """
        if self.padding > 0:
            return np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        return input
    
    def _get_output_tensor(self, in_height: int, in_width: int, batch_size: int) -> np.ndarray:
        """
        in_height: height of the input feature maps
        in_width: width of the input feature maps
        batch_size: number of samples in the batch
        returns: output tensor initialized to zeros, shape (batch_size, num_filters, out_height, out_width)
        """
        out_h = (in_height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (in_width - self.filter_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))
        return output
    
    def _convolve(self, input: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        input: numpy array of shape (batch_size, in_channels, height, width) - the input feature maps to be convolved with the filters
        output: numpy array of shape (batch_size, num_filters, out_height, out_width) - the output tensor
        returns: the convolved output tensor
        """

        batch_size, _, out_h, out_w = output.shape

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size

                    # slice input over all channels for current window and batch element
                    # then convolve with each filter and add bias
                    input_slice = input[b, :, h_start:h_end, w_start:w_end]
                    for f in range(self.num_filters):
                        output[b, f, i, j] = np.sum(input_slice * self.filters[f]) + self.biases[f]

        return output
    
    def _compute_gradients(self, grad_output: np.ndarray, grad_input_padded: np.ndarray) -> np.ndarray:
        """
        grad_output: numpy array of shape (batch_size, num_filters, out_height, out_width) - the gradient of the loss with respect to the output of the convolution layer
        grad_input_padded: numpy array of shape (batch_size, in_channels, height + 2*padding, width + 2*padding) - the gradient of the loss with respect to the padded input
        returns: the gradient of the loss with respect to the input of the convolution layer, shape (batch_size, in_channels, height, width)
        computes the gradients with respect to the filters, biases, and input using the chain rule of calculus and stores them for parameter updates
        """

        batch_size, _, out_h, out_w = grad_output.shape

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size

                    # compute slice of input corresponding to the current window and batch element
                    input_slice = self.input_padded_cache[b, :, h_start:h_end, w_start:w_end]

                    # loop over each filter to compute gradients
                    for f in range(self.num_filters):

                        # get the gradient at the current position for the current filter
                        grad_output_value = grad_output[b, f, i, j]

                        # accumulate gradients for filters and biases
                        self.grad_filters[f] += input_slice * grad_output_value
                        self.grad_biases[f] += grad_output_value

                        # accumulate gradient for input
                        grad_input_padded[b, :, h_start:h_end, w_start:w_end] += self.filters[f] * grad_output_value

        grad_input = self._remove_padding(grad_input_padded)

        return grad_input