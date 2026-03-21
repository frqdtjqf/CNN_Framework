from src.Layer.base import Layer
import numpy as np

class Pooling(Layer):
    def __init__(self, pool_size: int = 2, pool_type: str = 'max'):
        self.pool_size = pool_size
        self.pool_type = pool_type

        self.input_cache = None
        self.max_indices_cache = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input_cache = input
        batch_size, channels, height, width = input.shape

        out_height = height // self.pool_size
        out_width = width // self.pool_size

        output = np.zeros((batch_size, channels, out_height, out_width))

        
        if self.pool_type == 'max':
            self.max_indices_cache = np.zeros((batch_size, channels, out_height, out_width), dtype=int)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size

                if self.pool_type == 'max':
                    pool_region = input[:, :, h_start:h_end, w_start:w_end]
                    output[:, :, i, j] = np.max(pool_region, axis=(2, 3))
                    if self.max_indices_cache is not None:
                        self.max_indices_cache[:, :, i, j] = np.argmax(pool_region.reshape(batch_size, channels, -1), axis=2)
                elif self.pool_type == 'average':
                    output[:, :, i, j] = np.mean(input[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input_cache)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size

                if self.pool_type == 'max':
                    for b in range(batch_size):
                        for c in range(channels):
                            max_index = self.max_indices_cache[b, c, i, j]
                            max_h = h_start + max_index // self.pool_size
                            max_w = w_start + max_index % self.pool_size
                            grad_input[b, c, max_h, max_w] += grad_output[b, c, i, j]
                elif self.pool_type == 'average':
                    grad_input[:, :, h_start:h_end, w_start:w_end] += grad_output[:, :, i:i+1, j:j+1] / (self.pool_size * self.pool_size)

        return grad_input