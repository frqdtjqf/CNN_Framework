import numpy as np

class Input:
    def __init__(self, input_shape: tuple[int, int, int], compress: bool = False, norm: bool = False):
        if input_shape is None or len(input_shape) != 3:
            raise ValueError("Input shape must be a tuple of (channels, height, width)")

        # format: (channels, height, width)
        self.input_shape = input_shape

        # lower precision for memory efficiency, can be useful for large datasets or models
        self.compress = compress

        # normalization can help with training stability and convergence, but it is optional
        self.norm = norm

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        input: numpy array of shape (batch_size, channels, height, width) according to the input shape defined in the constructor
        returns: processed input data, potentially compressed and normalized (B,C,H,W)

        """
        # handle single input case (batch_size = 1)
        if input.ndim == 3:
            input = np.expand_dims(input, axis=0)

        if input.shape[1:] != self.input_shape:
            raise ValueError(f"Input shape must be (batch_size, C: {self.input_shape[0]}, H: {self.input_shape[1]}, W: {self.input_shape[2]})")
        
        if self.compress:
            input = self._compress(input)

        if self.norm:
            input = self._normalize(input)
        
        return input
    
    def _compress(self, input: np.ndarray) -> np.ndarray:
        input = input.astype(np.float32)
        return input
    
    def _normalize(self, input: np.ndarray) -> np.ndarray:
        input = input / 255.0
        return input


