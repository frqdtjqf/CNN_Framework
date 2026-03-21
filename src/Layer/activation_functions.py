import numpy as np

class BaseActivation:
    def activation(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation function must be implemented in subclass")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Derivative function must be implemented in subclass")

class ReLU(BaseActivation):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)