from src.Layer.base import Layer
import numpy as np

class SoftmaxCrossEntropyLoss(Layer):
    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # logits: (batch, classes)
        # labels: (batch, classes) one-hot
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.labels = labels
        loss = -np.sum(labels * np.log(self.probs + 1e-12)) / labels.shape[0]
        return loss

    def backward(self) -> np.ndarray:
        # Grad w.r.t logits
        return (self.probs - self.labels) / self.labels.shape[0]