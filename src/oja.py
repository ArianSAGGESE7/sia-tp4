import numpy as np

class Oja:
    def __init__(self, input_size, n_components=2, learning_rate=0.01):
        self.input_size = input_size
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.weights = np.random.randn(n_components, input_size)
        self.weights = self._normalize_rows(self.weights)

    def _normalize_rows(self, W):
        return W / np.linalg.norm(W, axis=1, keepdims=True)

    def train(self, X, epochs=50):
        for _ in range(epochs):
            for x in X:
                x = x.reshape(-1, 1)  # columna
                y = self.weights @ x
                dw = self.learning_rate * (y @ x.T - (y @ y.T) @ self.weights)
                self.weights += dw
            self.weights = self._normalize_rows(self.weights)

    def get_components(self):
        return self.weights
