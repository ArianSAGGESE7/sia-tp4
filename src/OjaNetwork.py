import numpy as np

class OjaNetwork:
    def __init__(self, input_dim, learning_rate=0.01, epochs=100, adaptive=False):
        """
        input_dim: dimensión del vector de entrada
        learning_rate: eta inicial (si adaptive=False, se mantiene constante)
        epochs: número de pasadas sobre los datos
        adaptive: si True, eta = 1 / t
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.adaptive = adaptive
        self.weights = np.random.randn(input_dim)  # inicialización normal
        self.history = []

    def fit(self, X):
        t = 1
        for epoch in range(self.epochs):
            for x in X:
                y = np.dot(self.weights, x)
                eta = 1 / t if self.adaptive else self.learning_rate
                self.weights += eta * y * (x - y * self.weights)
                self.history.append(self.weights.copy())  # opcional: guardar trayectoria
                t += 1
                self.weights += eta * y * (x - y * self.weights)

                # Normalizar pesos para mantenerlos estables
                norm = np.linalg.norm(self.weights)
                if norm > 0:
                    self.weights /= norm

        return self.weights

    def transform(self, X):
        return np.dot(X, self.weights)

    def get_weights(self):
        return self.weights
