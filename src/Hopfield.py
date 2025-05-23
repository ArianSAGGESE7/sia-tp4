import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self):
        self.weights = None
        self.num_neurons = 0

    def train(self, patterns):
        num_patterns, self.num_neurons = patterns.shape
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.num_neurons

    def recall(self, pattern, steps=10, verbose=False):
        x = pattern.copy()
        history = [x.copy()]
        for _ in range(steps):
            x = np.sign(self.weights @ x)
            x[x == 0] = 1
            history.append(x.copy())
            if verbose:
                print(x.reshape(5, 5))
        return x, history

# Función para mostrar un patrón
def show_pattern(pattern, title=""):
    plt.imshow(pattern.reshape(5, 5), cmap='gray_r')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para agregar ruido
def add_noise(pattern, noise_level=0.3):
    noisy = pattern.copy()
    n = len(pattern)
    num_flips = int(noise_level * n)
    flip_indices = np.random.choice(n, num_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy

# Verifica si dos patrones son iguales
def is_equal(p1, p2):
    return np.array_equal(p1, p2)
