import numpy as np


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