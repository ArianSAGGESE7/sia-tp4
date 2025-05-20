import numpy as np
from time import time


class Kohonen:
    alpha  = 0.1    # Learning rate
      
    def __init__(self, dims_in, dims_out):
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.m, self.n = dims_out

        self.w = np.random.rand(self.m, self.n, self.dims_in)
        self.sigma = max(dims_out) // 2
        

    def train(self, data, epochs=1, decay=True):
        _lambda = epochs / np.log(self.sigma + 1e-6) # Constante de tiempo
        np.random.shuffle(data)

        for epoch in range(epochs):
            for x in data:
                index = self.get_bmu(x)
                influence = self.get_influence(index, self.sigma)
                
                x = x.reshape(1, 1, self.dims_in)
                influence = influence.reshape(self.m, self.n, 1)

                self.w += self.alpha * influence * (x - self.w)

            if decay:
                self.sigma = self.sigma * np.exp(-epoch / _lambda)
                self.alpha = self.alpha * np.exp(-epoch / _lambda)


    def get_bmu(self, x):
        "devuelve la coordenada (i, j) de la BMU"
        d = np.linalg.norm(x - self.w, axis=2)
        return np.unravel_index(np.argmin(d), (self.m, self.n))


    def get_influence(self, idx, sigma):
        "devuelve la influencia del vecindario"
        ax = np.arange(self.m).reshape(self.m, 1)
        ay = np.arange(self.n).reshape(1, self.n)
        dist_to_bmu = (ax - idx[0]) ** 2 + (ay - idx[1]) ** 2
        return np.exp(-dist_to_bmu / (2 * sigma ** 2))


    def predict(self, data):
        predictions = []
        for input_vector in data:
            predictions.append(self.get_bmu(input_vector))
        return predictions
