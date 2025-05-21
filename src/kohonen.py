import numpy as np
from time import time


class Kohonen:
    
      
    def __init__(self, dims_in, dims_out, alpha_0=0.1):
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.m, self.n = dims_out

        self.w = np.random.rand(self.m, self.n, self.dims_in)
        
        self.sigma_0 = max(dims_out) // 2
        self.sigma = self.sigma_0

        self.alpha_0 = alpha_0
        self.alpha = alpha_0


    def train(self, data, epochs=1, decay=True, shuffle=True):
        data = np.array(data)
        time_const = epochs / np.log(self.sigma_0 + 1e-8)
        
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(data)

            for x in data:
                index = self.get_bmu(x)
                influence = self.get_influence(index)
                x = x.reshape(1, 1, self.dims_in)
                influence = influence[..., np.newaxis]
                self.w += self.alpha * influence * (x - self.w)

            if decay:
                self.sigma = self.sigma_0 * np.exp(-epoch / time_const)
                self.alpha = self.alpha_0 * np.exp(-epoch / time_const)
                

    def get_bmu(self, x):
        "devuelve la coordenada (i, j) de la BMU"
        dist = ((x - self.w)**2).sum(axis=2)
        return np.unravel_index(np.argmin(dist), (self.m, self.n))


    def get_influence(self, bmu_idx):
        "devuelve la influencia del vecindario"
        grid_x, grid_y = np.meshgrid(np.arange(self.m), np.arange(self.n), indexing='ij')
        dist_sq = (grid_x - bmu_idx[0])**2 + (grid_y - bmu_idx[1])**2
        return np.exp(-dist_sq / (2 * (self.sigma ** 2)))


    def predict(self, data):
        return [self.get_bmu(x) for x in data]
