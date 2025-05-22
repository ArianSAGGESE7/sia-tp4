import numpy as np

class KohonenSOM:
    def __init__(self, input_dim, map_width, map_height, learning_rate=0.5, sigma=1.0, num_epochs=100, init_with_data=False, data=None):
        self.input_dim = input_dim
        self.map_width = map_width
        self.map_height = map_height
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_epochs = num_epochs
        self.weights = np.zeros((map_width, map_height, input_dim))  # inicializamos vacía
        self.locations = np.array([[i, j] for i in range(map_width) for j in range(map_height)])

        if init_with_data and data is not None:
            self._initialize_with_data(data)
        else:
            self._initialize_random()
    
    def _initialize_random(self):
        self.weights = np.random.rand(self.map_width, self.map_height, self.input_dim)

    def _initialize_with_data(self, data):
        assert data.shape[1] == self.input_dim, "Dimensión de entrada incorrecta"
        indices = np.random.choice(data.shape[0], size=self.map_width * self.map_height, replace=True)
        selected = data[indices]
        self.weights = selected.reshape((self.map_width, self.map_height, self.input_dim))


    def _find_bmu(self, x):
        """Encuentra la neurona ganadora (BMU)"""
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.map_width, self.map_height))
        return bmu_idx

    def _neighborhood(self, bmu, neuron, sigma):
        """Función de vecindad Gaussiana"""
        return np.exp(-np.sum((np.array(bmu) - np.array(neuron)) ** 2) / (2 * sigma ** 2))

    def _update_weights(self, x, bmu, epoch):
        """Actualiza los pesos del BMU y vecinos"""
        alpha = self.learning_rate * (1 - epoch / self.num_epochs)
        sigma = self.sigma * (1 - epoch / self.num_epochs)
        for i in range(self.map_width):
            for j in range(self.map_height):
                neuron = (i, j)
                influence = self._neighborhood(bmu, neuron, sigma)
                self.weights[i, j] += alpha * influence * (x - self.weights[i, j])

    def train(self, X):
        """Entrena la red con los datos"""
        for epoch in range(self.num_epochs):
            np.random.shuffle(X)
            for x in X:
                bmu = self._find_bmu(x)
                self._update_weights(x, bmu, epoch)

    def map_input(self, x):
        """Devuelve la posición en el mapa del BMU para una entrada"""
        return self._find_bmu(x)
