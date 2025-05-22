import numpy as np
import matplotlib.pyplot as plt

class UMatrixVisualizer:
    def __init__(self, som):
        """
        som: instancia entrenada de KohonenSOM
        """
        self.som = som
        self.u_matrix = self._compute_u_matrix()

    def _compute_u_matrix(self):
        """
        Calcula la U-Matrix del SOM: distancia promedio a vecinos de cada neurona.
        """
        w, h = self.som.map_width, self.som.map_height
        umat = np.zeros((w, h))

        for i in range(w):
            for j in range(h):
                center = self.som.weights[i, j]
                neighbors = []
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # vecinos cardinales
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < w and 0 <= nj < h:
                        neighbors.append(self.som.weights[ni, nj])
                if neighbors:
                    dists = [np.linalg.norm(center - n) for n in neighbors]
                    umat[i, j] = np.mean(dists)
        return umat

    def plot(self, save_path=None):
        """
        Muestra la U-Matrix como círculos sombreados.
        Si save_path se especifica, guarda el gráfico como PNG.
        """
        w, h = self.u_matrix.shape
        fig, ax = plt.subplots(figsize=(8, 8))

        # Normalizar a [0, 1] para usar en tamaño y color
        norm = (self.u_matrix - np.min(self.u_matrix)) / (np.max(self.u_matrix) - np.min(self.u_matrix) + 1e-10)

        for i in range(w):
            for j in range(h):
                size = norm[i, j]
                color = str(1 - size)  # más distancia → más claro
                circle = plt.Circle((i, j), radius=0.4 * size + 0.05, color=color, ec='k')
                ax.add_patch(circle)

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Matriz U (U-Matrix) - distancias entre neuronas")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()
