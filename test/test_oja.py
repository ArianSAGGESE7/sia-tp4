import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.oja import Oja

def test_00():
    np.random.seed(42)
    n_samples = 500
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]  # matriz de covarianza no diagonal
    X = np.random.multivariate_normal(mean, cov, n_samples)

    oja = Oja(input_size=2, n_components=1)
    oja.train(X, epochs=100)

    w_oja = oja.weights[0]
    w_oja /= norm(w_oja)

    # Comparamos con PCA
    pca = PCA(n_components=1)
    pca.fit(X)
    pc1 = pca.components_[0]

    # Alineamos signos para comparar correctamente
    if np.dot(w_oja, pc1) < 0:
        pc1 = -pc1

    print("Vector aprendido (Oja):", w_oja)
    print("Primera componente principal (PCA):", pc1)
    print("Ángulo entre vectores (grados):", np.degrees(np.arccos(np.clip(np.dot(w_oja, pc1), -1.0, 1.0))))

    # Visualizamos los vectores
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    origin = np.mean(X, axis=0)
    plt.quiver(*origin, *w_oja, color='r', scale=3, label='Oja')
    plt.quiver(*origin, *pc1, color='g', scale=3, label='PCA')
    plt.axis('equal')
    plt.legend()
    plt.title("Comparación de vector aprendido (Oja) vs PCA")
    plt.show()
