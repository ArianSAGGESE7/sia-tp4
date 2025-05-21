import numpy as np
import pandas as pd
from src.kohonen import Kohonen

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def test_00():
    df = pd.read_csv("data/europe.csv")
    df = df.set_index("Country")
    df = (df - df.mean()) / df.std(ddof=0) 

    som = Kohonen(df.shape[1], dims_out=(5, 5))
    som.train(df.values, epochs=100)

    predictions = som.predict(df.values)

    distance_to_first = [distance(p, predictions[0]) for p in predictions]
    indices = np.argsort(distance_to_first)

    assert len(predictions) == len(df)


def test_01():
    x1, _ = make_circles(n_samples=100, noise=0.1, factor=0)
    x1 += np.array([-2.0, 0.0])

    x2, _ = make_circles(n_samples=100, noise=0.1, factor=0)
    x2 += np.array([2.0, 0.0])

    x3, _ = make_circles(n_samples=100, noise=0.1, factor=0)
    x3 += np.array([0.0, 2.0])

    x4, _ = make_circles(n_samples=100, noise=0.1, factor=0)
    x4 += np.array([0.0, -2.0])

    data = np.vstack([x1, x2, x3, x4])

    som = Kohonen(dims_in=2, dims_out=(2, 2))
    som.train(data, epochs=200, decay=False)
    bmus = som.predict(data)
    
    colors = np.array([i * som.n + j for i, j in bmus])
    colors_norm = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=colors_norm, cmap='tab20', s=30)
    plt.title("Datos coloreados por BMU asignado")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
