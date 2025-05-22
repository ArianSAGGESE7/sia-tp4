import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt
from Hopfield import HopfieldNetwork

# --- Funciones auxiliares ---
def show_pattern(vec, title=""):
    plt.imshow(vec.reshape(5,5), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def add_noise(pattern, noise_level=0.3):
    noisy = pattern.copy()
    n = len(noisy)
    flip_indices = np.random.choice(n, size=int(n * noise_level), replace=False)
    noisy[flip_indices] *= -1
    return noisy

# --- Letras codificadas ---
J = np.array([
      1,  1,  1,  1,  1,
     -1, -1,  -1, 1, -1,
     -1, -1,  -1, 1, -1,
      1,  -1,  -1, 1, -1,
      1, 1,  1, -1, -1
])

L = np.array([
      1, -1, -1, -1, -1,
      1, -1, -1, -1, -1,
      1, -1, -1, -1, -1,
      1, -1, -1, -1, -1,
      1,  1,  1,  1,  1
])

C = np.array([
      1,  1,  1,  1,  1,
      1, -1, -1, -1, -1,
      1, -1, -1, -1, -1,
      1, -1, -1, -1, -1,
      1,  1,  1,  1,  1
])

T = np.array([
      1,  1,  1,  1,  1,
     -1,  1, -1,  1, -1,
     -1,  1, -1,  1, -1,
     -1,  1, -1,  1, -1,
     -1,  1, -1,  1, -1
])

# --- Entrenamiento ---
patterns = np.stack([J, L, C, T])
net = HopfieldNetwork()
net.train(patterns)

# --- Prueba con ruido ---
test_index = 0  # Probamos con J
original = patterns[test_index]
noisy = add_noise(original, noise_level=0.3)

show_pattern(original, "Original")
show_pattern(noisy, "Noisy Input")

recovered, steps = net.recall(noisy, steps=10)

# Mostrar evolución
for i, s in enumerate(steps):
    show_pattern(s, f"Step {i}")

# --- Parte (b): patrón MUY ruidoso ---
very_noisy = add_noise(original, noise_level=0.6)
show_pattern(very_noisy, "Very Noisy Input")

recovered2, steps2 = net.recall(very_noisy, steps=10)

# Mostrar evolución
for i, s in enumerate(steps2):
    show_pattern(s, f"Step {i} (Possibly Spurious)")
