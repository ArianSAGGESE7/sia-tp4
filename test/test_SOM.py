from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from KohonemSOM import KohonenSOM
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from UMatrixVisualizer import UMatrixVisualizer
# Cargar y escalar datos
# Cargar datos
df = pd.read_csv("europe.csv")
countries = df["Country"]
X = df.drop(columns='Country').values
X = StandardScaler().fit_transform(X)

# Crear y entrenar SOM
# Crear y entrenar la red con inicialización basada en datos

# som = KohonenSOM(input_dim=X.shape[1],
#                   map_width=4,
#                   map_height=4,
#                   num_epochs=300)  # SIN init_with_data


som = KohonenSOM(input_dim=X.shape[1],
                  map_width=4,
                  map_height=4,
                  num_epochs=300,
                  init_with_data=True,
                  data=X)

som.train(X)



# Mapear una entrada
print("Posición en el mapa de la primera entrada:", som.map_input(X[0]))


## Interpretacion gráfica 

# Visualizar U-Matrix
uvis = UMatrixVisualizer(som)
uvis.plot()  # O 
# Simulamos etiquetas binarias para el ejemplo (reemplazá esto con tus verdaderos labels)
# Ejemplo: labels = ['1', '0', '1', ...]
# labels = np.random.choice(['0', '1'], size=X.shape[0])  

# # Inicializar contadores por celda
# map_shape = (som.map_width, som.map_height)
# counts = np.zeros(map_shape)
# label_counts = defaultdict(lambda: np.zeros(map_shape, dtype=int))

# # Recorremos cada muestra y actualizamos los contadores
# for i in range(len(X)):
#     bmu = som.map_input(X[i])
#     label = labels[i]
#     counts[bmu] += 1
#     label_counts[label][bmu] += 1

# # Crear texto a mostrar en cada celda (ejemplo: '1:72\n0:56')
# annotations = np.empty(map_shape, dtype=object)
# for i in range(som.map_width):
#     for j in range(som.map_height):
#         text = "\n".join([f"{label}:{label_counts[label][i, j]}" for label in sorted(label_counts)])
#         annotations[i, j] = text

# # Plot del heatmap
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(counts.T, annot=annotations.T, fmt='', cmap='magma', cbar_kws={'label': 'Entries amount'},
#                  linewidths=0.5, linecolor='gray', square=True)
# plt.title("Final entries per neuron (SOM)")
# plt.xlabel("Neuron X")
# plt.ylabel("Neuron Y")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

