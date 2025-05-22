import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Paso 1: Cargar datos (SUAR CSV)
df = pd.read_csv("europe.csv")
countries = df['Country']
X = df.drop(columns='Country')
var_names = X.columns

# Paso 2: Graficar cada característica por país (PODRIAMOS USAR POR CARACTERISTICA)
plt.figure(figsize=(18, 10))
for i in range(X.shape[1]):
    plt.subplot(3, int(np.ceil(X.shape[1]/3)), i + 1)
    plt.bar(range(len(countries)), X.iloc[:, i])
    plt.title(var_names[i])
    plt.xticks(ticks=range(len(countries)), labels=countries, rotation=90)
plt.suptitle('Distribución de cada característica por país', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Paso 3: Estandarizar datos (ESTANDARIZAR NO ES LO MISMO QUE NORMALIZAR, ACA HABLAMOS DE UNA NORMALIZACION EN VARIANZA)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Paso 4: Heatmap NO SIRVE PARA NADA PERO SE VE BIEN
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X_std, columns=var_names, index=countries),
            cmap='viridis', cbar=True)
plt.title("Heatmap normalizado (Z-score) por característica")
plt.tight_layout()
plt.show()

# Paso 5: PCA manual (con matriz de correlación)
R = np.corrcoef(X_std.T)
eigvals, eigvecs = np.linalg.eig(R)
idx = eigvals.argsort()[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]
Y = X_std @ eigvecs_sorted

# Paso 6: Varianza explicada
var_exp = eigvals_sorted / eigvals_sorted.sum()
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(var_exp)+1), var_exp*100, '-o')
plt.xlabel('Componente principal')
plt.ylabel('% de varianza explicada')
plt.title('VARIANZA EN TERMINOS DE DIRECCIONES INVARIANTES')
plt.grid(True)
plt.show()

print("Varianza explicada acumulada:")
print(np.cumsum(var_exp))

# Paso 7: Graficar PCA (PC1 vs PC2)
plt.figure(figsize=(10, 7))
plt.scatter(Y[:, 0], Y[:, 1], s=100, color='dodgerblue')
for i, name in enumerate(countries):
    plt.text(Y[i, 0] + 0.1, Y[i, 1], name, fontsize=9)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA de países europeos (primeras 2 componentes)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Paso 8: Mostrar loadings
loadings = pd.DataFrame(eigvecs_sorted,
                        index=var_names,
                        columns=[f'PC{i+1}' for i in range(X.shape[1])])
print("Loadings de las variables (cargas en PCs):")
print(loadings)

# Paso 9: K-means clustering sobre PC1 y PC2 CON USO DE LA LIBRERIA (NO SE PUEDE)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(Y[:, :2])

# Graficar clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=clusters, palette='Set1', s=100)
for i, name in enumerate(countries):
    plt.text(Y[i, 0] + 0.1, Y[i, 1], name, fontsize=9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering de países europeos (K-means sobre PC1-PC2)')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar países por cluster
for k in range(3):
    print(f"\nCluster {k}:")
    print(countries[clusters == k].values)
