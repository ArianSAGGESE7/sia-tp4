import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt
from OjaNetwork import OjaNetwork 

# Cargar y normalizar datos
df = pd.read_csv("europe.csv")
X = df.drop(columns='Country').values
X_std = StandardScaler().fit_transform(X)

# Oja con tasa constante
oja_const = OjaNetwork(input_dim=X.shape[1], learning_rate=0.01, epochs=500, adaptive=False)
oja_const.fit(X_std)
proj_const = oja_const.transform(X_std)

# Oja con tasa adaptativa (1/t)
oja_adapt = OjaNetwork(input_dim=X.shape[1], epochs=500, adaptive=True)
oja_adapt.fit(X_std)
proj_adapt = oja_adapt.transform(X_std)


weights = oja_adapt.get_weights()
variables = df.columns[1:]  # excluye "Country"

print("Componente principal aprendida (vector de pesos):\n")
for var, w in zip(variables, weights):
    print(f"{var:<15}: {w:.4f}")

projection = oja_adapt.transform(X_std)
ranking_df = pd.DataFrame({
    'Country': df['Country'],
    'Projection': projection
})

ranking_df = ranking_df.sort_values(by='Projection', ascending=False).reset_index(drop=True)

print("\n📊 Ranking de países por su proyección sobre la primera componente:\n")
print(ranking_df)


projection_df = ranking_df.sort_values(by="Projection", ascending=True)

plt.figure(figsize=(12, 8))
bars = plt.barh(projection_df["Country"], projection_df["Projection"],
                color=plt.cm.coolwarm((projection_df["Projection"] - projection_df["Projection"].min()) /
                                      (projection_df["Projection"].max() - projection_df["Projection"].min())))

plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Proyección sobre la primera componente (Oja)")
plt.title("Ranking de países según su posición sobre la 1ra componente principal")
plt.tight_layout()
plt.grid(axis='x', linestyle=':', alpha=0.5)
plt.show()
# # Graficar ambas proyecciones
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(proj_const, np.zeros_like(proj_const), color='cornflowerblue')
# for i, name in enumerate(df["Country"]):
#     plt.text(proj_const[i], 0.05, name, fontsize=8, ha='center')
# plt.title("Oja - tasa constante")
# plt.yticks([])

# plt.subplot(1, 2, 2)
# plt.scatter(proj_adapt, np.zeros_like(proj_adapt), color='salmon')
# for i, name in enumerate(df["Country"]):
#     plt.text(proj_adapt[i], 0.05, name, fontsize=8, ha='center')
# plt.title("Oja - tasa adaptativa (1/t)")
# plt.yticks([])

# plt.suptitle("Proyección sobre la 1ra componente principal (Regla de Oja)")
# plt.tight_layout()
# plt.show()