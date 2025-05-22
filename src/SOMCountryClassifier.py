import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from KohonemSOM import KohonenSOM

class SOMCountryClassifier:
    def __init__(self, df, selected_vars, map_size=(4, 4), num_epochs=300):
        self.df = df.copy()
        self.countries = self.df['Country']
        self.selected_vars = selected_vars
        self.X = self.df[selected_vars].values
        self.X_std = StandardScaler().fit_transform(self.X)
        self.map_size = map_size  

        self.som = KohonenSOM(input_dim=len(selected_vars),
                              map_width=map_size[0],
                              map_height=map_size[1],
                              num_epochs=num_epochs)
        self.som.train(self.X_std)

    def map_countries(self):
        self.bmu_map = defaultdict(list)
        for i, x in enumerate(self.X_std):
            bmu = self.som.map_input(x)
            self.bmu_map[bmu].append(self.countries.iloc[i])
        return self.bmu_map

    def plot_country_distribution(self):
        counts = np.zeros(self.map_size)
        annotations = np.empty(self.map_size, dtype=object)

        # Llenar los textos por celda
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                paises = self.bmu_map.get((i, j), [])
                counts[i, j] = len(paises)
                annotations[i, j] = "\n".join(paises) if paises else ""

        # Graficar
        plt.figure(figsize=(10, 8))
        sns.heatmap(counts.T, annot=annotations.T, fmt='', cmap='coolwarm',
                    cbar_kws={'label': 'Cantidad de países'}, linewidths=0.5,
                    linecolor='gray', square=True)
        plt.title(f'Asociación de países por características: {", ".join(self.selected_vars)}')
        plt.xlabel("Neurona X")
        plt.ylabel("Neurona Y")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()