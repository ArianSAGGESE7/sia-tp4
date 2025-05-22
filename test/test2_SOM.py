from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from KohonemSOM import KohonenSOM
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from SOMCountryClassifier import SOMCountryClassifier
# Cargar datos
df = pd.read_csv("europe.csv")

# Seleccionar variables a analizar
variables = ["GDP", "Inflation", "Life.expect", "Military", "Unemployment"]

# Crear y entrenar clasificador
classifier = SOMCountryClassifier(df, selected_vars=variables, map_size=(4, 4), num_epochs=300)

# Mapear países
classifier.map_countries()

# Visualizar resultados
classifier.plot_country_distribution()
