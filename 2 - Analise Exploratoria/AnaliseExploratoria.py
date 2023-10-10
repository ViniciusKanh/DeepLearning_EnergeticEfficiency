
"""
Bibliotecas para a Analise Exploratoria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names = ['DateTime','Temperatura','Umidade','Consumo de energia Zona 1']
features =['DateTime','Temperatura','Umidade','Consumo de energia Zona 1']
file = '0-Dataset/Dados_Tratados.data'
df = pd.read_csv(file,         # Nome do arquivo com dados
                    names = names,      # Nome das colunas
                    usecols = features)   # Define as colunas que serão  utilizadas

"""Visualizando as primeiras linhas do dataset"""

print(df.head())

"""## Histogramas para entender a distribuição dos dados"""

# Definindo tamanho da figura que vai conter todos os histogramas
plt.figure(figsize=(15, 15))

# Iterando por cada feature e plotando o histograma correspondente em um subplot
for idx, feature in enumerate(features[1:], 1):
    plt.subplot(len(features[1:]), 1, idx)  # Número total de linhas é len(features[1:]), e 1 coluna. idx é a posição atual.
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histograma para {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequência')
    plt.grid(True)

# Ajusta layout para evitar sobreposição
plt.tight_layout()

# Mostra a figura completa
plt.show()

"""
Mapa de calor para visualizar as correlações
"""

correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()

features_to_compare = ['Temperatura', 'Umidade']
zones = ['Umidade','Consumo de energia Zona 1']

plt.figure(figsize=(18, len(zones) * 6))

for idx_zone, zone in enumerate(zones, 1):
    for idx_feature, feature in enumerate(features_to_compare, 1):
        plt.subplot(len(zones), len(features_to_compare), (idx_zone - 1) * len(features_to_compare) + idx_feature)
        sns.scatterplot(data=df, x=feature, y=zone)
        plt.title(f'Gráfico de Dispersão entre {feature} e {zone}')
        plt.grid(True)

# Salva a figura combinada no Google Drive (opcional)
plt.savefig('2 - Analise Exploratoria/Imagens/scatter_plots_all_zones.png')

plt.tight_layout()
plt.show()