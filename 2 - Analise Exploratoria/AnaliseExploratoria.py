# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definindo o caminho do arquivo
names = ['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption'] 
features =['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption'] 
input_file = '0-Dataset/Dados_Tratados.data'
df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names,      # Nome das colunas 
                    usecols = features, # Define as colunas que serão  utilizadas
                    na_values='?')  

# Visualizando as primeiras linhas do dataset
print(df.head())

# Histogramas para entender a distribuição dos dados
for feature in names:
    plt.figure(figsize=(12,6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histograma para {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Timestamp'] = df['DateTime'].astype('int64')
df = df.drop(columns='DateTime')
# Matriz de correlação
correlation_matrix = df.corr()

# Mapa de calor para visualizar as correlações
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()

# Gráficos de dispersão para verificar a relação entre as características
features_to_compare = ['Temperature', 'Humidity', 'Wind Speed']
for feature in features_to_compare:
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, x=feature, y='Zone 1 Power Consumption')
    plt.title(f'Gráfico de Dispersão entre {feature} e Zone 1 Power Consumption')
    plt.grid(True)
    plt.show()
    
features_to_compare = ['Temperature', 'Humidity', 'Wind Speed']
for feature in features_to_compare:
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, x=feature, y='Zone 2  Power Consumption')
    plt.title(f'Gráfico de Dispersão entre {feature} e Zone 2 Power Consumption')
    plt.grid(True)
    plt.show()
    
features_to_compare = ['Temperature', 'Humidity', 'Wind Speed']
for feature in features_to_compare:
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, x=feature, y='Zone 3  Power Consumption')
    plt.title(f'Gráfico de Dispersão entre {feature} e Zone 3 Power Consumption')
    plt.grid(True)
    plt.show()

