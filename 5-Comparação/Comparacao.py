
"""Preparação dos Dados"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import cross_val_score, KFold

# Leitura dos dados
names = ['DateTime', 'Temperatura', 'Umidade', 'Consumo de energia Zona 1']
file = '0-Dataset/Dados_Tratados.data'
data = pd.read_csv(file, names=names)

# Convertendo DateTime para características numéricas
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['ano'] = data['DateTime'].dt.year
data['mes'] = data['DateTime'].dt.month
data['dia'] = data['DateTime'].dt.day
data['hora'] = data['DateTime'].dt.hour
data.drop('DateTime', axis=1, inplace=True)

# Divisão dos dados
X = data.drop('Consumo de energia Zona 1', axis=1)
y = data['Consumo de energia Zona 1']
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)

def construir_modelo():
    modelo = models.Sequential()
    modelo.add(layers.Dense(256, activation='relu', input_shape=(6,)))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(128, activation='relu'))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(64, activation='relu'))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(32, activation='relu'))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(1, activation='linear'))
    modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return modelo

"""Treinamento do Modelo"""

def get_data_by_index(data, indices):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[indices]
    else:
        return data[indices]

import numpy as np
from sklearn.model_selection import KFold

n_splits = 10
kf = KFold(n_splits=n_splits)

resultados = []

for train_index, val_index in kf.split(x_treino):
    x_train_fold = get_data_by_index(x_treino, train_index)
    y_train_fold = get_data_by_index(y_treino, train_index)
    x_val_fold = get_data_by_index(x_treino, val_index)
    y_val_fold = get_data_by_index(y_treino, val_index)

    modelo = construir_modelo()
    modelo.fit(x_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=0)

    score = modelo.evaluate(x_val_fold, y_val_fold, verbose=0)
    resultados.append(score[1])  # Adiciona o MAE ao array de resultados

"""Avaliação do Modelo"""

media_mae = np.mean(resultados)
desvio_padrao_mae = np.std(resultados)

print(f"Média do Erro Absoluto Médio (MAE) obtido com validação cruzada: {media_mae:.2f}")
print(f"Desvio Padrão do MAE: {desvio_padrao_mae:.2f}")

# Salvar o modelo

# Para reutilização futura, salvamos nosso modelo treinado e os escalares de normalização. Isso nos permitirá fazer previsões em novos dados sem ter que retratar o modelo ou recalibrar os escalares.
modelo.save("0-Modelos/modelo_DL_EE.h5")

"""Salvar os escalares"""

import joblib

# Supondo que o escalar se chame 'scaler'
joblib.dump(scaler, '0-Modelos/scaler.pkl')

"""## Processo 5 - Comparação com outros Modelos / Metodos

Para comparar diferentes modelos, é uma prática comum avaliá-los usando métricas específicas. Foi criado uma tabela comparativa simples usando a biblioteca Pandas e então a visualizaremos usando a biblioteca Matplotlib.

Assumiu-se os modelos para comparação como: Linear Regression, Decision Tree, Random Forest, SVM e a sua Deep Neural Network, em métricas como Mean Absolute Error (MAE), Mean Squared Error (MSE) e R-Squared.
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dicionário para armazenar os resultados
resultados_metricas = {
    'Model': [],
    'MAE': [],
    'MSE': [],
    'R-Squared': []
}

# Lista dos modelos a serem avaliados
modelos = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('SVM', SVR()),
    ('Deep Learning', construir_modelo())
]

# Avaliar cada modelo
for nome, modelo in modelos:
    if nome == 'Deep Learning':
        modelo.fit(x_treino, y_treino, epochs=100, batch_size=32, verbose=0)
        predicoes_dl = modelo.predict(x_teste)
        mae = mean_absolute_error(y_teste, predicoes_dl)
        mse = mean_squared_error(y_teste, predicoes_dl)
        r2 = r2_score(y_teste, predicoes_dl)
    else:
        modelo.fit(x_treino, y_treino)
        predicoes = modelo.predict(x_teste)
        mae = mean_absolute_error(y_teste, predicoes)
        mse = mean_squared_error(y_teste, predicoes)
        r2 = r2_score(y_teste, predicoes)

    # Adicionar os resultados ao dicionário
    resultados_metricas['Model'].append(nome)
    resultados_metricas['MAE'].append(mae)
    resultados_metricas['MSE'].append(mse)
    resultados_metricas['R-Squared'].append(r2)

# Convertendo os resultados para um DataFrame
df_resultados = pd.DataFrame(resultados_metricas)

# Visualizando os resultados (usando Matplotlib para criar gráficos)
plt.figure(figsize=(10,6))
plt.barh(df_resultados['Model'], df_resultados['MAE'], color='cyan')
plt.xlabel('Mean Absolute Error (MAE)')
plt.title('Comparação do MAE entre Modelos')
plt.gca().invert_yaxis()
plt.show()