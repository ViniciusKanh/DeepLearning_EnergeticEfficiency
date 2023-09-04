# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Carregando os dados
names = ['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption'] 
features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption']
input_file = '0-Dataset/Dados_Tratados.data'

data = pd.read_csv(input_file, names=names, usecols=features)

# Definindo características (X) e rótulos (y)
X = data[features].values
y = data[['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']].values

# Separando os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construindo o modelo de rede neural
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3))  # 3 neurônios na camada de saída, correspondendo às 3 zonas

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Avaliando o desempenho do modelo
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# TODO: Otimização baseada nos resultados obtidos
# Exemplo: Adicionando mais camadas, ajustando parâmetros, implementando regularização, etc.
