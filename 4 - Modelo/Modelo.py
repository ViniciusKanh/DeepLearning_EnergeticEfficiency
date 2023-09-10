import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Carregar os dados
names = ['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption'] 
features =['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption'] 
output_file = '0-Dataset/Dados_Tratados.data'
input_file = '0-Dataset/Dados_Brutos.data'
data = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names,      # Nome das colunas 
                    usecols = features, # Define as colunas que serão  utilizadas
                    na_values='?')      # Define que ? será considerado valores ausentes
# Processamento da coluna 'DateTime'
data['ano'] = pd.to_datetime(data['DateTime']).dt.year
data['mes'] = pd.to_datetime(data['DateTime']).dt.month
data['dia'] = pd.to_datetime(data['DateTime']).dt.day
data['hora'] = pd.to_datetime(data['DateTime']).dt.hour
data['minuto'] = pd.to_datetime(data['DateTime']).dt.minute
data = data.drop(columns=['DateTime'])  # Descarte a coluna original de data e hora

# 2. Separação dos Dados
features = data.drop(['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption'], axis=1)
labels = data[['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 3. Normalização
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 4. Construção do Modelo
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # A saída possui 3 neurônios, um para cada zona de consumo.
])

# Salvar modelo
model.save("0-Dataset/my_model.h5")

# Salvar scalers
import joblib
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Treinamento
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.1, batch_size=32)
