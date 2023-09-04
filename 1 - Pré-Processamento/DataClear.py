import pandas as pd
import numpy as np
from missingno import matrix
from pandas_profiling import ProfileReport


def UpdateMissingValues(df, column, method="median", number=0):
    if method == 'number':
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        median = round(df[column].median(), 2)
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        mean = round(df[column].mean(), 2)
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)
        
def verify_invalid_values(df):
    # Exemplo: Verificar se há algum valor negativo na coluna 'Temperature'
    if (df['Temperature'] < 0).any():
        print("Valores inválidos encontrados na coluna 'Temperature'")
        # Aqui você pode adicionar um tratamento específico, por exemplo, substituir por NaN
        df['Temperature'] = df['Temperature'].apply(lambda x: np.nan if x < 0 else x)
    return df

# Função para obter índices de outliers usando o método IQR
def get_outliers_indices(df, col_name):
    if df[col_name].dtype not in ['int64', 'float64']:  # Verifica o tipo de dados
        return []
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_indices = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)].index
    return outliers_indices


# Carregar os dados
names = ['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption'] 
features =['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption'] 
output_file = '0-Dataset/Dados_Tratados.data'
input_file = '0-Dataset/Dados_Brutos.data'
df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names,      # Nome das colunas 
                    usecols = features, # Define as colunas que serão  utilizadas
                    na_values='?')      # Define que ? será considerado valores ausentes
df_original = df.copy()

# Determina os índices de todos os outliers
all_outliers_indices = set()  # Conjunto para armazenar todos os índices únicos de outliers

for column in df.columns:
    if column != "DateTime":
        outliers_indices = get_outliers_indices(df, column)
        all_outliers_indices.update(outliers_indices)

# Remove todas as linhas de outliers do DataFrame
df.drop(all_outliers_indices, inplace=True)

# Visualizar dados faltantes
print("Visualização de dados faltantes:")
matrix(df)
print("\n")

# Tratamento para registros duplicados
print(f"Número de linhas duplicadas antes do tratamento: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Número de linhas duplicadas após tratamento: {df.duplicated().sum()}\n")

# Tratamento para registros faltantes
columns_missing_value = df.columns[df.isnull().any()]
method = 'mean'
for c in columns_missing_value:
    UpdateMissingValues(df, c, method)

# Verificar e tratar valores inválidos ou inconsistentes
df = verify_invalid_values(df)
 
# Salva arquivo com o tratamento para dados faltantes
df.to_csv(output_file, header=False, index=False)  
    




