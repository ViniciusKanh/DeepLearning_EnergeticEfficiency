

import pandas as pd
import numpy as np
from missingno import matrix


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
    # Exemplo: Verificar se há algum valor negativo na coluna 'Temperatura'
    if (df['Temperatura'] < 0).any():
        print("Valores inválidos encontrados na coluna 'Temperatura'")
        # Aqui você pode adicionar um tratamento específico, por exemplo, substituir por NaN
        df['Temperatura'] = df['Temperatura'].apply(lambda x: np.nan if x < 0 else x)
    return df

"""Detecção e Exclusão de Outliers na coluna 'Consumo de energia Zona 1'"""

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

names = ['DateTime','Temperatura','Umidade','Consumo de energia Zona 1']
features =['DateTime','Temperatura','Umidade','Consumo de energia Zona 1']
output_file = '0-Dataset/Dados_Tratados.data'
input_file = '0-Dataset/Dados_Brutos.data'
df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names,      # Nome das colunas
                    usecols = features, # Define as colunas que serão  utilizadas
                    na_values='?')      # Define que ? será considerado valores ausentes


df['Consumo de energia Zona 1'] = df['Consumo de energia Zona 1'].str.replace('.', '').astype(float)

"""Data Filtering"""

# Exemplo: Remova quaisquer linhas que não são relevantes para sua análise
df = df[df['Consumo de energia Zona 1'] > 0]

"""Detecção e Exclusão de Outliers"""

cols_for_outliers = ['Temperatura', 'Umidade', 'Consumo de energia Zona 1']
all_outliers_indices = []
for col in cols_for_outliers:
    outlier_indices = get_outliers_indices(df, col)
    if outlier_indices.size > 0:
        print(f"Outliers encontrados na coluna {col}.")
        all_outliers_indices.extend(outlier_indices.tolist())

# Removendo os outliers do DataFrame
all_outliers_indices = list(set(all_outliers_indices))
df.drop(index=all_outliers_indices, inplace=True)

"""Mostrando Dados Parcialmente"""

df_original = df.copy()
# Imprime as 15 primeiras linhas do arquivo
print("PRIMEIRAS 10 LINHAS\n")
print(df.head(10))
print("\n")

"""Imprimindo Informaçoes sobre os Dados"""

# Imprime informações sobre dos dados
print("INFORMAÇÕES GERAIS DOS DADOS\n")
print(df.info())
print("\n")

"""Imprime uma analise descritiva sobre dos dados"""

print("DESCRIÇÃO DOS DADOS\n")
print(df.describe())
print("\n")

"""Dados Duplicados"""

print(f"Número de linhas duplicadas antes do tratamento: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Número de linhas duplicadas após tratamento: {df.duplicated().sum()}\n")

"""Imprimindo Dados Faltantes"""

print("VALORES FALTANTES\n")
print(df.isnull().sum())
print("\n")

"""Tratamento para registros faltantes"""

columns_missing_value = df.columns[df.isnull().any()]
method = 'mean'
for c in columns_missing_value:
    UpdateMissingValues(df, c, method)
    
# Detecta outliers na coluna "Consumo de energia Zona 1"
outlier_indices = get_outliers_indices(df, 'Consumo de energia Zona 1')
if outlier_indices.size > 0:
    print(f"Outliers encontrados na coluna Consumo de energia Zona 1: {outlier_indices.size} registros.")
    df.drop(index=outlier_indices, inplace=True)
else:
    print("Nenhum outlier detectado na coluna 'Consumo de energia Zona 1'.")

"""Verificar e tratar valores inválidos ou inconsistentes"""

df = verify_invalid_values(df)

"""Salva arquivo com o tratamento para dados faltantes"""

df.to_csv(output_file, header=False, index=False)