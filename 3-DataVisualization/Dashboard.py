# Importação das bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento dos dados
@st.cache_data
def fetch_data():
    names = ['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption']
    features =['DateTime','Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2 Power Consumption','Zone 3 Power Consumption']
    input_file = '0-Dataset/Dados_Tratados.data'
    return pd.read_csv(input_file, names=names, usecols=features, na_values='?')
    pass

df = fetch_data()

# Convertendo a coluna 'DateTime' para o tipo datetime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Título
st.title("Análise de Eficiência Energética")


# Evolução temporal da temperatura, umidade e velocidade do vento
st.subheader("Evolução Temporal: Temperatura, Umidade e Velocidade do Vento")
fig, ax = plt.subplots(figsize=(12, 6))
df.set_index("DateTime")[["Temperature", "Humidity", "Wind Speed"]].plot(ax=ax)
st.pyplot(fig)

# Histogramas
variables = ["Temperature", "Humidity", "Wind Speed"]
for var in variables:
    st.subheader(f"Distribuição de {var}")
    plt.figure(figsize=(10, 5))
    sns.histplot(df[var], kde=True, bins=30)
    st.pyplot(plt)

# Consumo de Energia
st.subheader("Consumo de Energia para as Zonas 1, 2 e 3")
fig, ax = plt.subplots(figsize=(12, 6))
df.set_index("DateTime")[["Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]].plot(ax=ax)
st.pyplot(fig)

# Distribuição do consumo de energia
st.subheader("Distribuição do Consumo de Energia")
labels = ["Zone 1", "Zone 2", "Zone 3"]
sizes = [df["Zone 1 Power Consumption"].sum(), df["Zone 2 Power Consumption"].sum(), df["Zone 3 Power Consumption"].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
st.pyplot(plt)

# Análise dos Fluxos
st.subheader("Evolução dos 'general diffuse flows' e 'diffuse flows'")
fig, ax = plt.subplots(figsize=(12, 6))
df.set_index("DateTime")[["general diffuse flows", "diffuse flows"]].plot(ax=ax)
st.pyplot(fig)

st.subheader("Correlação entre 'general diffuse flows' e 'diffuse flows'")
plt.scatter(df["general diffuse flows"], df["diffuse flows"])
plt.xlabel("General Diffuse Flows")
plt.ylabel("Diffuse Flows")
st.pyplot(plt)
