## Processo 6 - Previsão de Consumo


import pandas as pd

# Criando o DataFrame para o ano de 2018
dates_2018 = [(2018, mes, dia, hora) for mes in range(1, 13) for dia in range(1, 32) for hora in range(24)]

# Convertendo em um DataFrame
df_2018 = pd.DataFrame(dates_2018, columns=['ano', 'mes', 'dia', 'hora'])

# Adicionando valores médios para Temperatura e Umidade
df_2018['Temperatura'] = data['Temperatura'].mean()
df_2018['Umidade'] = data['Umidade'].mean()

# Reordenando as colunas para combinar com o dataset de treinamento
df_2018 = df_2018[['Temperatura', 'Umidade', 'ano', 'mes', 'dia', 'hora']]

# Aplicando a transformação
x_2018 = scaler.transform(df_2018)

# Fazendo previsões
previsoes_2018 = modelo.predict(x_2018)

# Retificando as previsões
min_positive_val = min(previsao for previsao in previsoes_2018 if previsao > 0)
previsoes_2018[previsoes_2018 <= 0] = min_positive_val

# Adicionando previsões ao DataFrame
df_2018['Consumo de energia Zona 1'] = previsoes_2018

# Salvar o DataFrame em um arquivo Excel
df_2018.to_excel("/content/drive/My Drive/0-Dataset/Previsoes_2018.xlsx", index=False)

print(df_2018.columns)

import plotly.express as px

# Convertendo a coluna 'DateTime' para o formato datetime
df_2018['DateTime'] = pd.to_datetime(df_2018['DateTime'])

# Reordenando as colunas do df_2018
cols = ['DateTime', 'Temperatura', 'Umidade', 'Consumo de energia Parcial']

# Certificando-se de que df_2018 tem as colunas corretas
df_2018 = df_2018[cols]

# Concatenando o DataFrame original e o da previsão
df_total = pd.concat([data, df_2018], ignore_index=True)

# Convertendo para o formato longo
data_melted = df_total.melt(id_vars='DateTime',
                            value_vars=['Consumo de energia Parcial'],
                            var_name='Zone', value_name='Power Consumption')

# Criando a animação
fig = px.line(data_melted,
              x='DateTime',
              y='Power Consumption',
              color='Zone',
              title='Consumo de Energia ao Longo do Tempo',
              labels={'Power Consumption': 'Consumo de Energia', 'DateTime': 'Data e Hora'},
              template='plotly_dark')

fig.show()

fig.write_html("6-Dataset/DadosAtemporal.html")