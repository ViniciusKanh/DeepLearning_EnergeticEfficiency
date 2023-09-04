import pandas as pd
import plotly.express as px

# Carregue sua base de dados
data = pd.read_csv("0 - Dataset/Dados_Brutos.data")
print(data.columns)


# Convertendo a coluna 'DateTime' para o tipo datetime
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Reformatando o DataFrame para o formato longo usando melt
data_melted = data.melt(id_vars='DateTime', 
                        value_vars=['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption'], 
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

fig.write_html("3-DataVisualization/DadosAtemporal.html")
