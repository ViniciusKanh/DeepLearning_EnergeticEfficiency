# Importando bibliotecas
import streamlit as st
import pandas as pd
import plotly.express as px

# Função principal do Streamlit
def main():
    st.title('Exemplo de Dashboard com Streamlit')

    # Simulando alguns dados
    df = pd.DataFrame({
        'Categoria': ['A', 'B', 'C', 'D'],
        'Valores': [10, 30, 40, 20]
    })

    df_hist = pd.DataFrame({
        'Valores': [10, 20, 20, 30, 30, 30, 40, 40, 50]
    })

    df_map = pd.DataFrame({
        'lat': [-23.5617, -23.6825, -23.5083],
        'lon': [-46.6589, -46.6848, -46.6265],
        'Nome': ['Ponto1', 'Ponto2', 'Ponto3']
    })

    # Gráfico de Pizza
    fig_pizza = px.pie(df, names='Categoria', values='Valores', title='Gráfico de Pizza')
    st.plotly_chart(fig_pizza)

    # Histograma
    fig_hist = px.histogram(df_hist, x='Valores', title='Histograma')
    st.plotly_chart(fig_hist)

    # Mapa
    fig_map = px.scatter_mapbox(df_map, lat='lat', lon='lon', hover_name='Nome',
                                mapbox_style='open-street-map', zoom=10, title='Mapa')
    st.plotly_chart(fig_map)

# Executando a aplicação
if __name__ == "__main__":
    main()
