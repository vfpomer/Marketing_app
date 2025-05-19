#------------Librerías-----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#---------------------------------------

#-----------------Configuración de la página------------
st.set_page_config(
    page_title="Panel de Análisis Marketing",
    page_icon="📢",
    layout="wide"
)

#--------------Título y descripción--------------------------------------
st.title("📢 Panel Interactivo de Campañas de Marketing")
st.markdown("""
Este panel te permite explorar datos de las campañas de Marketing.
Utiliza los filtros y selectores en la barra lateral para personalizar tu análisis.
""")
#---------------------------------------------------------------
#-----------------Carga de datos de ejemplo-----------------
@st.cache_data
def cargar_datos():
    # Datos de ejemplo de una campaña de marketing
    data = {
        'Fecha': pd.date_range(start='2024-01-01', periods=30, freq='D'),
        'Impresiones': np.random.randint(1000, 5000, 30),
        'Clics': np.random.randint(100, 1000, 30),
        'Conversiones': np.random.randint(10, 200, 30),
        'Gasto': np.random.uniform(100, 1000, 30).round(2)
    }
    df = pd.DataFrame(data)
    df['CTR'] = (df['Clics'] / df['Impresiones'] * 100).round(2)
    df['CPC'] = (df['Gasto'] / df['Clics']).round(2)
    df['CPA'] = (df['Gasto'] / df['Conversiones']).round(2)
    return df

df = cargar_datos()

#-----------------Visualización de datos-----------------
st.subheader("Vista general de la campaña")
st.dataframe(df)

#-----------------Gráficos principales-----------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df, x='Fecha', y=['Impresiones', 'Clics', 'Conversiones'],
                   title='Evolución diaria de Impresiones, Clics y Conversiones')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df, x='Fecha', y='Gasto', title='Gasto diario')
    st.plotly_chart(fig2, use_container_width=True)

#-----------------Métricas clave-----------------
st.subheader("Métricas clave de la campaña")
col3, col4, col5, col6 = st.columns(4)
col3.metric("Total Impresiones", f"{df['Impresiones'].sum():,}")
col4.metric("Total Clics", f"{df['Clics'].sum():,}")
col5.metric("Total Conversiones", f"{df['Conversiones'].sum():,}")
col6.metric("Gasto Total", f"${df['Gasto'].sum():,.2f}")