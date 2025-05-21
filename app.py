# ------------ Librerías -----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import traceback
# ----------------------------------------

# ------------ Configuración de la página ------------
st.set_page_config(
    page_title="Panel de Análisis Marketing",
    page_icon="📢",
    layout="wide"
)

# ------------ Título y descripción ------------
st.title("📢 Panel Interactivo de Campañas de Marketing")
st.markdown("""
Este panel te permite explorar datos de las campañas de Marketing.
Utiliza los filtros y selectores en la barra lateral para personalizar tu análisis.
""")

#------------------ Variables globales para inicialización segura----------------
filtered_df = None
df = None

# ------------ Función para cargar datos ------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("data/marketingcampaigns_limpio.csv")
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
    
    # Categorizar canales   

        conditions = [
            (df['channel'].str.contains('referral', case=False)),
            (df['channel'].str.contains('promotion', case=False)),
            (df['channel'].str.contains('paid', case=False)),
            (df['channel'].str.contains('organic', case=False))
        ]       
        choices = ['Referral', 'Promotion', 'Paid', 'Organic']
        df['canal_categoria'] = np.select(conditions, choices, default='No clasificado')
        
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.text(traceback.format_exc())
        return None

# ------------ Esquema de colores para canales ------------
channel_colors = {
    'referral': '#4E79A7',   # Azul suave
    'promotion': "#78C46E",  # Verde fresco
    'paid': "#EDAA68",       # Naranja cálido
    'organic': "#FC8688"     # Rojo coral
}

# ------------ Función para asegurar tamaños positivos ------------
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# ------------ Cargar y filtrar datos ------------
try:
    with st.spinner('Cargando datos...'):
        df = load_data()

    if df is not None and not df.empty:
        st.sidebar.header("📅 Filtros de Fecha")

        # Rango mínimo y máximo del DataFrame
        min_date = df['start_date'].min().date()
        max_date = df['end_date'].max().date()

        # Filtro de fechas en la barra lateral
        date_range = st.sidebar.date_input(
            "Selecciona el rango de fechas",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Aplicar filtro sin mostrar resultados
        if len(date_range) == 2:
            start_date, end_date = date_range

            # Filtrar DataFrame (no se muestra al usuario)
            filtered_df = df[
                (df['start_date'].dt.date >= start_date) &
                (df['end_date'].dt.date <= end_date)
            ].copy()
        else:
            # Si no hay selección válida, usar todo el DataFrame
            filtered_df = df.copy()


            # Validar si hay datos después del filtro de fecha
            if filtered_df.empty:
                st.warning("No hay campañas dentro del rango de fechas seleccionado.")

        # Filtro de ROI
        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("📈 Filtro de ROI\nRetorno sobre la inversión")

            roi_min_value = round(filtered_df['roi'].min(), 2)
            roi_max_value = round(filtered_df['roi'].max(), 2)

            if roi_min_value < roi_max_value:
                roi_range = st.sidebar.slider(
                    label="Selecciona el rango de ROI",
                    min_value=roi_min_value,
                    max_value=roi_max_value,
                    value=(roi_min_value, roi_max_value),
                    step=0.01,
                    help="Filtra campañas según su retorno de inversión (ROI)."
                )
                roi_min, roi_max = roi_range

                # Aplicar filtro
                filtered_df = filtered_df[
                    (filtered_df['roi'] >= roi_min) & (filtered_df['roi'] <= roi_max)
                ]
            else:
                st.sidebar.warning("⚠️ No hay suficiente variación en el ROI para aplicar el filtro.")


        #Filtro de presupuesto
        # Obtener los valores mínimos y máximos del presupuesto
            budget_min_value = filtered_df['budget'].min()
            budget_max_value = filtered_df['budget'].max()

            # Validar que el mínimo sea menor que el máximo
            if budget_min_value < budget_max_value:
                budget_range = st.sidebar.slider(
                    label="Selecciona el rango de Presupuesto",
                    min_value=float(budget_min_value),
                    max_value=float(budget_max_value),
                    value=(float(budget_min_value), float(budget_max_value)),
                    step=1000.0,
                    help="Filtra campañas según su presupuesto."
                )
                budget_min, budget_max = budget_range
                # Filtrar el DataFrame con base en el presupuesto seleccionado
                filtered_df = filtered_df[(filtered_df['budget'] >= budget_min) & (filtered_df['budget'] <= budget_max)]
            else:
                st.sidebar.warning("⚠️ No hay suficiente variación en el presupuesto para aplicar el filtro.")
    


        # Filtro por canal
        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("📊 Filtro por Canal")

            # Obtener los canales únicos
            unique_channels = filtered_df['channel'].unique()

            # Crear un multiselect para los canales
            selected_channels = st.sidebar.multiselect(
                "Selecciona los canales",
                options=unique_channels,
                default=unique_channels,
                help="Filtra campañas según el canal de marketing."
            )

            # Aplicar filtro silenciosamente (sin mostrar los datos)
            filtered_df = filtered_df[filtered_df['channel'].isin(selected_channels)]

        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("📊 Filtro por Tipo de Campaña")

            unique_types = sorted(filtered_df['type'].dropna().unique())

            selected_types = st.sidebar.multiselect(
                label="Selecciona los tipos de campaña",
                options=unique_types,
                default=unique_types,
                help="Filtra campañas según el tipo de campaña."
            )

            if selected_types:
                filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]


        # Mostrar cantidad de eventos filtrados
        st.sidebar.metric("Eventos seleccionados", len(filtered_df))
         #-------fin filtros

        # Verificar si hay datos después de aplicar los filtros
        if len(filtered_df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
        # Pestañas principales para organizar el panel
            main_tabs = st.tabs([
                "📊 Resumen General",
                "📡 Canal más Utilizado",
                "🏆 Mejor Campaña",
                "👥 B2B / B2C",
                "💰 Presupuesto vs Ingresos",
                "📈 Análisis Avanzado"
            ])
           
           #------------------Pestaña 1: Resumen General
            with main_tabs[0]:
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Eventos totales", len(filtered_df))
                col2.metric("Presupuesto medio", f"{filtered_df['budget'].mean():.2f} €")
                col3.metric("ROI promedio", f"{filtered_df['roi'].mean():.2f}")
                col4.metric("ROI promedio %", f"{filtered_df['roi'].mean() * 100:.2f} %")
                
                # Distribución de canales
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.subheader("Distribución de canales")
                    
                    fig_channel = px.histogram(
                        filtered_df,
                        x="channel",
                        nbins=30,
                        color="canal_categoria",
                        color_discrete_map=channel_colors,
                        labels={
                            "channel": "Canal", 
                            "count": "Frecuencia"
                        },
                        title="Distribución de canales por categoría"
                    )
                    fig_channel.update_layout(bargap=0.1)
                    st.plotly_chart(fig_channel, use_container_width=True, key="fig_channel")
                
                with col_dist2:
                    st.subheader("Ingreso medio por campaña")
                    fig_income = px.histogram(
                        filtered_df,
                        x="type",
                        y="revenue",
                        color="type",
                        histfunc="avg",  # <- ESTA LÍNEA es la clave para calcular el promedio
                        color_discrete_map=channel_colors,
                        labels={
                            "type": "Tipo de campaña", 
                            "revenue": "Ingreso medio"
                        },
                        title="Ingreso medio por campaña"
                    )
                    fig_income.update_layout(bargap=0.1)
                    st.plotly_chart(fig_income, use_container_width=True, key="fig_income")
                
                # Relación ROI por canal
                st.subheader("Relación de ROI por canal")
                fig_roi = px.histogram(
                    filtered_df,
                    x="roi",
                    color="canal_categoria",
                    color_discrete_map=channel_colors,
                    labels={
                        "roi": "ROI", 
                        "count": "Frecuencia"
                    },
                    title="Distribución de ROI por canal"
                )       
                fig_roi.update_layout(bargap=0.1)
                st.plotly_chart(fig_roi, use_container_width=True, key="fig_roi")
                
                # Top 10 campañas por ROI
                st.subheader("Top 10 campañas por ROI")
                top_campaigns = filtered_df.nlargest(10, 'roi')[['campaign_name', 'roi']]
                top_campaigns.columns = ['Campaña', 'ROI']
                fig_top_campaigns = px.bar(
                    top_campaigns,
                    x='ROI',
                    y='Campaña',
                    orientation='h',
                    text='ROI',
                    color='ROI',
                    color_continuous_scale='Viridis'
                )
                fig_top_campaigns.update_traces(textposition='outside')
                fig_top_campaigns.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_top_campaigns, use_container_width=True, key="fig_top_campaigns")
     
            #..------------------fin pestaña 1

            #------------------Pestaña 2: Canal más utilizado
            with main_tabs[1]:
                mark_tabs = st.tabs(["📊 Canal más utilizado", "📈 ROI por canal"])

                # 📌 Pestaña 1: Canal más utilizado
                with mark_tabs[0]:
                    st.subheader("Canal más utilizado")     
                    # Contar campañas por canal
                    channel_counts = filtered_df['channel'].value_counts().reset_index()
                    channel_counts.columns = ['Canal', 'Cantidad de campañas']
                    channel_counts['Canal'] = channel_counts['Canal'].str.capitalize()

                    # Gráfico de barras
                    fig_channel_counts = px.bar(
                        channel_counts,
                        x='Canal',
                        y='Cantidad de campañas',
                        color='Cantidad de campañas',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        labels={'Canal': 'Canal', 'Cantidad de campañas': 'Cantidad de campañas'},
                        title='Cantidad de campañas por canal'
                    )
                    fig_channel_counts.update_traces(texttemplate='%{y}', textposition='outside')
                    fig_channel_counts.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                    st.plotly_chart(fig_channel_counts, use_container_width=True, key="fig_channel_counts")

                    # Tabla por canal
                    st.subheader("Tabla de campañas por canal")
                    channel_table = filtered_df.groupby('channel').agg(
                        Total_Campañas=('campaign_name', 'count'),
                        Presupuesto_Medio=('budget', 'mean'),
                        ROI_Medio=('roi', 'mean'),
                        Ingreso_Medio=('revenue', 'mean')
                    ).reset_index()
                    channel_table['channel'] = channel_table['channel'].str.capitalize()
                    channel_table.columns = ['Canal', 'Total Campañas', 'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']
                    channel_table[['Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']] = channel_table[[
                        'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio'
                    ]].round(2)
                    st.dataframe(channel_table, use_container_width=True)

                    #texto adicional
                    st.markdown("""
                        Distribución y Efectividad de Canales con todos los datos:
                        \n- La distribución de canales es equilibrada: promotion (27.2%), referral (25%), organic (24.1%) y paid (23.7%).
                        \n- El canal promotion destaca con el mejor ROI, seguido por organic, mientras que referral presenta el ROI más bajo.
                        \n- La estrategia multicanal muestra ser efectiva, sin dependencia excesiva de un solo canal.
                        """)
                    

                    # 📌 Pestaña 2: ROI
                    with mark_tabs[1]:
                        st.subheader("ROI promedio por canal")

                        # Calcular ROI promedio por canal
                        roi_por_canal = (
                            filtered_df.groupby("channel")["roi"]
                            .mean()
                            .reset_index()
                            .rename(columns={"channel": "Canal", "roi": "ROI Promedio"})
                            .sort_values(by="ROI Promedio", ascending=False)
                        )

                        # Gráfico ROI promedio
                        fig_roi_channel = px.bar(
                            roi_por_canal,
                            x='Canal',
                            y='ROI Promedio',
                            color='ROI Promedio',
                            color_continuous_scale=px.colors.sequential.Viridis,
                            labels={'Canal': 'Canal', 'ROI Promedio': 'ROI Promedio'},
                            title='ROI promedio por canal'
                        )
                        fig_roi_channel.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                        fig_roi_channel.update_layout(xaxis_title="Canal", yaxis_title="ROI Promedio", height=400)
                        st.plotly_chart(fig_roi_channel, use_container_width=True,key="fig_roi_channel")

                        # Tabla ROI promedio
                        st.subheader("Tabla de ROI promedio por canal")
                        roi_table = filtered_df.groupby('channel').agg(
                            Total_Campañas=('campaign_name', 'count'),
                            Presupuesto_Medio=('budget', 'mean'),
                            ROI_Medio=('roi', 'mean'),
                            Ingreso_Medio=('revenue', 'mean')
                        ).reset_index()
                        roi_table['channel'] = roi_table['channel'].str.capitalize()
                        roi_table.columns = ['Canal', 'Total Campañas', 'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']
                        roi_table[['Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']] = roi_table[[
                            'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio'
                        ]].round(2)
                        st.dataframe(roi_table, use_container_width=True)

                        # Distribución del ROI
                        st.subheader("Distribución del ROI")
                        st.markdown("Este gráfico muestra la distribución del ROI.")
                        fig_roi_dist = px.histogram(
                            filtered_df,
                            x="roi",
                            color="canal_categoria",
                            color_discrete_map=channel_colors,
                            labels={"roi": "ROI", "count": "Frecuencia"},
                            title="Distribución del ROI"
                        )
                        fig_roi_dist.update_layout(bargap=0.1)
                        st.plotly_chart(fig_roi_dist, use_container_width=True,key="fig_roi_dist")

                        # Dispersión ROI vs Presupuesto
                        st.subheader("ROI vs Presupuesto")
                        st.markdown("Este gráfico muestra la dispersión de ROI vs Presupuesto por canal.")
                        fig_roi_budget = px.scatter(
                            filtered_df,
                            x='budget',
                            y='roi',
                            color='channel',
                            hover_name='campaign_name',
                            size='revenue',
                            size_max=20,
                            color_discrete_map=channel_colors,
                            labels={'budget': 'Presupuesto', 'roi': 'ROI'},
                            title='ROI vs Presupuesto por Canal'
                        )
                        fig_roi_budget.update_traces(marker=dict(opacity=0.7))
                        fig_roi_budget.update_layout(xaxis_title="Presupuesto", yaxis_title="ROI", height=400)
                        st.plotly_chart(fig_roi_budget, use_container_width=True,key="fig_roi_budget")

                        #texto adicional
                        st.markdown("""
                            El ROI promedio es 0.53, con desviación estándar de 0.26.
                            Factores asociados a ROI alto:
                            \n- Uso de canales orgánicos y promocionales.
                            \n- Campañas tipo podcast y social media.
                            \n- Ejecución en segundo trimestre.
                            """)

            #..------------------fin pestaña 2

            #------------------Pestaña 3: Mejor Campaña
            with main_tabs[2]:
                st.subheader("Mejor Campaña")

                # Verificar que el dataframe no esté vacío
                if filtered_df is not None and not filtered_df.empty:

                    # Calcular la mejor campaña (mayor ROI)
                    best_campaign = filtered_df.loc[filtered_df['roi'].idxmax()]
                    best_campaign_name = best_campaign['campaign_name']
                    best_campaign_roi = best_campaign['roi']
                    best_campaign_budget = best_campaign['budget']
                    best_campaign_revenue = best_campaign['revenue']

                    # Mostrar información
                    st.write(f"**📛 Nombre de la campaña:** {best_campaign_name}")
                    st.write(f"**📈 ROI:** {best_campaign_roi:.2f}")
                    st.write(f"**💰 Presupuesto:** {best_campaign_budget:.2f} €")
                    st.write(f"**💵 Ingreso:** {best_campaign_revenue:.2f} €")

                    # Gráfico de dispersión
                    fig_best_campaign = px.scatter(
                        filtered_df,
                        x='budget',
                        y='roi',
                        color='channel',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'roi': 'ROI'
                        },
                        title='Mejor Campaña por ROI'
                    )
                    fig_best_campaign.update_traces(marker=dict(opacity=0.7))
                    fig_best_campaign.update_layout(
                        xaxis_title="Presupuesto",
                        yaxis_title="ROI",
                        height=400
                    )
                    st.plotly_chart(fig_best_campaign, use_container_width=True,key="fig_best_campaign")

                    # Tabla de la mejor campaña
                    best_campaign_table = filtered_df.loc[[filtered_df['roi'].idxmax()]][[
                        'campaign_name', 'start_date', 'end_date', 'channel', 'type', 'budget', 'roi'
                    ]].copy()

                    # Asegurar que las fechas sean datetime
                    best_campaign_table['start_date'] = pd.to_datetime(best_campaign_table['start_date'], errors='coerce')
                    best_campaign_table['end_date'] = pd.to_datetime(best_campaign_table['end_date'], errors='coerce')

                    # Renombrar y formatear columnas
                    best_campaign_table.columns = ['Nombre de la campaña', 'Fecha de inicio', 'Fecha de fin', 'Canal', 'Tipo', 'Presupuesto', 'ROI']
                    best_campaign_table['Fecha de inicio'] = best_campaign_table['Fecha de inicio'].dt.date
                    best_campaign_table['Fecha de fin'] = best_campaign_table['Fecha de fin'].dt.date

                    st.dataframe(best_campaign_table, use_container_width=True)

                    # Tabla agregada por canal
                    st.subheader("Tabla de campañas por canal")
                    channel_table = filtered_df.groupby('channel').agg(
                        Total_Campañas=('campaign_name', 'count'),
                        Presupuesto_Medio=('budget', 'mean'),
                        ROI_Medio=('roi', 'mean'),
                        Ingreso_Medio=('revenue', 'mean')
                    ).reset_index()
                    channel_table['channel'] = channel_table['channel'].str.capitalize()
                    channel_table.columns = ['Canal', 'Total Campañas', 'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']
                    channel_table['Total Campañas'] = channel_table['Total Campañas'].astype(int)
                    channel_table[['Presupuesto Medio', 'ROI Medio', 'Ingreso Medio']] = channel_table[[
                        'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio'
                    ]].round(2)
                    st.dataframe(channel_table, use_container_width=True)

                    #campaña que genera más ingresos
                    # Crear una nueva columna para el tamaño con valores positivos
                    filtered_df['roi_size'] = filtered_df['roi'].clip(lower=0.01)


                    # Gráfico de dispersión corregido
                    fig_highest_revenue_campaign = px.scatter(
                        filtered_df,
                        x='budget',
                        y='revenue',
                        color='channel',
                        hover_name='campaign_name',
                        size='roi_size',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'revenue': 'Ingreso'
                        },
                        title='Campaña con mayor ingreso'
                    ) 
                  

                    #que campaña tiene mejor conversión
                    # Calcular la tasa de conversión si no existe
                    if 'conversion_rate' not in filtered_df.columns:
                        filtered_df['conversion_rate'] = filtered_df['conversions'] / filtered_df['visits']

                    # Obtener la campaña con mejor conversión
                    best_conversion_campaign = filtered_df.loc[filtered_df['conversion_rate'].idxmax()]
                    st.subheader("📊 Campaña con mejor conversión")

                    # Mostrar información
                    st.write(f"**🎯 Campaña:** {best_conversion_campaign['campaign_name']}")
                    st.write(f"**📈 Tasa de conversión:** {best_conversion_campaign['conversion_rate']:.2%}")
                    st.write(f"**💰 Ingreso:** {best_conversion_campaign['revenue']:.2f} €")
                    st.write(f"**💵 Presupuesto:** {best_conversion_campaign['budget']:.2f} €")

                    # Gráfico de dispersión: Presupuesto vs Conversión
                    fig_best_conversion = px.scatter(
                        filtered_df,
                        x='budget',
                        y='conversion_rate',
                        color='channel',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'conversion_rate': 'Tasa de conversión'
                        },
                        title='Tasa de conversión por campaña'
                    )
                    fig_best_conversion.update_traces(marker=dict(opacity=0.7))
                    fig_best_conversion.update_layout(
                        xaxis_title="Presupuesto",
                        yaxis_title="Tasa de conversión",
                        height=400
                    )
                    st.plotly_chart(fig_best_conversion, use_container_width=True,key="fig_best_conversion")

                    # Tabla de la campaña con mejor conversión
                    best_conversion_table = filtered_df.loc[[filtered_df['conversion_rate'].idxmax()]][[
                        'campaign_name', 'start_date', 'end_date', 'channel', 'type', 'budget', 'revenue', 'conversion_rate'
                    ]].copy()

                    # Asegurar que las fechas sean datetime
                    best_conversion_table['start_date'] = pd.to_datetime(best_conversion_table['start_date'], errors='coerce')
                    best_conversion_table['end_date'] = pd.to_datetime(best_conversion_table['end_date'], errors='coerce')

                    # Renombrar columnas
                    best_conversion_table.columns = [
                        'Nombre de la campaña', 'Fecha de inicio', 'Fecha de fin',
                        'Canal', 'Tipo', 'Presupuesto', 'Ingreso', 'Tasa de conversión'
                    ]
                    best_conversion_table['Fecha de inicio'] = best_conversion_table['Fecha de inicio'].dt.date
                    best_conversion_table['Fecha de fin'] = best_conversion_table['Fecha de fin'].dt.date
                    best_conversion_table['Tasa de conversión'] = best_conversion_table['Tasa de conversión'].map("{:.2%}".format)

                    # Mostrar tabla
                    st.dataframe(best_conversion_table, use_container_width=True)

                  
                   # Filtrar campañas con ROI > 0.5 e ingresos > 500,000
                    high_roi_campaigns = filtered_df[
                        (filtered_df['roi'] > 0.5) & 
                        (filtered_df['revenue'] > 500000)
                    ].copy()

                    # Redondear valores
                    high_roi_campaigns['roi'] = high_roi_campaigns['roi'].round(2)
                    high_roi_campaigns['revenue'] = high_roi_campaigns['revenue'].round(2)

                    # Crear gráfico de dispersión
                    st.subheader("📊 Campañas con ROI > 0.5 e ingresos > 500,000")
                    fig_high_roi_campaigns = px.scatter(
                        high_roi_campaigns,
                        x='roi',
                        y='revenue',
                        color='campaign_name',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        labels={
                            'roi': 'ROI',
                            'revenue': 'Ingreso'
                        },
                        title='Campañas con ROI > 0.5 e Ingreso > 500,000',
                    )
                    fig_high_roi_campaigns.update_traces(marker=dict(opacity=0.7))
                    fig_high_roi_campaigns.update_layout(
                        xaxis_title="ROI",
                        yaxis_title="Ingreso",
                        height=400
                    )
                    st.plotly_chart(fig_high_roi_campaigns, use_container_width=True,key="fig_high_roi_campaigns")

                    # Crear tabla y mostrarla
                    tabla_campanias_altas = high_roi_campaigns[['campaign_name', 'roi', 'revenue']].copy()
                    tabla_campanias_altas.columns = ['Nombre de la campaña', 'ROI', 'Ingreso']
                    st.subheader("📋 Tabla de campañas")
                    st.dataframe(tabla_campanias_altas, use_container_width=True)

                    #Top 5 mejores campañas
                    st.subheader("Top 5 mejores campañas")
                    top_5_campaigns = filtered_df.nlargest(5, 'roi')[['campaign_name', 'roi']]
                    top_5_campaigns.columns = ['Campaña', 'ROI']
                    fig_top_5_campaigns = px.bar(
                        top_5_campaigns,
                        x='ROI',
                        y='Campaña',
                        orientation='h',
                        text='ROI',
                        color='ROI',
                        color_continuous_scale='Viridis'
                    )
                    fig_top_5_campaigns.update_traces(textposition='outside')
                    fig_top_5_campaigns.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                    st.plotly_chart(fig_top_5_campaigns, use_container_width=True, key="top_5_campaigns_chart")


                    #campaña con mas beneficio neto
                    st.subheader("Campaña con mayor beneficio neto")
                    # Calcular el beneficio neto    
                    filtered_df['net_profit'] = filtered_df['revenue'] - filtered_df['budget']
                    # Obtener la campaña con mayor beneficio neto
                    highest_profit_campaign = filtered_df.loc[filtered_df['net_profit'].idxmax()]
                    st.write(f"**📈 Campaña:** {highest_profit_campaign['campaign_name']}")
                    st.write(f"**💰 Beneficio neto:** {highest_profit_campaign['net_profit']:.2f} €")
                    st.write(f"**💵 Presupuesto:** {highest_profit_campaign['budget']:.2f} €")
                    st.write(f"**💵 Ingreso:** {highest_profit_campaign['revenue']:.2f} €")
                    # Gráfico de dispersión: Presupuesto vs Beneficio neto
                    fig_highest_profit_campaign = px.scatter(
                        filtered_df,
                        x='budget',
                        y='net_profit',
                        color='channel',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'net_profit': 'Beneficio neto'
                        },
                        title='Campaña con mayor beneficio neto'
                    )
                    fig_highest_profit_campaign.update_traces(marker=dict(opacity=0.7))
                    fig_highest_profit_campaign.update_layout(
                        xaxis_title="Presupuesto",
                        yaxis_title="Beneficio neto",
                        height=400
                    )
                    st.plotly_chart(fig_highest_profit_campaign, use_container_width=True ,key="fig_highest_profit_campaign")
                    # Tabla de la campaña con mayor beneficio neto
                    highest_profit_table = filtered_df.loc[[filtered_df['net_profit'].idxmax()]][[
                        'campaign_name', 'start_date', 'end_date', 'channel', 'type', 'budget', 'net_profit'
                    ]].copy()
                    # Asegurar que las fechas sean datetime
                    highest_profit_table['start_date'] = pd.to_datetime(highest_profit_table['start_date'], errors='coerce')
                    highest_profit_table['end_date'] = pd.to_datetime(highest_profit_table['end_date'], errors='coerce')
                    # Renombrar columnas

                    highest_profit_table.columns = [
                        'Nombre de la campaña', 'Fecha de inicio', 'Fecha de fin',
                        'Canal', 'Tipo', 'Presupuesto', 'Beneficio neto'
                    ]
                    highest_profit_table['Fecha de inicio'] = highest_profit_table['Fecha de inicio'].dt.date
                    highest_profit_table['Fecha de fin'] = highest_profit_table['Fecha de fin'].dt.date
                    # Mostrar tabla
                    st.dataframe(highest_profit_table, use_container_width=True)
                  



                    #texto adicional
                    st.markdown("""
                        Rendimiento por Tipo de Campaña:
                        \n- Los webinars muestran la mejor tasa de conversión (55.64%), seguidos por social media (53.96%).
                        \n- Las campañas de podcast y social media generan los mayores ingresos promedio (~529,000€).
                        \n- Los eventos presenciales muestran el rendimiento más bajo tanto en conversión como en ingresos.
                                
                        \nCampañas de Alto Rendimiento:
                        \n- 10 campañas superan ROI > 0.5 e ingresos > 500,000€.
                        \n- Predominan canales organic y paid.
                        \n- Destacan tipos podcast y social media.
                    """)
      

                else:
                    st.warning("No hay campañas disponibles para mostrar.")

            #..------------------fin pestaña 3

            #------------------Pestaña 4: B2B / B2C
            with main_tabs[3]:
                st.subheader("📊 Comparación: B2B vs B2C")

                if filtered_df is not None and not filtered_df.empty:
                    if 'conversion_rate' not in filtered_df.columns:
                        filtered_df['conversion_rate'] = filtered_df['conversions'] / filtered_df['visits']

                    fig_b2b_b2c = px.scatter(
                        filtered_df,
                        x='budget',
                        y='conversion_rate',
                        color='type',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'conversion_rate': 'Tasa de conversión'
                        },
                        title='Tasa de conversión por tipo de campaña'
                    )
                    fig_b2b_b2c.update_traces(marker=dict(opacity=0.7))
                    fig_b2b_b2c.update_layout(
                        xaxis_title="Presupuesto",
                        yaxis_title="Tasa de conversión",
                        height=400
                    )

                 

                    # Gráfico: ROI vs Tasa de conversión
                    fig_roi_conversion = px.scatter(
                        filtered_df,
                        x='roi',
                        y='conversion_rate',
                        color='type',
                        hover_name='campaign_name',
                        size='revenue',
                        size_max=20,
                        color_discrete_map=channel_colors,
                        labels={'roi': 'ROI', 'conversion_rate': 'Tasa de conversión'},
                        title='ROI vs Tasa de conversión (B2B vs B2C)'
                    )
                    fig_roi_conversion.update_traces(marker=dict(opacity=0.7))
                    fig_roi_conversion.update_layout(height=400)
                    st.plotly_chart(fig_roi_conversion, use_container_width=True, key="roi_conversion_chart")

                    # Tabla resumen por tipo
                    conversion_table = filtered_df.groupby('type').agg(
                        Total_Campañas=('campaign_name', 'count'),
                        Presupuesto_Medio=('budget', 'mean'),
                        ROI_Medio=('roi', 'mean'),
                        Ingreso_Medio=('revenue', 'mean'),
                        Tasa_Conversión_Media=('conversion_rate', 'mean')
                    ).reset_index()

                    conversion_table['type'] = conversion_table['type'].str.capitalize()
                    conversion_table.columns = ['Tipo', 'Total Campañas', 'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio', 'Tasa de Conversión Media']
                    conversion_table['Total Campañas'] = conversion_table['Total Campañas'].astype(int)
                    conversion_table[['Presupuesto Medio', 'ROI Medio', 'Ingreso Medio', 'Tasa de Conversión Media']] = conversion_table[[
                        'Presupuesto Medio', 'ROI Medio', 'Ingreso Medio', 'Tasa de Conversión Media'
                    ]].round(2)

                    st.subheader("📋 Resumen estadístico por tipo")
                    st.dataframe(conversion_table, use_container_width=True)

                     #texto adicional
                    st.markdown("""
                        Comparación B2B vs B2C:
                        \n- No existen diferencias estadísticamente significativas (p-valor = 0.2775).
                        \n- B2B muestra una conversión ligeramente superior (55.02% vs 53.20%).
                        \n- La variabilidad es similar en ambos segmentos

                    """)

                else:
                    st.warning("⚠️ No hay campañas disponibles para mostrar.")
                #..------------------fin pestaña 4

                #------------------Pestaña 5: Presupuesto vs ingresos
            with main_tabs[4]:
               # Subtítulo de la sección
                st.subheader("📊 Presupuesto vs Ingresos")

                # Verificar que el dataframe esté disponible y no vacío
                if filtered_df is not None and not filtered_df.empty:
                    # Calcular correlación
                    correlation = filtered_df['budget'].corr(filtered_df['revenue'])

                    # Mostrar valor de correlación
                    st.write(f"**🔗 Correlación entre presupuesto e ingresos (Pearson):** {correlation:.2f}")

                    # Interpretación rápida
                    if correlation > 0.7:
                        st.success("Existe una correlación fuerte y positiva: a mayor presupuesto, mayores ingresos.")
                    elif correlation > 0.4:
                        st.info("Existe una correlación moderada: el presupuesto influye en los ingresos, pero no completamente.")
                    elif correlation > 0:
                        st.warning("Correlación débil: hay una ligera relación positiva, pero no es concluyente.")
                    else:
                        st.error("No hay correlación positiva entre presupuesto e ingresos.")

                    # Gráfico de dispersión
                    fig_budget_vs_revenue = px.scatter(
                        filtered_df,
                        x='budget',
                        y='revenue',
                        color='channel',
                        hover_name='campaign_name',
                        trendline='ols',  # Línea de regresión
                        color_discrete_map=channel_colors,
                        labels={
                            'budget': 'Presupuesto',
                            'revenue': 'Ingreso'
                        },
                        title='Presupuesto vs Ingreso por campaña'
                    )
                    fig_budget_vs_revenue.update_traces(marker=dict(opacity=0.7))
                    fig_budget_vs_revenue.update_layout(height=400)
                    st.plotly_chart(fig_budget_vs_revenue, use_container_width=True, key="budget_vs_revenue_chart")

                     #texto adicional
                    st.markdown("""
                        Correlación Presupuesto-Ingresos:
                        \n- No existe correlación fuerte entre presupuesto e ingresos.
                        \n- La eficiencia en la asignación de recursos es más importante que el volumen de inversión.

                    """)

                else:
                    st.warning("No hay datos disponibles para calcular la correlación.")


                #-------Fin de la pestaña 5

                #-------------------Pestaña 6: Correlación
                with main_tabs[5]:
                    st.subheader("📅 Patrones Estacionales o Temporales")

                    if filtered_df is not None and not filtered_df.empty:
                        # Asegurar que las fechas sean tipo datetime
                        filtered_df['start_date'] = pd.to_datetime(filtered_df['start_date'], errors='coerce')

                        # Crear una nueva columna de mes/año
                        filtered_df['month'] = filtered_df['start_date'].dt.to_period('M').astype(str)

                        # Agrupar por mes y calcular métricas clave
                        monthly_metrics = filtered_df.groupby('month').agg({
                            'revenue': 'sum',
                            'budget': 'sum',
                            'roi': 'mean',
                            'conversion_rate': 'mean' if 'conversion_rate' in filtered_df.columns else lambda x: (filtered_df['conversions'] / filtered_df['visits']).mean()
                        }).reset_index()

                        # Gráfico de ingresos mensuales
                        fig_revenue_trend = px.line(
                            monthly_metrics,
                            x='month',
                            y='revenue',
                            markers=True,
                            labels={'month': 'Mes', 'revenue': 'Ingreso Total'},
                            title='Ingreso mensual total'
                        )
                        fig_revenue_trend.update_layout(xaxis_title='Mes', yaxis_title='Ingreso', height=400)
                        st.plotly_chart(fig_revenue_trend, use_container_width=True,  key="fig_revenue_trend")
                       

                        # Gráfico de ROI mensual
                        fig_roi_trend = px.line(
                            monthly_metrics,
                            x='month',
                            y='roi',
                            markers=True,
                            labels={'month': 'Mes', 'roi': 'ROI Promedio'},
                            title='Evolución mensual del ROI'
                        )
                        fig_roi_trend.update_layout(xaxis_title='Mes', yaxis_title='ROI', height=400)
                        st.plotly_chart(fig_roi_trend, use_container_width=True, key="fig_roi_trend")

                        # Tabla resumen
                        monthly_metrics.columns = ['Mes', 'Ingreso Total', 'Presupuesto Total', 'ROI Promedio', 'Tasa de Conversión Media']
                        monthly_metrics[['Ingreso Total', 'Presupuesto Total', 'ROI Promedio', 'Tasa de Conversión Media']] = monthly_metrics[
                            ['Ingreso Total', 'Presupuesto Total', 'ROI Promedio', 'Tasa de Conversión Media']
                        ].round(2)
                        st.dataframe(monthly_metrics, use_container_width=True)


                          #texto adicional
                        st.markdown("""
                                Patrones Estacionales:
                                \n- Segundo trimestre muestra el mejor rendimiento.
                                \n- Cuarto trimestre presenta caídas significativas, especialmente diciembre.
                                \n- Primer trimestre 2025 muestra recuperación notable en conversión (65%).
                                    
                                📌 Recomendaciones Estratégicas:
                                \n- Priorizar campañas de podcast y social media en canales orgánicos y promocionales.
                                \n- Concentrar inversiones importantes en Q2.
                                \n- Optimizar o reducir campañas en Q4.
                                \n- Mantener estrategia multicanal balanceada.
                                \n- Enfocarse en eficiencia presupuestaria más que en volumen.
                                \n- Implementar estrategias específicas para B2B y B2C según temporada.
                                \n- El éxito en marketing digital depende más de la optimización táctica y temporal que del volumen de inversión, destacando la importancia de una estrategia diversificada y bien temporizada.
                        """)


                    else:
                        st.warning("No hay campañas disponibles para analizar patrones temporales.")
                #..------------------fin pestaña 6
                #-------------------- Fin de las pestañas principales
            
        #-------------------- Descargable ---------------------------------
            # Tabla de datos (expandible)
            with st.expander("Ver datos en formato tabla"):
                try:
                    # Columnas disponibles para mostrar
                    display_cols = [col for col in ['start_date', 'end_date', 'campaign_name', 'channel', 'type', 'budget', 'roi'] if col in filtered_df.columns]
                  # Opciones de ordenamiento
                    sort_col = st.selectbox(
                        "Ordenar por",
                        options=display_cols,
                        index=0
                    )
                    
                    sort_order = st.radio(
                        "Orden",
                        options=['Descendente', 'Ascendente'],
                        index=0,
                        horizontal=True
                    )
                    
                    # Ordenar datos
                    sorted_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == 'Ascendente')
                    )
                    
                    # Mostrar tabla
                    st.dataframe(
                        sorted_df[display_cols],
                        use_container_width=True
                    )
                    
                    # Opción para descargar datos filtrados
                    csv = sorted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar datos filtrados (CSV)",
                        data=csv,
                        file_name="marketingcampaign.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error al mostrar la tabla de datos: {e}")
        
    #else final
    else:
        st.warning("No hay datos disponibles.")
        df_filtered = None

except Exception as e:
    st.error(f"Error al ejecutar la aplicación: {e}")
    st.text(traceback.format_exc())
    df_filtered = None

# ------------ Información del dashboard ------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Acerca de este Panel**

Este panel muestra datos de una campaña de Marketing de un canal de audiencia.
\nDesarrollado con Streamlit y Plotly Express.
""")

  