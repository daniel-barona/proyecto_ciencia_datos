import streamlit as st


def render():
    
    st.header("🏠 Introducción")

    st.write(
        """
        Bienvenido al **Modelo de Predicción SIPSA**. Esta aplicación tiene como
        objetivo analizar y predecir el comportamiento de los precios de productos
        agropecuarios a partir de los datos del **Sistema de Información de Precios
        y Abastecimiento del Sector Agropecuario (SIPSA)** para facilitar la toma de
        decisiones de todos los actores en los mercados mayoristas.
        """
    )

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fuente de datos", "SIPSA - DANE")
        st.link_button("Ir a la fuente ↗", "https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa/mayoristas-boletin-mensual-1")
    with col2:
        st.metric("Problema de prediccion", "Series de tiempo")
    with col3:
        st.caption("Dirigido a")
        st.markdown("**Comerciantes y consumidores de mercados mayoristas**")

    st.divider()

    st.subheader("¿Qué encontrarás aquí?")
    st.markdown(
        """
        - **Instrucciones:** cómo usar la aplicación paso a paso.
        - **EDA:** análisis exploratorio de los datos de precios.
        - **Predicción:** genera pronósticos con el modelo entrenado.
        """
    )
    st.divider()

    st.subheader("Modelos predictivos ultizados")
    st.markdown(
            """
                - **ARIMA** : modelo de series temporales utilizado para representar la dependencia entre las observaciones actuales y los valores y errores pasados. Se emplea para series que no presentan un componente estacional explícito.
                - **SARIMA**: extensión del modelo ARIMA que incorpora componentes estacionales. En este estudio, al trabajar con datos de frecuencia mensual, se consideró una periodicidad estacional de 12 meses, permitiendo modelar patrones que se repiten anualmente.
                - **Configuraciones ARIMA**: se evaluaron diferentes combinaciones de los parámetros p,d,q, incluyendo modelos automáticos y configuraciones específicas, con y sin deriva, con el fin de determinar la estructura que ofreciera el mejor ajuste para cada serie.
                - **Configuraciones SARIMA**: se probaron diferentes combinaciones de los parámetros no estacionales (p,d,q) y estacionales (P,D,Q) , incluyendo modelos con y sin deriva. Esto permitió comparar distintas representaciones de la tendencia, autocorrelación y estacionalidad presentes en las series.
                - **Naive estacional**: se puede utilizar como modelo de referencia o línea base, estimando el valor futuro a partir de la observación correspondiente al mismo periodo de la temporada anterior. Su inclusión permite determinar si los modelos ARIMA/SARIMA logran mejorar una estrategia de predicción sencilla.
            """
    )

    
