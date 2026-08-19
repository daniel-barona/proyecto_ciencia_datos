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
            - **SARIMA:** Modelo que predice series de tiempo considerando tendencia, estacionalidad y autocorrelación. Es adecuado cuando los datos presentan patrones que se repiten periódicamente.
            - **ARIMA (sin estacional):** Modelo que utiliza los valores pasados y los errores anteriores para realizar predicciones, sin considerar patrones estacionales.
            - **ARIMA (Configuración):** Versión específica del modelo ARIMA cuyos parámetros fueron ajustados para adaptarse al comportamiento de la serie de tiempo y mejorar la precisión de las predicciones.
            - **Naive estacional:** Variante del modelo Naive que predice el siguiente valor utilizando el observado en el mismo período de la temporada anterior (por ejemplo, el mismo mes del año pasado).
            """
    )

    
