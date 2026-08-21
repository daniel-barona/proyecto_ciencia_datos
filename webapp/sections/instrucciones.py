import streamlit as st


def render():
    st.header("📋 Instrucciones")

    st.write(
        "Sigue estos pasos para aprovechar al máximo el modelo de predicción SIPSA:"
    )

    st.markdown(
        """
        **La idea del algoritmo es la siguiente:**
        1.    El usuario debe seleccionar el departamento que desee estudiar. Puede escoger entre los departamentos disponibles; Bogotá está incluido en la lista debido al dataset, y Colombia también estará disponible si el usuario desea realizar un estudio a nivel nacional.
        2.    Después de seleccionar el departamento, excluyendo Bogotá y Colombia, el usuario podrá escoger entre los municipios de ese departamento que cuentan con centros de venta mayorista activos para productos alimentarios.
        3.    Una vez seleccionado el municipio, se le deben mostrar los productos disponibles en la ciudad.
        4.    El usuario debe seleccionar el producto que desea investigar.
        5.    Una vez seleccionado el producto, se comenzará a realizar fase por fase el procesamiento de manera interna, preparando los datos que posteriormente se deben presentar al usuario.
        6.    Se debe realizar el análisis y mostrar al usuario el pronóstico de la predicción a 12 meses, presentando el comportamiento tanto en porcentaje como en precio.
        
        """
    )

    st.subheader("Recomendaciones")
    with st.expander("📂 Sobre los datos"):
        st.write(
            "Los datos provienen del SIPSA. las actualizaciones se realizan en tiempo real segun la plataforma "
            "para obtener predicciones más confiables."
        )
    with st.expander("⚙️ Sobre los parámetros"):
        st.write(
            "Puedes consultar un producto en el mercado mayorista de tu ciudad que se de tu interes "
            "según tus necesidades de análisis."
        )
    with st.expander("📈 Sobre los resultados"):
        st.write(
            "Los resultados son estimaciones basadas en datos históricos"
            "se presentan en varios modelos, y se basa en el registro del SIPSA"
        )

