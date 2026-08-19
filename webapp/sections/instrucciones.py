import streamlit as st


def render():
    st.header("📋 Instrucciones")

    st.write(
        "Sigue estos pasos para aprovechar al máximo el modelo de predicción SIPSA:"
    )

    st.markdown(
        """
        **La idea del algoritmo es la siguiente:**

        1. El usuario debe seleccinar el departamento que desee estudiar, (puede escoger los departamentos disponibles, bogota esta incluido en la lista por el datset y Colombia tambien va a estar disponible si el usuario desea hacer un estudio a nivel nacional)
        2. Despues de escoger el departamento (excluyendo bogota y colombia) el usuario podra escoger entre de las ciudades de ese departamento que pertenecen y que tienen activo centros de venta mayorista para poductos alimentarios.
        3. Una vez escogido se le debe mostrar los productos disponibles en la ciudad.
        4. el usuario escoge el producto a investigar.
        5. Alli se empieza a realizar la estadistica, la idea es hacer selecion de todas las soliitudes realizadas por el usuario y realizar un problema de serie de tiempo
        - Opcion A: usar los promedios mensuales por año (solicitud del cliente)
        - Opcion B: recoletar todos los datos obtenidos de la solicitud de usuario por año
        6. Debe realizar el analisis, presentar las graficas, presentar promedio de mes mensual actual (hasta junio) y dar la probabilidad del precio del producto en la cuidad seleccionada.
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

