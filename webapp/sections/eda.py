import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from backend.config import TRUSTED_DIR

def render():
    st.header("📊 Análisis Exploratorio de Datos (EDA)")
    st.write(
            "En esta sección se visualizará el análisis exploratorio de los datos "
            "de precios del SIPSA, tendrás la oportunidad de experimentar y entender " \
            "el comportamiento, variable objetivo y funcionamiento de los datos"
    )

    st.divider()

    st.header("INFORMACIÓN PREVIA")

    st.markdown(
        """
        - **Objetivo del proyecto:** Se quiere conocer el precio futuro de un producto "x" en un mercado "x"
        - **Problema seleccionado:** Serie de tiempo
        - **Variable objetivo:** precio_promedio_kg
        - **Preparación:** Seleccionar las variables para el algoritmo y posteriormente el modelo
        """
    )

    st.divider()

    st.header("SE REALIZA EL LLAMADO AL ARCHIVO Y VISUALIZACIÓN DE LOS DATOS")
    
    df = pd.read_parquet(TRUSTED_DIR / "SIPSA_2013_2026_trusted.parquet")
    st.write(df)

    st.text("Se observa la descripción del dataset")
    st.write(df.describe())

    st.divider()
    st.text("Se observa la info del dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.divider()
    st.text("Se observa la descripción de la variable objetivo")
    st.write(df["precio_promedio_kg"].describe())
    st.divider()
    st.header("ESTUDIO DE LAS VARIABLES")
    st.subheader("1. Variable: GRUPO")
    st.markdown(
        """
        1.1 Presentar los grupos y la cantidad de productos que presentan en el dataset, a través del tiempo.
        - **Pregunta:** ¿Cuáles son los grupos y cuál es la cantidad de productos que presentan?
        """
    )
    st.write(df.groupby('grupo')['producto'].nunique().sort_values(ascending=False).reset_index(name='Cantidad_Productos'))

    st.write("Podemos observar que se va a trabajar con 8 grupos, " \
    "el grupo que más productos ha presentado son las frutas " \
    "y el que menos productos presentó fueron los lácteos")

    st.markdown(
        """
        1.2 Mostrar el producto que más se comercializó por grupo en el actual año.
        - **Pregunta:** ¿Cuál es el producto más comercializado o que más registros realizó por cada grupo en 2026?
        """
    )
    productoMayorFrecuencia2026 = (
    df[df["fecha"].dt.year == 2026]
        .groupby(["grupo", "producto"])
        .size()
        .reset_index(name="Cantidad")
        .sort_values(["grupo", "Cantidad"], ascending=[True, False])
        .groupby("grupo", as_index=False)
        .first()
    )

    st.write(productoMayorFrecuencia2026)

    st.write("Podemos observar los productos que más se comercializan por grupo para este año; podemos ver que aunque en la anterior los lácteos marcaban de últimos, aquí no lo son, suben una posición. También que las frutas siguen manteniendo la corona de los grupos y de los productos por grupo; observemos el tomate de árbol, se encontró registrado **225** veces en todos los mercados de Colombia"
    )
    st.markdown(
        """
        1.3 Ahora se puede ver el producto más costoso (por promedio) por cada grupo en el actual 2026.
        - **Pregunta:** ¿El producto más frecuente es el más costoso (en promedio), o cuál es?
        """
    )
    productoCaroDelGrupo2026 = (
    df[df["fecha"].dt.year == 2026]
        .groupby(['grupo', 'producto'])['precio_promedio_kg']
        .mean()
        .round(2)
        .reset_index(name='Precio_Promedio')
        .sort_values(['grupo', 'Precio_Promedio'], ascending=[True, False])
        .groupby('grupo')
        .head(1)
    )
    st.write(productoCaroDelGrupo2026)
    st.write("El promedio se realiza porque se puede presentar que un producto que presenta un precio muy pequeño tuvo un pico muy alto por una situación alterna; encontramos que los procesados tienen el producto más caro del mercado, que es el **café instantáneo**") 
    st.write("**En razón de eso podemos tomar cualquier serie y graficarla, si tomamos la última gráfica perteneciente al producto más costoso**")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        productoCaroDelGrupo2026["producto"],
        productoCaroDelGrupo2026["Precio_Promedio"]
    )

    ax.set_title("Producto con mayor precio promedio por grupo")
    ax.set_xlabel("Producto")
    ax.set_ylabel("Precio promedio kg")
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.write("**Conclusión de estudio** La variable del grupo tiene una alta relación con la variable producto, aunque no va a ser utilizada para el modelo; es interesante que se pueda llegar a usar. El problema es que el modelo puede usar otras variables para llegar más cerca de lo solicitado, que es **predecir el precio futuro del producto**")

    st.divider()
    
    st.header("Estudio de la variable 'Mercado'")

    st.markdown(
        """
            2.1 Para iniciar se presentará los mercados disponibles en el año 2026.
            - **Pregunta:** ¿Cuáles son los mercados presentados en el 2026?
        """
    )
    mercados_2026 = (
        df.loc[df["fecha"].dt.year == 2026, ["mercado"]]
        .drop_duplicates()
        .dropna()
        .sort_values("mercado")
        .reset_index(drop=True)
        )
    st.write(mercados_2026)
    st.write("En el año vigente nos encontramos con un total de 138 mercados disponibles en todo el país (Colombia)")
    st.markdown(
        """
            2.2 Ahora, se quiere descubrir los mercados con la mayor cantidad de productos registrados a nivel nacional
            - **Pregunta:** ¿Cuáles son los mercados que presentan una mayor cantidad de productos registrados en el 2026?
        """
    )
    mercados_2026_prod = (
        df.loc[df["fecha"].dt.year == 2026]
        .groupby('mercado')['producto'] \
        .nunique()\
        .sort_values(ascending=False)\
        .reset_index(name='Cantidad_Productos')
    )
    st.write(mercados_2026_prod)
    st.write("Con esto se puede decir que los mercados (2026) que más productos comercializan o registran es la central mayorista ubicada en Medellín, porque lleva una gran ventaja sobre el segundo; también hay mercados que solo cuentan con un producto. Es preocupante para el seguimiento del modelo si eso se sigue presentando durante los años pasados")
    st.markdown(
        """
            2.3 El siguiente paso puede ser buscar el producto que más se comercializa por cada mercado
            - **Pregunta:** ¿Cuál es el producto que más se comercializa o que más registros presenta en la historia?
        """
    )
    productoFavoritoMercado = (
        df.groupby(['mercado', 'producto'])
        .size()
        .reset_index(name='Cantidad')
        .sort_values(['mercado', 'Cantidad'], ascending=[True, False])
        .groupby('mercado')
        .head(1)
    )
    st.write(productoFavoritoMercado)

    st.write("Se encuentra con mercados que solo presentan 1 producto comercializado, algo que puede afectar el modelo si ese producto presenta pocos registros. Por ende, se tendría que implementar una regla para evitar que se presenten esos mercados con 1 solo producto y pocos mercados")

    st.markdown(
        """
            2.4 Se podría mostrar en un mercado específico cuál es el producto que más se comercializa o que más registros presenta en la historia
            - **Pregunta:** Dado un mercado "x", ¿cuál es el producto que más se comercializa?
        """
    )
    mercados_2026 = (
        df[df["fecha"].dt.year == 2026]["mercado"]
        .dropna()
        .sort_values()
        .unique()
    )
    mercadoSeleccionado = st.selectbox(
        "Seleccione un mercado",
        mercados_2026
    )

    starMercadoSelectButton = st.button("Iniciar análisis del mercado seleccionado")

    if starMercadoSelectButton:
        st.write("Mercado seleccionado:", mercadoSeleccionado)
        ProductoEspecificoMercado = (
            df[df['mercado'] == mercadoSeleccionado]['producto']
            .value_counts()
        )
        st.write(ProductoEspecificoMercado.head(5))
        st.markdown(
            f"""
                Esta es una función vital para el algoritmo, pues se necesita fijar el mercado para realizar consultas de una manera mucho más específica. Aquí encontramos que los productos que tienen más registros se presentan en el mercado seleccionado por el usuario, que es: **{mercadoSeleccionado}**.
            """
        )
    st.write("**Conclusión** Los mercados deben ser fundamentales para el modelo, ya que son el último paso antes de realizar la serie preparativa para el modelo. Debemos identificar los modelos que no tengan la suficiente cantidad de datos o mercados, o encontrar la forma de explicarle al usuario por qué no se puede realizar la serie y, por ende, el funcionamiento del modelo, y presentar alternativas")

    st.divider()

    st.header("ANÁLISIS DE LAS VARIABLES DE 'MUNICIPIO' Y 'DEPARTAMENTO'")

    st.write("Estas dos variables se estudiarán en conjunto; al representar una variable de territorio nos ayudará a encontrar similitudes, cantidad de mercados por ciudad, la ciudad con más productos, sus productos únicos, etc.")

    st.subheader("Estudio para la variable 'Municipio'")
    st.markdown(
        """
            3.1 Identificar las ciudades registradas y la cantidad de mercados que tienen 
            - **Pregunta:** ¿Cuáles son las ciudades que tienen mercados y cuántas tienen?
        """
    )
    mercados_por_ciudad = (
        df.groupby("municipio")["mercado"]
        .nunique()
        .reset_index(name="Cantidad de mercados")
        .sort_values("Cantidad de mercados", ascending=False)
    )
    st.write(mercados_por_ciudad)
    st.write("Se puede observar que se registra un total de 78 ciudades con al menos un mercado registrado; Bogotá es la que más mercados presenta, con la cantidad de 7")

    st.markdown(
        """
            3.2 Una parte fundamental para el algoritmo del modelo es ser capaz de mostrar los mercados por cada ciudad.
            - **Pregunta:** Según una ciudad "X", ¿cuántos mercados presenta en el año actual?
        """
    )

    municipio_2026 = (
        df[df["fecha"].dt.year == 2026]["municipio"]
        .dropna()
        .sort_values()
        .unique()
    )
    municipioSeleccionado = st.selectbox(
        "Seleccione un municipio",
        municipio_2026
    )

    starMunicipioSelectButton = st.button("Iniciar análisis del municipio seleccionado")

    if starMunicipioSelectButton:
        st.write("Municipio seleccionado:", municipioSeleccionado)
        mercado_en_ciudad = (
            df[
                (df["municipio"] == municipioSeleccionado) &
                (df["fecha"].dt.year == 2026)
            ]["mercado"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        st.write(mercado_en_ciudad)
        st.write("Se logró mostrar cuántos mercados tiene disponible; es un buen comienzo para poder mostrarle a los usuarios los mercados disponibles según la ciudad")
        st.markdown(
            """
                3.3 Se puede filtrar también los productos que solo son únicos en la ciudad
                - **Pregunta:** ¿Qué productos se presentan solo en la ciudad "X" solicitada?
            """
        )
        st.write("Se continúa trabajando con el municipio previamente seleccionado")
        st.write("Municipio seleccionado:", municipioSeleccionado)

        productos_ciudad = (
            df.loc[
                (df["municipio"] == municipioSeleccionado) &
                (df["fecha"].dt.year == 2026),
                "producto"
            ]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        st.write(f"En el municipio de {municipioSeleccionado} se comercializan {len(productos_ciudad)} productos únicos, los cuales son: ")
        st.write(productos_ciudad)

        st.write(f"Se obtiene el total de {len(productos_ciudad)} productos comercializados en la ciudad de {municipioSeleccionado}, eso nos ayuda para que el usuario sepa escoger los productos. Igualmente, el algoritmo final los presentará para evitar las confusiones con productos que no están disponibles")

    st.write("3.3 Para descifrar que los productos no son los mismos en cada mercado, desarrollaremos una comparativa entre ambos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Municipio A")
        municipioSeleccionadoA = st.selectbox(
            "Seleccione un municipio A (para el estudio)",
            municipio_2026
        )
    with col2:
        st.subheader("Municipio B")
        municipioSeleccionadoB = st.selectbox(
            "Seleccione un municipio B (para el estudio)",
            municipio_2026
        )

    starMunicipioSelectComparationButton = st.button("Iniciar análisis de comparación de municipios")
    if starMunicipioSelectComparationButton:
        st.write("Municipio seleccionado A:", municipioSeleccionadoA)
        st.write("Municipio seleccionado B:", municipioSeleccionadoB)
        comparacionCiudades = (
            df[
                (df["fecha"].dt.year == 2026) &
                (df["municipio"].isin([municipioSeleccionadoA, municipioSeleccionadoB]))
            ]
            .groupby('municipio')['producto']
            .nunique()
            .reset_index(name='Cantidad_Productos')
        )
        st.write(comparacionCiudades)
        st.write("Con esta comparativa se puede concluir que no todos los mercados tienen los mismos productos; por ende, es esencial presentar en el algoritmo qué productos están presentes en cada mercado de cada ciudad")

    st.write("3.4 El paso siguiente es realizar la comparativa con nuestra variable objetivo precio, por ende se pueden presentar los precios de los productos de una ciudad 'X' en el 2026")

    municipioPorProducto = st.selectbox(
        "Seleccione un municipio para observar sus productos (para el estudio)",
        municipio_2026
    )

    startMunicipioPorProductoButton = st.button("Iniciar análisis de comparación de precios por producto")

    if startMunicipioPorProductoButton:
        st.write("Municipio seleccionado:", municipioPorProducto)
        precios_productos = (
            df[
                (df["municipio"] == municipioPorProducto) &
                (df["fecha"].dt.year == 2026)
            ]
            .groupby("producto")
            .agg(
                Precio_Promedio=("precio_promedio_kg", "mean")
            )
            .round(2)
            .reset_index()
            .sort_values("producto")
        )
        st.write(precios_productos)
        st.write("Esta toma representa todos los productos y su precio promedio registrados en el 2026; el proceso se puede registrar para cualquier municipio de Colombia")

    st.divider()

    st.subheader("Estudio de la variable 'departamento'")

    st.write("3.5 Para departamento la tarea es muy sencilla: para el correcto funcionamiento debe ser capaz de mostrar las ciudades que tienen mercados disponibles")

    departamento_2026 = (
        df[df["fecha"].dt.year == 2026]["departamento"]
        .dropna()
        .sort_values()
        .unique()
    )
    departamentoSeleccionado = st.selectbox(
        "Seleccione un departamento",
        departamento_2026
    )

    departamentoSelectBtn = st.button("Iniciar análisis del departamento seleccionado")

    if departamentoSelectBtn:
        st.write("Departamento seleccionado: ", departamentoSeleccionado)
        ciudades_departamento = (
            df.loc[
                (df["departamento"] == departamentoSeleccionado) &
                (df["fecha"].dt.year == 2026),
                "municipio"
            ]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        st.write(f"El departamento de {departamentoSeleccionado} registra {len(ciudades_departamento)} municipios.\n")
        st.write(ciudades_departamento)
        st.write("La tarea se desarrolla con éxito; se pueden mostrar las ciudades que tienen mercados disponibles en el 2026. Eso se hace con la necesidad de que el usuario no busque mercados que anteriormente estaban en vigencia o que no presentan registros en la fecha actual.")

    
    st.divider()

    st.header("Estudio de la variable de precio en relación con el tiempo y el producto")

    st.markdown(
        """
        - 4.1 Como primera actividad se va a consultar los productos desde el más caro hasta el menor en el 2026
        - **Pregunta:** ¿Como estas ordenados los productos en terminos de precio en el 2026?
        """
    )

    productos_caros_2026 = (
        df[df["fecha"].dt.year == 2026]
        .groupby("producto")["precio_promedio_kg"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="Precio_Promedio_kg")
    )

    st.write(productos_caros_2026)

    st.write("Como se puede observar, los productos están organizados desde el más caro hasta el más barato. El más caro sigue siendo el café instantáneo, que controla todos los picos en términos de precios altos")


    st.markdown(
        """
        - 4.2 El siguiente paso es evaluar el comportamiento de un producto 'X' a través del tiempo
        - **Pregunta:** ¿Cómo es el comportamiento de un producto 'X' a través del tiempo?
        """
    )


    producto_2026 = (
        df[df["fecha"].dt.year == 2026]["producto"]
        .dropna()
        .sort_values()
        .unique()
    )

    productoSeleccionado = st.selectbox(
        "Seleccione un producto",
        producto_2026
    )


    starProductoBtn = st.button("Iniciar análisis del producto seleccionado") 


    if starProductoBtn:
        st.write("Producto seleccionado:", productoSeleccionado)
        comportamiento_producto = (
            df[df["producto"].str.lower() == productoSeleccionado.lower()]
            .groupby(df["fecha"].dt.year)
            .agg(
                Precio_Promedio=("precio_promedio_kg", "mean"),
                Precio_Minimo=("precio_promedio_kg", "min"),
                Precio_Maximo=("precio_promedio_kg", "max"),
                Registros=("precio_promedio_kg", "count")
            )
            .round(2)
            .reset_index()
            .rename(columns={"fecha": "Año"})
        )
        st.write(comportamiento_producto)
        st.write("Aquí podemos presenciar si el producto cumple con la regla de estar presente en el año actual, también cuáles registros presentó, su pico máximo y mínimo, y la cantidad de registros que presenta")

        st.subheader("Grafica")

        comportamiento_producto = (
            df[df["producto"].str.lower() == productoSeleccionado.lower()]
            .groupby(df["fecha"].dt.year)
            .agg(
                Precio_Promedio=("precio_promedio_kg", "mean")
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            comportamiento_producto["fecha"],
            comportamiento_producto["Precio_Promedio"],
            marker="o",
            linewidth=2
        )

        ax.set_title(f"Comportamiento del precio promedio de '{productoSeleccionado}'")
        ax.set_xlabel("Año")
        ax.set_ylabel("Precio promedio por kilogramo")

        ax.set_xticks(comportamiento_producto["fecha"])
        ax.grid(True)

        st.pyplot(fig)

        st.text("Con la gráfica se puede determinar el comportamiento de los productos a través del tiempo presente en el dataset de estudio; con eso se puede determinar la viabilidad del producto y la estacionalidad del mismo")


    st.divider()
    st.header("Análisis de correlación")

    st.text("Identificar las columnas que tienen mayor impacto sobre la variable objetivo con el fin de desarrollar un modelo más optimizado y centrado para el estudio")

    st.divider()

    st.text("Por recomendaciones de la IA (ChatGPT) se recomendó usar un análisis Eta Squared (η²), porque no manejo específicamente variables numéricas; la mayoría son categóricas.")

    df["año"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    st.markdown(
        """
            **Mi recomendación**:

            Dado el tipo de datos que tienes (SIPSA) y que tu variable objetivo es Precio promedio por kilogramo*, en lugar de una matriz de correlación te recomendaría construir una tabla de importancia de variables categóricas respecto al precio (por ejemplo, usando ANOVA o η²). Ese análisis será mucho más informativo para justificar qué variables afectan el precio y cuáles conviene incluir en el modelo.
        """
    )

    target = "precio_promedio_kg"
    variables_categoricas = [
        "grupo",
        "producto",
        "departamento",
        "municipio",
        "mercado",
        "año",
        "mes"
    ]

    def eta_squared(df, categoria, objetivo):
        datos = df[[categoria, objetivo]].dropna()
        media_total = datos[objetivo].mean()
        # Suma de cuadrados entre grupos
        ss_between = (
            datos.groupby(categoria)[objetivo]
                .apply(lambda x: len(x) * (x.mean() - media_total) ** 2)
                .sum()
        )
        # Suma de cuadrados total
        ss_total = ((datos[objetivo] - media_total) ** 2).sum()
        return ss_between / ss_total if ss_total != 0 else np.nan

    resultado = pd.DataFrame({
        "Variable": variables_categoricas,
        "Eta²": [eta_squared(df, v, target) for v in variables_categoricas]
    })

    resultado = resultado.sort_values("Eta²", ascending=False)
    st.write(resultado)

    st.text("Presenciamos las variables que mejor se conectan con nuestra variable objetivo, pero entendiendo que el problema es de serie de tiempo, no pondremos atención a las estadísticas de año y mes, además de ser variables que fueron separadas")