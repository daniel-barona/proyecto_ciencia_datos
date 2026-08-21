import gc
import os
import sys

import streamlit as st

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_RAIZ, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from backend import modelo as mo


@st.cache_data(show_spinner=False, ttl=3600)
def _cargar_datos():
    return mo.cargar_trusted()


@st.cache_data(ttl=1800, show_spinner="Modelando... esto puede tardar ~30s")
def _ejecutar_pipeline(_df_hash: str, producto: str, mercado, _v: int = 4):
    df = _cargar_datos()
    res = mo.ejecutar_pipeline(df, producto, mercado)
    return dict(
        resumen=res["resumen"],
        forecast=res["forecast"],
    )


def render():
    st.header("Prediccion")
    st.write(
        "Selecciona la ubicacion y el producto para ejecutar el modelo SARIMA/ARIMA "
        "sobre la serie historica del SIPSA y obtener el pronostico de precios."
    )

    try:
        df = _cargar_datos()
    except FileNotFoundError as e:
        st.warning(
            "No se encontro la capa **trusted** de datos. Ejecuta primero el "
            "backend local para generar `SIPSA_2013_2026_trusted.xlsx`.",
            icon="⚠️",
        )
        st.caption(str(e))
        return
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}", icon="🚫")
        return

    col1, col2 = st.columns(2)

    with col1:
        departamento = st.selectbox(
            "Departamento",
            mo.opciones_departamentos(df),
            help="Elige 'colombia' para un analisis nacional.",
            key="pred_departamento",
        )

    municipio = None
    mercado = None

    if mo.requiere_municipio(departamento):
        with col2:
            municipios = mo.opciones_municipios(df, departamento)
            municipio = st.selectbox("Municipio", municipios, key="pred_municipio")

    if departamento.lower() != "colombia":
        mercados = mo.opciones_mercados(df, departamento, municipio)
        with (col1 if not mo.requiere_municipio(departamento) else col2):
            mercado = st.selectbox("Mercado", mercados, key="pred_mercado")
    else:
        st.info("Modo nacional: el producto se promediara entre todos los mercados del pais.", icon="🌎")

    productos = mo.opciones_productos(df, mercado)
    producto = st.selectbox("Producto", productos, key="pred_producto")

    st.divider()

    df_hash = f"{df.shape[0]}_{df['fecha'].max().timestamp()}"

    if st.button("Generar prediccion", type="primary"):
        with st.spinner("Ejecutando el modelo... esto puede tardar un momento."):
            try:
                y_temp = mo.preparar_serie(df, producto, mercado)
                n_obs = len(y_temp)
                if n_obs < 24:
                    st.error(
                        f"La serie solo tiene **{n_obs} observaciones** "
                        f"de {y_temp.index[0]:%Y-%m} a {y_temp.index[-1]:%Y-%m}. "
                        f"Se necesitan minimo 24 meses para modelar.",
                        icon="⚠️",
                    )
                    return
                resultado = _ejecutar_pipeline(df_hash, producto, mercado, 4)
                st.session_state["pred_resultado"] = resultado
            except Exception as e:
                st.error(f"El modelo no pudo completarse: {e}", icon="🚫")
                return
            finally:
                gc.collect()

    resultado = st.session_state.get("pred_resultado")
    if not resultado:
        st.caption("Configura los parametros y presiona **Generar prediccion**.")
        return

    resumen = resultado["resumen"]
    forecast = resultado["forecast"]

    st.subheader("Resultado del pronostico")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modelo ganador", resumen["ganador"])
    m2.metric("Accuracy (test)", f"{resumen['accuracy']:.1f}%")
    m3.metric(
        f"Ultimo obs. ({resumen['ultima_fecha']})",
        f"${resumen['ultimo_observado']:,.0f}",
    )
    delta = resumen["pred_prox_mes"] - resumen["ultimo_observado"]
    m4.metric(
        f"Pronostico {resumen['pred_prox_mes_fecha']}",
        f"${resumen['pred_prox_mes']:,.0f}",
        delta=f"{delta:,.0f}",
    )

    st.divider()

    tabla_usuario = forecast[["Pronostico", "Cambio_$", "Cambio_%"]].copy()
    tabla_usuario["Pronostico"] = tabla_usuario["Pronostico"].apply(lambda x: f"${x:,.0f}")
    tabla_usuario["Cambio_$"] = tabla_usuario["Cambio_$"].apply(lambda x: f"{x:+,.0f}")
    tabla_usuario["Cambio_%"] = tabla_usuario["Cambio_%"].apply(lambda x: f"{x:+.2f}%")
    tabla_usuario.index = tabla_usuario.index.strftime("%Y-%m")
    tabla_usuario.index.name = "Mes"
    tabla_usuario = tabla_usuario.rename(columns={
        "Pronostico": "Pronostico ($/kg)",
        "Cambio_$": "Cambio $ vs mes anterior",
        "Cambio_%": "Cambio % vs mes anterior",
    })

    st.caption(f"Pronostico de precios - proximos {mo.H_FUTURO} meses")
    st.dataframe(tabla_usuario, use_container_width=True)
