import os
import sys

import streamlit as st

# Permite importar el paquete backend/ desde la raiz del proyecto
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_RAIZ, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
    
from backend import modelo as mo


@st.cache_data(show_spinner=False)
def _cargar_datos():
    """Carga el dataset trusted (cacheado)."""
    return mo.cargar_trusted()


def _render_fase(fase):
    """Dibuja una fase: titulo, logs, tablas y figuras."""
    st.markdown(f"#### {fase.id} — {fase.titulo}")
    if fase.logs:
        st.code(fase.texto, language="text")
    for titulo, df in fase.tablas:
        st.caption(titulo)
        st.dataframe(df, use_container_width=True)
    for fig in fase.figuras:
        st.pyplot(fig, use_container_width=True)
    st.divider()


def render():
    st.header("🔮 Predicción")
    st.write(
        "Selecciona la ubicación y el producto para ejecutar el modelo SARIMA/ARIMA "
        "sobre la serie histórica del SIPSA y obtener el pronóstico de precios."
    )

    # 1) Cargar datos trusted

    try:
        df = _cargar_datos()
    except FileNotFoundError as e:
        st.warning(
            "No se encontró la capa **trusted** de datos. Ejecuta primero el "
            "backend local (descarga → consolidación → limpieza) para generar "
            "`SIPSA_2013_2026_trusted.xlsx`.",
            icon="⚠️",
        )
        st.caption(str(e))
        return
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}", icon="🚫")
        return

    # 2) Selectores en cascada
    
    col1, col2 = st.columns(2)

    with col1:
        departamento = st.selectbox(
            "Departamento",
            mo.opciones_departamentos(df),
            help="Elige 'colombia' para un análisis nacional (promedio de todos los mercados).",
            key="pred_departamento",
        )

    municipio = None
    mercado = None

    # Municipio (solo si el departamento lo requiere)
    if mo.requiere_municipio(departamento):
        with col2:
            municipios = mo.opciones_municipios(df, departamento)
            municipio = st.selectbox("Municipio", municipios, key="pred_municipio")

    # Mercado (no aplica en modo nacional 'colombia')
    if departamento.lower() != "colombia":
        mercados = mo.opciones_mercados(df, departamento, municipio)
        with (col1 if not mo.requiere_municipio(departamento) else col2):
            mercado = st.selectbox("Mercado", mercados, key="pred_mercado")
    else:
        st.info(
            "Modo nacional: el producto se promediará entre todos los mercados del país.",
            icon="🌎",
        )

    # Producto (depende del mercado; en nacional depende de todo el país)
    productos = mo.opciones_productos(df, mercado)
    producto = st.selectbox("Producto", productos, key="pred_producto")

    st.divider()

    # ------------------------------------------------------------------
    # 3) Ejecutar el pipeline
    # ------------------------------------------------------------------
    if st.button("Generar predicción", type="primary"):
        with st.spinner("Ejecutando el modelo (15 fases)... esto puede tardar un momento."):
            try:
                resultado = mo.ejecutar_pipeline(df, producto, mercado)
                st.session_state["pred_resultado"] = resultado
                st.session_state["pred_ver_todo"] = False
            except Exception as e:
                st.error(f"El modelo no pudo completarse: {e}", icon="🚫")
                return

    # ------------------------------------------------------------------
    # 4) Mostrar resultados
    # ------------------------------------------------------------------
    resultado = st.session_state.get("pred_resultado")
    if not resultado:
        st.caption("Configura los parámetros y presiona **Generar predicción**.")
        return

    resumen = resultado["resumen"]
    fases = resultado["fases"]

    # Tarjetas resumen (parte alta de la Fase 15)
    st.subheader("Resultado del pronóstico")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modelo ganador", resumen["ganador"])
    m2.metric("Accuracy (test)", f"{resumen['accuracy']:.1f}%")
    m3.metric(
        f"Último obs. ({resumen['ultima_fecha']})",
        f"${resumen['ultimo_observado']:,.0f}",
    )
    delta = resumen["pred_prox_mes"] - resumen["ultimo_observado"]
    m4.metric(
        f"Pronóstico {resumen['pred_prox_mes_fecha']}",
        f"${resumen['pred_prox_mes']:,.0f}",
        delta=f"{delta:,.0f}",
    )

    st.divider()

    # --- Fase 15 (siempre visible por defecto) ------------------------
    fase15 = fases[-1]
    _render_fase(fase15)

    # --- Boton para ver el resto de fases -----------------------------
    if "pred_ver_todo" not in st.session_state:
        st.session_state["pred_ver_todo"] = False

    if not st.session_state["pred_ver_todo"]:
        if st.button("🔎 Ver los otros pasos del proceso"):
            st.session_state["pred_ver_todo"] = True
            st.rerun()
    else:
        if st.button("🔼 Ocultar los pasos intermedios"):
            st.session_state["pred_ver_todo"] = False
            st.rerun()

        st.markdown("### Proceso completo del modelo (Fases 1 → 15)")
        for fase in fases:  # ya vienen en orden 1..15
            _render_fase(fase)
