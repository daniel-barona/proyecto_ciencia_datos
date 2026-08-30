import streamlit as st


from sections import introduccion, instrucciones, eda, prediccion

# ----------------------------------------------------------------------
# Configuracion general de la pagina
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Modelo de Predicción SIPSA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Estilos personalizados
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Paleta de la app */
        :root {
            --primary: #16a34a;
            --primary-dark: #15803d;
        }

        /* Titulo principal */
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: var(--primary-color);
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            font-size: 1.05rem;
            color: var(--text-color);
            margin-top: 0;
        }

        /* Encabezado del sidebar */
        .sidebar-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #14532d;
            margin-bottom: 0.25rem;
        }
        .sidebar-caption {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }

        /* Botones del menu */
        div[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            text-align: left;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            background-color: #ffffff;
            color: #14532d;
            font-weight: 600;
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.4rem;
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--primary);
            background-color: #f0fdf4;
            color: var(--primary-dark);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Estado de navegacion
# ----------------------------------------------------------------------
PAGINAS = {
    "Introducción": introduccion,
    "EDA": eda,
    "Instrucciones": instrucciones,
    "Predicción": prediccion,
}

ICONOS = {
    "Introducción": "🏠",
    "EDA": "📊",
    "Instrucciones": "📋",
    "Predicción": "🔮",
}

if "pagina_activa" not in st.session_state:
    st.session_state.pagina_activa = "Introducción"


# ----------------------------------------------------------------------
# Sidebar / Menu de navegacion
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📈 SIPSA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Menú de navegación</div>',
        unsafe_allow_html=True,
    )

    for nombre in PAGINAS:
        etiqueta = f"{ICONOS[nombre]}  {nombre}"
        if st.button(etiqueta, key=f"nav_{nombre}"):
            st.session_state.pagina_activa = nombre

    st.divider()
    st.caption("Modelo de Predicción SIPSA · 2026")

# ----------------------------------------------------------------------
# Encabezado principal
# ----------------------------------------------------------------------
st.markdown(
    '<div class="app-title">Modelo de Predicción SIPSA</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Trabajo para el diplomado de ciencia de datos '
    "- Daniel Andres Barona Sandoval</p>",
    unsafe_allow_html=True,
)

st.info(
            "Usa el menú lateral para navegar entre las diferentes secciones de la aplicación.",
            icon="💡",
        )
st.success(
            "Puedes cambiar el tema (claro u oscuro) desde el menú de la esquina "
            "superior derecha sobre los 3 puntos",
            icon="🎨",
        )

st.divider()
# ----------------------------------------------------------------------
# Render de la pagina seleccionada
# ----------------------------------------------------------------------
PAGINAS[st.session_state.pagina_activa].render()
