# Web App — Modelo de Predicción SIPSA

Aplicación web construida con **Streamlit** para analizar y predecir los precios de productos agropecuarios en los mercados mayoristas de Colombia, a partir de los datos del **SIPSA (Sistema de Información de Precios y Abastecimiento del Sector Agropecuario)** del DANE.

> Trabajo para el diplomado de ciencia de datos — Daniel Andres Barona Sandoval

---

## Tabla de contenido

- [Descripción general](#descripción-general)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Secciones de la aplicación](#secciones-de-la-aplicación)
  - [Introducción](#introducción)
  - [EDA](#eda)
  - [Instrucciones](#instrucciones)
  - [Predicción](#predicción)
- [Backend](#backend)
  - [Pipeline de datos](#pipeline-de-datos)
  - [Motor de predicción (`modelo.py`)](#motor-de-predicción-modelopy)
- [Modelos utilizados](#modelos-utilizados)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Datos](#datos)
- [Notas técnicas](#notas-técnicas)

---

## Descripción general

La aplicación tiene como objetivo **predecir el precio futuro** (a 12 meses) de cualquier producto agropecuario en un mercado mayorista específico de Colombia, facilitando la toma de decisiones de comerciantes y consumidores de los mercados mayoristas.

- **Fuente de datos:** SIPSA — DANE ([boletín mensual mayorista](https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa/mayoristas-boletin-mensual-1))
- **Tipo de problema:** Series de tiempo
- **Variable objetivo:** `precio_promedio_kg` (precio promedio por kilogramo)
- **Horizonte de pronóstico:** 12 meses

## Estructura del proyecto

```
webapp/
├── app.py                  # Punto de entrada de Streamlit (navegación y estilos)
├── sections/               # Vistas de la aplicación
│   ├── introduccion.py     # Sección "Introducción"
│   ├── instrucciones.py    # Sección "Instrucciones"
│   ├── eda.py              # Sección "EDA" (análisis exploratorio)
│   └── prediccion.py       # Sección "Predicción"
├── backend/                # Lógica de datos y modelado
│   ├── config.py           # Rutas, URLs del DANE y mapeo de archivos
│   ├── descargas.py        # Paso 00 — descarga del Excel actual desde el DANE
│   ├── consolidar.py       # Paso 01 — unifica hojas y consolida 2013–2026
│   ├── limpieza.py         # Paso 02 — genera el dataset final (trusted)
│   ├── pipeline.py         # Orquestador CLI del pipeline
│   ├── modelo.py           # Motor de predicción SARIMA/ARIMA (8 fases)
│   ├── utils.py            # Utilidades (logs, normalización, Excel, etc.)
│   └── data/               # Capas de datos locales (Descargas → raw → landing → trusted)
└── README.md
```

## Secciones de la aplicación

Al ejecutar `app.py` se muestra una barra lateral de navegación con cuatro secciones (Introducción, EDA, Instrucciones y Predicción). La sección activa se guarda en `st.session_state.pagina_activa`.

### Introducción

Página de bienvenida que presenta el objetivo del proyecto, enlaza a la fuente oficial de datos del DANE, indica el tipo de problema (series de tiempo), el público objetivo (comerciantes y consumidores de mercados mayoristas) y resume los modelos predictivos utilizados (ARIMA, SARIMA y naive estacional).

### EDA

Análisis exploratorio interactivo sobre el dataset consolidado `backend/data/trusted/SIPSA_2013_2026_trusted.parquet`. Incluye:

- Vista general del dataset (`describe`, `info`) y descripción de la variable objetivo.
- Estudio de la variable **Grupo**: cantidad de productos por grupo, producto más comercializado por grupo en 2026 y producto más costoso (promedio) por grupo, con gráfica de barras.
- Estudio de la variable **Mercado**: mercados disponibles en 2026 (~138 a nivel nacional), mercados con mayor variedad de productos, producto más comercializado por mercado histórico y selector interactivo para explorar un mercado específico.
- Estudio de las variables **Municipio / Departamento**: ciudades con mercados, mercados disponibles por ciudad, productos únicos por ciudad, comparativa entre dos municipios y precios promedio por producto en una ciudad.
- Estudio del **precio en el tiempo**: ranking de productos por precio en 2026 y comportamiento anual (promedio, mínimo, máximo, registros) de un producto seleccionado, con gráfica de línea.
- **Análisis de correlación** mediante **Eta Squared (η²)** entre variables categóricas (grupo, producto, departamento, municipio, mercado, año, mes) y el precio.

### Instrucciones

Guía paso a paso del flujo de uso del predictor:

1. Seleccionar el **departamento** (incluye Bogotá D.C. y la opción `colombia` para análisis nacional).
2. Si aplica (fuera de Bogotá y Colombia), seleccionar el **municipio** con centros mayoristas activos.
3. Se muestran los **productos disponibles** en ese mercado.
4. Seleccionar el **producto** a investigar.
5. Internamente se ejecuta el procesamiento fase por fase.
6. Se presenta el **pronóstico a 12 meses**, en precio y en variaciones ($ y %) mes a mes.

### Predicción

Sección principal del modelo. Flujo:

1. Selectores en cascada: **Departamento → Municipio → Mercado → Producto** (las opciones dependen de la selección anterior; solo se listan registros vigentes del último año disponible).
2. Modo **nacional** (`colombia`): el producto se promedia entre todos los mercados del país.
3. Validación mínima: la serie debe tener al menos **24 observaciones mensuales** para poder modelar.
4. Al presionar **Generar prediccion** se ejecuta el pipeline completo de 8 fases del motor (`modelo.py`).
5. Resultados mostrados:
   - Métricas resumen: modelo ganador, Accuracy (%), última observación y pronóstico del próximo mes con delta.
   - Tabla del pronóstico de 12 meses: `Pronostico ($)`, `Cambio_$` y `Cambio_%` vs mes anterior.

Los resultados se cachean con `st.cache_data` (TTL de 30–60 min) para agilizar consultas repetidas.

## Backend

### Pipeline de datos

Todo el procesamiento es local (carpeta `backend/data/`), organizado en capas estilo medallion. Se orquesta con `backend/pipeline.py`:

```bash
python -m backend.pipeline                # pipeline completo
python -m backend.pipeline --paso 00      # solo descargas
python -m backend.pipeline --paso 01      # solo consolidar
python -m backend.pipeline --paso 02      # solo limpieza
python -m backend.pipeline --sin-descarga # usa lo ya existente en data/Descargas
```

| Paso | Módulo | Entrada | Salida |
|------|--------|---------|--------|
| 00 | `descargas.py` | Excel SIPSA del año en curso (URL oficial del DANE) | `data/Descargas/` → copia a `data/raw/` |
| 01 | `consolidar.py` | Excels de `data/raw/` | `data/landing/*_unificado.xlsx` + `SIPSA_2013_2026_FINAL_consolidado.xlsx` |
| 02 | `limpieza.py` | Consolidado de `data/landing/` | `data/trusted/SIPSA_2013_2026_trusted.{xlsx,csv,parquet}` |

Durante la consolidación/limpieza se normalizan textos (minúsculas, sin tildes), se renombran columnas al esquema final (`fecha`, `grupo`, `producto`, `mercado`, `departamento`, `municipio`, `precio_promedio_kg`, ...), se eliminan duplicados, fechas/nulos inválidos y precios ≤ 0. La raíz de datos puede redirigirse con la variable de entorno `SIPSA_DATA_DIR`.

### Motor de predicción (`modelo.py`)

Implementa las mismas **8 fases del notebook** adaptadas para la web:

1. **Fase 1 — Creación de la serie:** filtra por producto y mercado, agrega por mes y completa frecuencias (`MS`, interpolación lineal).
2. **Fase 2 — Verificación y split Train/Test:** evalúa nivel de registros, fuerza estacional/tendencia (STL robusto) y divide la serie (test ≈ 12 meses).
3. **Fase 3 — Exploración e identificación:** descomposición estacional, detección de outliers (z-score robusto sobre diferencias), pruebas ADF/KPSS, estimación de `d` y `D` (con `pmdarima` si está disponible) y gráficas ACF/PACF.
4. **Fase 4 — Estimación y diagnóstico:** ajuste SARIMAX (transformación log), selección de órdenes (`auto_arima` o búsqueda en grilla fallback), verificación de estabilidad (raíces AR) y diagnóstico de residuos (Ljung-Box, Jarque-Bera, histograma, Q-Q).
5. **Fase 5 — Backtest rolling-origin en test:** compara candidatos (SARIMA auto, SARIMA(0,1,1)(0,1,1)[12] + deriva, ARIMA auto) contra referencias (naive estacional, deriva 12m) con métricas MAE, RMSE, MAPE, sMAPE, MASE y Accuracy.
6. **Fase 6 — Selección del ganador:** descarta pronósticos planos/degenerados, aplica tolerancia de empates y prioriza modelos con forma estacional real; veredicto según MASE (<0.8 aporta valor claro vs naive).
7. **Fase 7 — Reentrenamiento** con la serie completa usando la especificación ganadora.
8. **Fase 8 — Pronóstico a futuro** (12 meses) con intervalos de confianza al 95%, cambios $/% mes a mes y validación de rango plausible.

Constantes clave: periodo estacional `m=12`, horizonte `H_FUTURO=12`, transformación log activada, optimizador `lbfgs` con reintento `powell`.

## Modelos utilizados

- **ARIMA:** modela la dependencia entre observaciones actuales y valores/errores pasados; para series sin componente estacional explícito.
- **SARIMA:** extensión de ARIMA con componentes estacionales (periodicidad mensual `m=12` para capturar patrones anuales).
- **Configuraciones ARIMA/SARIMA:** distintas combinaciones de `(p,d,q)` y `(P,D,Q)`, con y sin deriva, automáticas y manuales.
- **Naive estacional:** línea base que replica el valor del mismo mes de la temporada anterior; permite verificar si ARIMA/SARIMA realmente aportan valor.

## Instalación y ejecución

Requisitos principales (ver `requirements.txt` en la raíz del repo): `streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `pmdarima`, `matplotlib`, `openpyxl`, `pyarrow`, `requests`, `joblib`.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Regenerar la capa trusted con datos frescos del DANE
cd webapp
python -m backend.pipeline

# 3. Levantar la aplicación
streamlit run app.py
```

> La sección Predicción requiere que exista `backend/data/trusted/SIPSA_2013_2026_trusted.parquet`. Si no existe, la app lo advierte y hay que ejecutar primero el pipeline.

## Datos

Capas dentro de `backend/data/`:

```
data/
├── Descargas/           # Excel descargados del DANE (paso 00)
├── raw/                 # Copia de trabajo de las descargas
├── landing/             # Hojas unificadas por archivo + consolidado histórico
├── trusted/             # Dataset limpio final (xlsx/csv/parquet) usado por la app
└── mercados_faltantes/  # Catálogos manuales opcionales (renombres, mercados extintos)
```

Cobertura histórica: **2013 – 2026** (se actualiza descargando el anexo SIPSA del año en curso).

## Notas técnicas

- El render de cada sección se hace mediante la función `render()` de cada módulo de `sections/`.
- `prediccion.py` inserta `backend/` en `sys.path` para importar el motor sin instalar paquete.
- Las figuras de Matplotlib se serializan a PNG en memoria (`Fase.figuras`) para su posible visualización/descarga.
- La recolección de basura (`gc.collect()`) se usa tras los entrenamientos pesados para liberar memoria.
- Los warnings de statsmodels se suprimen durante el ajuste para mantener limpia la interfaz.
