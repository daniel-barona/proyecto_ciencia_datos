# Proyecto de Análisis de Datos

## Descripción
Este proyecto está diseñado para procesar, analizar y modelar los datos mensuales de precios mayoristas del **SIPSA (Sistema de Información de Precios y Abastecimiento del Sector Agropecuario)** del DANE, con cobertura histórica **2013 – 2026**. El flujo de trabajo incluye la descarga de datos, consolidación, limpieza, análisis exploratorio de datos (EDA) y modelado predictivo (ARIMA/SARIMA).

## Estructura del Proyecto

```
.
├── .github                     # Automatizaciones de GitHub
│   └── workflows
│       └── pipeline_sipsa.yml  # GitHub Action: pipeline diario (descarga + actualización del dataset)
├── src                         # Código fuente (cuadernos Jupyter)
├── data                        # Directorio para todos los datos
│   ├── Descargas               # Excel descargados directamente del DANE
│   ├── raw                     # Datos brutos organizados por año y mes
│   ├── landing                 # Datos consolidados sin procesar
│   ├── trusted                 # Datos limpios listos para análisis y modelado
│   ├── surface                 # Datos organizados para compartir con usuarios
│   └── mercados_faltantes      # Catálogos manuales (renombres de mercados, extintos)
├── api                         # API para servir el modelo 
├── reportes                    # Informes generados y visualizaciones
├── docs                        # Documentación adicional del proyecto
├── webapp                      # Aplicación web Streamlit (frontend + backend)
│   └── backend/data            # Capas de datos locales que usa la app
├── ejecutar_pipeline.bat       # Script para ejecutar el pipeline en Windows
└── requirements.txt            # Dependencias del proyecto
```

## Flujo de trabajo

El proyecto sigue un flujo de trabajo secuencial implementado en varios cuadernos Jupyter:

1. **Descarga de datos** (`00_descargas.ipynb`): Proceso para obtener los datos desde la fuente original.
2. **Consolidación** (`01_consolidar.ipynb`): Integración de los archivos mensuales en un solo conjunto de datos.
3. **Limpieza de datos** (`02_limpieza.ipynb`): Preprocesamiento para manejar valores faltantes, outliers y transformaciones.
4. **Análisis Exploratorio de Datos** (`03_EDA.ipynb`): Visualizaciones y estadísticas descriptivas.
5. **Modelado** (`04_modelo.ipynb`): Desarrollo e implementación de modelos predictivos.

Además, el pipeline puede ejecutarse de forma automática mediante **GitHub Actions** (ver [Carpeta `.github`](#carpeta-github)).

## Datos

Los datos están organizados de la siguiente manera:

- **raw**: Datos crudos (tal cual se descargan), archivos de ejemplo en formato Excel
- **landing**: Datos consolidados de todas las fuentes, sin más procesamiento
- **trusted**: Datos limpios, normalizados, estandarizados, etc. listos para análisis

## Requisitos

Para ejecutar este proyecto tienes dos opciones:

### Opción 1: Google Colab (recomendado para análisis)

Los cuadernos Jupyter de `src/` pueden ejecutarse directamente en [Google Colab](https://colab.research.google.com/), sin instalar nada en tu equipo:

1. Abre [Google Colab](https://colab.research.google.com/)
2. Selecciona **Archivo → Subir cuaderno** y carga el `.ipynb` que desees ejecutar (o ábrelo directo desde GitHub: **Archivo → Abrir cuaderno → GitHub**).
3. Ejecuta las celdas en orden siguiendo el flujo de trabajo numerado.

> Los datasets pesados se generan en el entorno de Colab; descárgalos o guárdalos en Drive si quieres conservarlos.

### Opción 2: Desarrollo local (aplicación web / API)

Para el desarrollo de la app (`webapp/`), la API (`api/`) o ejecutar el pipeline localmente, consulta e instala las dependencias del archivo [`requirements.txt`](requirements.txt):

```
python>=3.11
```

```bash
pip install -r requirements.txt
```

## Instalación

1. Crear un **fork** de este repositorio, cambia el nombre según el tema de tu proyecto:

    Esto crea una copia en tu cuenta de GitHub: 

    `https://github.com/daniel-barona/proyecto_ciencia_datos`

2. Clona el repositorio de tu proyecto (el **fork** que hicieron):

    ```bash
    git clone https://github.com/usuario/nombre-del-proyecto.git
    cd nombre-del-proyecto
    ```

2. Crea un entorno virtual e instala las dependencias:

    ```bash
    python -m venv venv
    pip install -r requirements.txt
    ```

## Uso

Ejecuta los cuadernos Jupyter en orden:

```bash
jupyter notebook src/00_descargas.ipynb
```

Continúa con los siguientes cuadernos siguiendo el flujo de trabajo numerado.

## Carpeta `.github`

El directorio [`.github/`](.github) contiene la automatización del proyecto con **GitHub Actions**:

- **`workflows/pipeline_sipsa.yml`** — Pipeline SIPSA diario:
  - Se ejecuta automáticamente todos los días a las **4:40 PM hora de Colombia** (21:40 UTC) y también permite ejecución manual (`workflow_dispatch`).
  - Instala las dependencias de `requirements.txt`, ejecuta `python -m webapp.backend.pipeline` (descarga del anexo actual del DANE, consolidación y limpieza).
  - Hace commit automático del dataset actualizado en `webapp/backend/data/**`.

Esto garantiza que la aplicación web siempre trabaje con los datos más recientes sin intervención manual.

## API

La API proporciona acceso a las predicciones del modelo. Consulta la documentación en el directorio `api` para más detalles sobre los endpoints disponibles y su uso.

## Aplicación Web

La aplicación web permite visualizar los resultados y las predicciones del modelo de manera interactiva. Consulta el README en el directorio `webapp` para instrucciones de instalación y uso.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## Contacto

[Daniel Andres Barona Sandoval] - [daniel.baronasa.com]

Enlace del proyecto: [https://github.com/daniel-barona/proyecto_ciencia_datos]

