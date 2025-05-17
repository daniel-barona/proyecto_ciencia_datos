# Proyecto de Análisis de Datos

## Descripción

Este proyecto está diseñado para procesar, analizar y modelar datos mensuales recopilados desde noviembre de 2022 hasta marzo de 2024. El flujo de trabajo incluye la descarga de datos, consolidación, limpieza, análisis exploratorio de datos (EDA) y modelado predictivo.

## Estructura del Proyecto

```
.
├── data                  # Directorio para todos los datos
│   ├── raw               # Datos brutos/crudos tal cual vienen de la fuente
│   ├── landing           # Datos consolidados sin procesar
│   ├── trusted           # Datos limpios listos para análisis y modelado - para científicos
│   └── surface           # Directorio disponibles para los analistas
├── src                   # Código fuente (cuadernos Jupyter)
├── api                   # API para servir el modelo 
├── reportes              # Informes generados y visualizaciones
└── webapp                # Aplicación web para visualizar resultados
```

## Flujo de trabajo

El proyecto sigue un flujo de trabajo secuencial implementado en varios cuadernos Jupyter:

1. **Descarga de datos** (`00_descargas.ipynb`): Proceso para obtener los datos desde la fuente original.
2. **Consolidación** (`01_consolidar.ipynb`): Integración de los archivos mensuales en un solo conjunto de datos.
3. **Limpieza de datos** (`02_limpieza.ipynb`): Preprocesamiento para manejar valores faltantes, outliers y transformaciones.
4. **Análisis Exploratorio de Datos** (`03_EDA.ipynb`): Visualizaciones y estadísticas descriptivas.
5. **Modelado** (`04_modelo.ipynb`): Desarrollo e implementación de modelos predictivos.

## Requisitos

Para ejecutar este proyecto necesitarás:

```
python>=3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/usuario/ciencia_de_datos.git
```

## Uso

Ejecuta los cuadernos Jupyter en orden:

```bash
cd ciencia_de_datos
jupyter-notebook src/
```

