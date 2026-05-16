# Proyecto de Análisis de Datos

## Descripción
Este proyecto está diseñado para procesar, analizar y modelar datos mensuales recopilados desde noviembre de 2022 hasta marzo de 2024. El flujo de trabajo incluye la descarga de datos, consolidación, limpieza, análisis exploratorio de datos (EDA) y modelado predictivo.

## Estructura del Proyecto

```
.
├── src                   # Código fuente (cuadernos Jupyter)
├── data                  # Directorio para todos los datos
│   ├── raw               # Datos brutos organizados por año y mes
│   ├── landing           # Datos consolidados sin procesar
│   ├── trusted           # Datos limpios listos para análisis y modelado
│   └── surface           # Datos organizados para compartir con usuarios
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
5. **Modelado** (`04_modelos.ipynb`): Desarrollo e implementación de modelos predictivos.

## Datos

Los datos están organizados de la siguiente manera:

- **raw**: Datos crudos (tal cual se descargan), archivos de ejemplo en formato Excel
- **landing**: Datos consolidados de todas las fuentes, sin más procesamiento
- **trusted**: Datos limpios, normalizados, estandarizados, etc. listos para análisis
- **surface**: Datos organizados para compartir con usuarios

## Requisitos

Para ejecutar este proyecto necesitarás:

```
python>=3.12
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Instalación

1. Crear un **fork** de este repositorio, cambia el nombre según el tema de tu proyecto:

    Esto crea una copia en tu cuenta de GitHub: 

    `https://github.com/dfmarin/proyecto_diplomado`

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

[Tu nombre] - [tu.email@ejemplo.com]

Enlace del proyecto: [https://github.com/usuario/nombre-del-proyecto](https://github.com/usuario/nombre-del-proyecto)

