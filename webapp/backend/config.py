"""Configuracion de rutas locales del proyecto SIPSA.

Todo se guarda en el disco local (carpeta ``data/``), sin GitHub.
Se puede cambiar la raiz con la variable de entorno ``SIPSA_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent

# La carpeta de datos vive dentro de backend/ (backend/data).
# Si existe una carpeta data/ en la raiz del proyecto y no dentro de backend,
# se usa esa como respaldo. Se puede forzar con SIPSA_DATA_DIR.
_env_dir = os.environ.get("SIPSA_DATA_DIR")
if _env_dir:
    DATA_DIR = Path(_env_dir).resolve()
elif not (BACKEND_DIR / "data").exists() and (BASE_DIR / "data").exists():
    DATA_DIR = (BASE_DIR / "data").resolve()
else:
    DATA_DIR = (BACKEND_DIR / "data").resolve()

DESCARGAS_DIR = DATA_DIR / "Descargas"
RAW_DIR = DATA_DIR / "raw"
LANDING_DIR = DATA_DIR / "landing"
TRUSTED_DIR = DATA_DIR / "trusted"
MERCADOS_FALTANTES_DIR = DATA_DIR / "mercados_faltantes"

TODAS_LAS_CARPETAS = [
    DESCARGAS_DIR,
    RAW_DIR,
    LANDING_DIR,
    TRUSTED_DIR,
    MERCADOS_FALTANTES_DIR,
]

# Fuente oficial DANE - archivo del anio en curso
URL_ANIO_ACTUAL = (
    "https://www.dane.gov.co/files/operaciones/SIPSA/"
    "anex-SIPSA-SerieHistoricaMayorista-2026.xlsx"
)

# Hojas auxiliares que nunca contienen datos
HOJAS_EXCLUIDAS = {"indice", "índice", "fuentes", "cpc"}

# Archivos historicos y cuantas filas de encabezado saltar en cada uno
ARCHIVOS_HISTORICOS = {
    "series-historicas-precios-mayoristas-2018.xlsx": 6,
    "series-historicas-precios-mayoristas-2019.xlsx": 6,
    "series-historicas-precios-mayoristas-2020.xlsx": 6,
    "series-historicas-precios-mayoristas-2021.xlsx": 6,
    "series-historicas-precios-mayoristas-2022.xlsx": 6,
    "anex-SIPSA-SerieHistoricaMayorista-Dic2023.xlsx": 6,
    "anex-SIPSA-SerieHistoricaMayorista-2024.xlsx": 5,
    "anex-SIPSA-SerieHistoricaMayorista-2025 (2).xlsx": 5,
}

# Archivo base 2013-2017 (encabezado mas largo y columna "Fuente")
ARCHIVO_BASE = "series-historicas-precios-mayoristas.xlsx"
ARCHIVO_BASE_SKIPROWS = 9

# Archivo del anio en curso (sirve tambien como catalogo de mercados)
ARCHIVO_ACTUAL = "anex-SIPSA-SerieHistoricaMayorista-2026.xlsx"
ARCHIVO_ACTUAL_SKIPROWS = 5
ARCHIVO_ACTUAL_UNIFICADO = "anex-SIPSA-SerieHistoricaMayorista-2026_unificado.xlsx"

# Salidas
CONSOLIDADO_XLSX = "SIPSA_2013_2026_consolidado.xlsx"
FINAL_CONSOLIDADO_XLSX = "SIPSA_2013_2026_FINAL_consolidado.xlsx"
TRUSTED_XLSX = "SIPSA_2013_2026_trusted.xlsx"

# Catalogos manuales opcionales (los coloca el usuario en mercados_faltantes/)
ARCHIVO_CAMBIOS = "Cambio_de_nombre o errores.xlsx"
ARCHIVO_EXTINTOS = "extintos.xlsx"

COLUMNAS_FINALES = [
    "Fecha",
    "Grupo",
    "Producto",
    "Mercado",
    "Departamento",
    "Código departamento",
    "Municipio",
    "Código municipio",
    "Precio promedio por kilogramo*",
]

RENOMBRES_FINALES = {
    "Fecha": "fecha",
    "Grupo": "grupo",
    "Producto": "producto",
    "Mercado": "mercado",
    "Departamento": "departamento",
    "Código departamento": "codigo_departamento",
    "Municipio": "municipio",
    "Código municipio": "codigo_municipio",
    "Precio promedio por kilogramo*": "precio_promedio_kg",
    "Circulación": "circulacion",
}


def crear_carpetas() -> None:
    """Crea la estructura local de carpetas si no existe."""
    for carpeta in TODAS_LAS_CARPETAS:
        carpeta.mkdir(parents=True, exist_ok=True)