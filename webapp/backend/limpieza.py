"""Paso 02 - Limpieza.

Toma el consolidado de ``data/landing`` y produce el dataset de aprendizaje
``data/trusted/SIPSA_2013_2026_trusted.xlsx`` (tambien en CSV/Parquet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .utils import guardar_excel, log, normalizar_texto, titulo


def cargar_consolidado() -> pd.DataFrame:
    ruta = config.LANDING_DIR / config.FINAL_CONSOLIDADO_XLSX
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero el paso 01 (consolidar)."
        )
    return pd.read_excel(ruta)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normalizar texto: sin tildes, minusculas, sin espacios sobrantes
    for columna in df.select_dtypes(include="object").columns:
        df[columna] = df[columna].apply(normalizar_texto)

    # Espacios dobles y punto final en los nombres de mercado
    modificados = 0
    for columna in df.select_dtypes(include="object").columns:
        antes = df[columna].copy()
        df[columna] = (
            df[columna]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        modificados += int((antes.astype("string") != df[columna]).sum())
    log(f"Valores de texto ajustados: {modificados:,}")

    if "Mercado" in df.columns:
        df["Mercado"] = df["Mercado"].astype("string").str.rstrip(".").str.strip()

    df = df.rename(columns=config.RENOMBRES_FINALES)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    if "precio_promedio_kg" in df.columns:
        df["precio_promedio_kg"] = pd.to_numeric(
            df["precio_promedio_kg"], errors="coerce"
        )
        df = df.dropna(subset=["precio_promedio_kg"])
        df = df[df["precio_promedio_kg"] > 0]

    df = df.drop_duplicates()
    return df.sort_values("fecha").reset_index(drop=True)


def ejecutar() -> Path:
    """Ejecuta el paso completo de limpieza y guarda el dataset final."""
    titulo("PASO 02 - LIMPIEZA")
    config.crear_carpetas()

    df = limpiar(cargar_consolidado())
    ruta = guardar_excel(df, config.TRUSTED_DIR / config.TRUSTED_XLSX)

    # Formatos livianos para que Streamlit cargue rapido
    df.to_csv(config.TRUSTED_DIR / "SIPSA_2013_2026_trusted.csv", index=False)
    try:
        df.to_parquet(config.TRUSTED_DIR / "SIPSA_2013_2026_trusted.parquet", index=False)
    except Exception as error:
        log(f"No se genero el parquet (opcional): {error}")

    log(f"Columnas finales: {df.columns.tolist()}")
    return ruta


if __name__ == "__main__":
    ejecutar()
