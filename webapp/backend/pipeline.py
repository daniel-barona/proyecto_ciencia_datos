"""Orquestador del pipeline SIPSA (todo local, sin GitHub).

Uso:
    python -m backend.pipeline                # pipeline completo
    python -m backend.pipeline --paso 00      # solo descargas
    python -m backend.pipeline --paso 01      # solo consolidar
    python -m backend.pipeline --paso 02      # solo limpieza
    python -m backend.pipeline --sin-descarga # usa lo que ya esta en data/Descargas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, consolidar, descargas, limpieza
from .utils import titulo


def ruta_dataset() -> Path:
    return config.TRUSTED_DIR / config.TRUSTED_XLSX


def dataset_disponible() -> bool:
    return ruta_dataset().exists()


def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset limpio para la app de Streamlit."""
    parquet = config.TRUSTED_DIR / "SIPSA_2013_2026_trusted.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if not dataset_disponible():
        raise FileNotFoundError(
            "No existe el dataset limpio. Ejecuta: python -m backend.pipeline"
        )
    return pd.read_excel(ruta_dataset())


def ejecutar_todo(descargar: bool = True) -> Path:
    config.crear_carpetas()
    descargas.ejecutar(descargar=descargar)
    consolidar.ejecutar()
    ruta = limpieza.ejecutar()
    titulo("PIPELINE FINALIZADO")
    print(f"Dataset de aprendizaje: {ruta}")
    return ruta


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline SIPSA local")
    parser.add_argument("--paso", choices=["00", "01", "02"], help="Ejecutar un solo paso")
    parser.add_argument(
        "--sin-descarga",
        action="store_true",
        help="No descargar del DANE; usar los archivos en data/Descargas",
    )
    args = parser.parse_args()

    config.crear_carpetas()
    if args.paso == "00":
        descargas.ejecutar(descargar=not args.sin_descarga)
    elif args.paso == "01":
        consolidar.ejecutar()
    elif args.paso == "02":
        limpieza.ejecutar()
    else:
        ejecutar_todo(descargar=not args.sin_descarga)


if __name__ == "__main__":
    main()
