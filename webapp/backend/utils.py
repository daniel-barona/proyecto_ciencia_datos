"""Utilidades compartidas del pipeline SIPSA."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


def log(mensaje: str) -> None:
    print(mensaje, flush=True)


def titulo(mensaje: str) -> None:
    log("\n" + "=" * 80)
    log(mensaje)
    log("=" * 80)


def sin_tildes(texto: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def limpiar_mercado(valor):
    """Normaliza el nombre de un mercado (espacios, tildes, alias conocidos)."""
    if pd.isna(valor):
        return valor
    texto = re.sub(r"\s+", " ", str(valor).strip())
    texto = sin_tildes(texto)
    texto = texto.rstrip(".,").strip()
    texto = texto.replace("Santa Helena", "Santa Elena")
    texto = texto.replace(", panela", "")
    return texto


def normalizar_texto(valor):
    """Minusculas, sin tildes y sin espacios sobrantes."""
    if pd.isna(valor) or not isinstance(valor, str):
        return valor
    return sin_tildes(valor).lower().strip()


def hojas_de_datos(xls: pd.ExcelFile) -> list[str]:
    return [h for h in xls.sheet_names if h.strip().lower() not in config.HOJAS_EXCLUIDAS]


def quitar_notas_al_pie(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas de aclaraciones ('Los precios reportados...')."""
    if df.empty:
        return df
    mascara = (
        df.astype(str)
        .apply(
            lambda fila: fila.str.contains(
                "Los precios reportados", case=False, na=False, regex=False
            ).any(),
            axis=1,
        )
    )
    return df[~mascara]


def limpiar_hoja(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return quitar_notas_al_pie(df)


def vaciar_carpeta(carpeta: Path, conservar_subcarpetas: bool = True) -> int:
    """Borra los archivos de una carpeta local. Devuelve cuantos elimino."""
    if not carpeta.exists():
        return 0
    eliminados = 0
    for archivo in carpeta.rglob("*"):
        if archivo.is_file():
            archivo.unlink()
            eliminados += 1
    if not conservar_subcarpetas:
        for sub in sorted(carpeta.glob("*"), reverse=True):
            if sub.is_dir():
                sub.rmdir()
    return eliminados


def guardar_excel(df: pd.DataFrame, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)
    log(f"Archivo guardado: {ruta}  ({len(df):,} filas, {df.shape[1]} columnas)")
    return ruta


def listar_excels(carpeta: Path, excluir: Iterable[str] = ()) -> list[Path]:
    excluidos = set(excluir)
    return sorted(
        p
        for p in carpeta.glob("*.xlsx")
        if p.name not in excluidos and not p.name.startswith("~$")
    )
