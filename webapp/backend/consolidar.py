"""Paso 01 - Consolidar.

Unifica las hojas de cada Excel de ``data/raw`` en ``data/landing`` y luego
construye el consolidado historico 2013-2026 con la informacion geografica.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .utils import (
    guardar_excel,
    hojas_de_datos,
    limpiar_hoja,
    limpiar_mercado,
    listar_excels,
    log,
    titulo,
    vaciar_carpeta,
)

RENOMBRES_COLUMNAS = {
    "Fuente": "Mercado",
    "Precio  por kilogramo*": "Precio promedio por kilogramo*",
    "Precio": "Precio promedio por kilogramo*",
    "Precio promedio por kilogramo": "Precio promedio por kilogramo*",
    "Precio promedio por kilogramo ": "Precio promedio por kilogramo*",
    "Precio promedio por kilogramo* ": "Precio promedio por kilogramo*",
}


def unificar_hojas(ruta: Path, skiprows: int, marcar_hoja: bool = False) -> pd.DataFrame:
    """Lee todas las hojas de datos de un Excel y las apila."""
    xls = pd.ExcelFile(ruta)
    dfs = []
    for hoja in hojas_de_datos(xls):
        log(f"   Leyendo hoja: {hoja}")
        df = limpiar_hoja(pd.read_excel(ruta, sheet_name=hoja, skiprows=skiprows))
        df = df.drop(columns=["Código CPC"], errors="ignore")
        if marcar_hoja:
            df["MES_AÑO"] = hoja
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def generar_landing(limpiar_destino: bool = True) -> list[Path]:
    """Genera un archivo ``*_unificado.xlsx`` por cada Excel de raw."""
    config.crear_carpetas()
    if limpiar_destino:
        eliminados = vaciar_carpeta(config.LANDING_DIR)
        log(f"Archivos eliminados de landing: {eliminados}")

    plan: list[tuple[str, int, bool]] = [
        (nombre, skip, True) for nombre, skip in config.ARCHIVOS_HISTORICOS.items()
    ]
    plan.append((config.ARCHIVO_BASE, config.ARCHIVO_BASE_SKIPROWS, True))
    plan.append((config.ARCHIVO_ACTUAL, config.ARCHIVO_ACTUAL_SKIPROWS, False))

    generados: list[Path] = []
    for nombre, skiprows, marcar in plan:
        origen = config.RAW_DIR / nombre
        if not origen.exists():
            log(f"Omitido (no esta en raw): {nombre}")
            continue
        titulo(f"Procesando: {nombre}")
        try:
            df = unificar_hojas(origen, skiprows, marcar_hoja=marcar)
            if df.empty:
                log("Sin datos utiles, se omite.")
                continue
            df = df.rename(columns=RENOMBRES_COLUMNAS)
            salida = config.LANDING_DIR / nombre.replace(".xlsx", "_unificado.xlsx")
            guardar_excel(df, salida)
            generados.append(salida)
        except Exception as error:
            log(f"Error procesando {nombre}: {error}")

    return generados


def _cargar_catalogo() -> pd.DataFrame:
    """Catalogo de mercados (departamento/municipio) desde el archivo del anio en curso."""
    ruta = config.LANDING_DIR / config.ARCHIVO_ACTUAL_UNIFICADO
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta.name} en landing. Ejecuta primero el paso 00 y generar_landing()."
        )
    catalogo = pd.read_excel(ruta)
    catalogo.columns = catalogo.columns.str.strip()
    catalogo["Mercado"] = catalogo["Mercado"].apply(limpiar_mercado)
    return catalogo


def unir_historicos() -> pd.DataFrame:
    """Apila los archivos de landing (sin el del anio en curso) y ordena por fecha."""
    archivos = listar_excels(
        config.LANDING_DIR,
        excluir={
            config.ARCHIVO_ACTUAL_UNIFICADO,
            config.CONSOLIDADO_XLSX,
            config.FINAL_CONSOLIDADO_XLSX,
        },
    )
    dfs = []
    for ruta in archivos:
        log(f"Procesando: {ruta.name}")
        df = pd.read_excel(ruta)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=["AÑO-MES", "MES_AÑO", "Código CPC"], errors="ignore")
        df = df.rename(columns=RENOMBRES_COLUMNAS)
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.dropna(subset=["Fecha"])
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No hay archivos unificados en landing.")

    df_final = pd.concat(dfs, ignore_index=True)
    return df_final.sort_values("Fecha").reset_index(drop=True)


def enriquecer_geografia(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega departamento y municipio cruzando por el nombre del mercado."""
    catalogo = _cargar_catalogo()
    columnas_geo = [
        "Mercado",
        "Departamento",
        "Código departamento",
        "Municipio",
        "Código municipio",
    ]
    catalogo = catalogo[columnas_geo].drop_duplicates(subset="Mercado")

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Mercado"] = df["Mercado"].apply(limpiar_mercado)
    df = df.drop(columns=[c for c in columnas_geo if c != "Mercado"], errors="ignore")
    return df.merge(catalogo, on="Mercado", how="left")


def _aplicar_catalogo_manual(df: pd.DataFrame, nombre_archivo: str) -> pd.DataFrame:
    """Aplica los catalogos manuales opcionales de data/mercados_faltantes."""
    ruta = config.MERCADOS_FALTANTES_DIR / nombre_archivo
    if not ruta.exists():
        log(f"Catalogo manual opcional no encontrado (se omite): {nombre_archivo}")
        return df

    tabla = pd.read_excel(ruta)
    tabla.columns = tabla.columns.astype(str).str.strip()
    catalogo = _cargar_catalogo().drop_duplicates(subset="Mercado").set_index("Mercado")
    df = df.copy()
    actualizados = 0

    if {"orginal", "Final"}.issubset(tabla.columns):
        # Correcciones de nombre: original -> final
        for _, fila in tabla.iterrows():
            original = str(fila["orginal"]).strip()
            nuevo = str(fila["Final"]).strip()
            mask = df["Mercado"].astype(str).str.strip() == original
            if not mask.any():
                continue
            df.loc[mask, "Mercado"] = nuevo
            if nuevo in catalogo.index:
                for columna in [
                    "Departamento",
                    "Código departamento",
                    "Municipio",
                    "Código municipio",
                ]:
                    df.loc[mask, columna] = catalogo.loc[nuevo, columna]
                actualizados += int(mask.sum())
    elif "Mercado" in tabla.columns:
        # Mercados extintos con su geografia manual
        tabla["Mercado"] = tabla["Mercado"].astype(str).str.strip()
        extintos = tabla.drop_duplicates(subset="Mercado").set_index("Mercado")
        for mercado in df.loc[df["Departamento"].isna(), "Mercado"].dropna().unique():
            if mercado in extintos.index:
                mask = df["Mercado"] == mercado
                for columna in [
                    "Departamento",
                    "Código departamento",
                    "Municipio",
                    "Código municipio",
                ]:
                    if columna in extintos.columns:
                        df.loc[mask, columna] = extintos.loc[mercado, columna]
                actualizados += int(mask.sum())

    log(f"{nombre_archivo}: registros actualizados {actualizados:,}")
    return df


def marcar_circulacion(df: pd.DataFrame, anios_vigentes=(2025, 2026)) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    vigentes = set(
        df.loc[df["Fecha"].dt.year.isin(anios_vigentes), "Mercado"].dropna().unique()
    )
    df["Circulación"] = df["Mercado"].apply(lambda m: "Sí" if m in vigentes else "No")
    extinguidos = sorted(df.loc[df["Circulación"] == "No", "Mercado"].dropna().unique())
    log(f"Mercados extinguidos: {len(extinguidos)}")
    return df


def ejecutar() -> Path:
    """Ejecuta el paso completo de consolidacion."""
    titulo("PASO 01 - CONSOLIDAR")
    generar_landing()

    df = unir_historicos()
    df = enriquecer_geografia(df)
    df = _aplicar_catalogo_manual(df, config.ARCHIVO_CAMBIOS)
    df = _aplicar_catalogo_manual(df, config.ARCHIVO_EXTINTOS)

    faltantes = sorted(df.loc[df["Departamento"].isna(), "Mercado"].dropna().unique())
    log(f"Mercados sin informacion geografica: {len(faltantes)}")
    for mercado in faltantes:
        log(f" - {mercado}")

    # Agregar el anio en curso y dejar el orden de columnas definitivo
    actual = pd.read_excel(config.LANDING_DIR / config.ARCHIVO_ACTUAL_UNIFICADO)
    actual.columns = actual.columns.str.strip()
    actual = actual.rename(columns=RENOMBRES_COLUMNAS)
    actual["Mercado"] = actual["Mercado"].apply(limpiar_mercado)

    for columna in config.COLUMNAS_FINALES:
        if columna not in df.columns:
            df[columna] = pd.NA
        if columna not in actual.columns:
            actual[columna] = pd.NA

    df = df[config.COLUMNAS_FINALES]
    actual = actual[config.COLUMNAS_FINALES]

    consolidado = pd.concat([df, actual], ignore_index=True)
    consolidado["Fecha"] = pd.to_datetime(consolidado["Fecha"], errors="coerce")
    consolidado = consolidado.dropna(subset=["Fecha"])
    consolidado = consolidado.sort_values("Fecha").reset_index(drop=True)
    guardar_excel(consolidado, config.LANDING_DIR / config.CONSOLIDADO_XLSX)

    consolidado = marcar_circulacion(consolidado)
    return guardar_excel(consolidado, config.LANDING_DIR / config.FINAL_CONSOLIDADO_XLSX)


if __name__ == "__main__":
    ejecutar()
