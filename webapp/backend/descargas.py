"""Paso 00 - Descargas.

Descarga el archivo SIPSA del anio en curso desde el DANE y prepara la capa
``data/raw`` a partir de ``data/Descargas``. Todo queda en el disco local.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import requests

from . import config
from .utils import listar_excels, log, titulo, vaciar_carpeta


def descargar_anio_actual(url: str | None = None, forzar: bool = True) -> Path:
    """Descarga el Excel del anio en curso a ``data/Descargas``."""
    config.crear_carpetas()
    url = url or config.URL_ANIO_ACTUAL
    destino = config.DESCARGAS_DIR / Path(url).name

    if destino.exists():
        if not forzar:
            log(f"Ya existe (no se descarga de nuevo): {destino.name}")
            return destino
        destino.unlink()
        log(f"Archivo previo eliminado: {destino.name}")

    titulo(f"Descargando {destino.name}")
    respuesta = requests.get(url, stream=True, timeout=120)
    respuesta.raise_for_status()
    with open(destino, "wb") as archivo:
        for bloque in respuesta.iter_content(chunk_size=8192):
            if bloque:
                archivo.write(bloque)

    log(f"Descarga completada: {destino}")
    return destino


def copiar_a_raw(limpiar_destino: bool = True) -> list[Path]:
    """Copia los Excel de ``Descargas`` hacia ``raw``."""
    config.crear_carpetas()

    if limpiar_destino:
        eliminados = vaciar_carpeta(config.RAW_DIR)
        log(f"Archivos eliminados de raw: {eliminados}")

    copiados: list[Path] = []
    for archivo in config.DESCARGAS_DIR.rglob("*"):
        if not archivo.is_file() or archivo.name.startswith("~$"):
            continue
        destino = config.RAW_DIR / archivo.relative_to(config.DESCARGAS_DIR)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archivo, destino)
        copiados.append(destino)

    log(f"Archivos copiados a raw: {len(copiados)}")
    return copiados


def inventario_raw() -> list[str]:
    archivos = listar_excels(config.RAW_DIR)
    titulo(f"Archivos en raw: {len(archivos)}")
    for archivo in archivos:
        log(f" - {archivo.name}")
    return [a.name for a in archivos]


def ejecutar(descargar: bool = True) -> list[str]:
    """Ejecuta el paso completo de descargas."""
    titulo("PASO 00 - DESCARGAS")
    if descargar:
        try:
            descargar_anio_actual()
        except Exception as error:  # la fuente puede estar caida
            log(f"No se pudo descargar el archivo del anio en curso: {error}")
            log("Se continua con los archivos ya presentes en data/Descargas.")
    copiar_a_raw()
    return inventario_raw()


if __name__ == "__main__":
    ejecutar()
