"""Paso 00 - Descargas.

Descarga el archivo SIPSA del anio en curso desde el DANE y prepara la capa
``data/raw`` a partir de ``data/Descargas``. Todo queda en el disco local.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import requests

from . import config
from .utils import listar_excels, log, titulo, vaciar_carpeta

# El sitio del DANE usa un WAF que bloquea peticiones de bots y de IPs de
# nube (como los runners de GitHub Actions), por lo que se envian cabeceras
# de navegador y se reintenta varias veces antes de rendirse.
CABECERAS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.dane.gov.co/",
}

INTENTOS = 4
PAUSA_SEGUNDOS = 30

# Tamano minimo razonable para el Excel real; respuestas atrapadas por el
# WAF suelen ser paginas de error muy pequenas.
TAMANO_MINIMO_BYTES = 100_000


def _descargar_a(url: str, destino_temporal: Path) -> None:
    respuesta = requests.get(url, stream=True, timeout=120, headers=CABECERAS)
    respuesta.raise_for_status()
    with open(destino_temporal, "wb") as archivo:
        for bloque in respuesta.iter_content(chunk_size=8192):
            if bloque:
                archivo.write(bloque)
    if destino_temporal.stat().st_size < TAMANO_MINIMO_BYTES:
        raise ValueError(
            f"Respuesta sospechosamente pequena ({destino_temporal.stat().st_size} bytes)"
        )


def descargar_anio_actual(url: str | None = None, forzar: bool = True) -> Path:
    """Descarga el Excel del anio en curso a ``data/Descargas``.

    La descarga va primero a un archivo temporal y el destino solo se
    reemplaza al final si todo salio bien, para conservar la copia previa
    cuando el servidor falla o bloquea la peticion.
    """
    config.crear_carpetas()
    url = url or config.URL_ANIO_ACTUAL
    destino = config.DESCARGAS_DIR / Path(url).name

    if destino.exists() and not forzar:
        log(f"Ya existe (no se descarga de nuevo): {destino.name}")
        return destino

    titulo(f"Descargando {destino.name}")
    temporal = destino.with_name(destino.name + ".part")

    for intento in range(1, INTENTOS + 1):
        try:
            _descargar_a(url, temporal)
            break
        except Exception as error:
            temporal.unlink(missing_ok=True)
            log(f"Intento {intento}/{INTENTOS} fallo: {error}")
            if intento == INTENTOS:
                raise
            log(f"Reintento en {PAUSA_SEGUNDOS} s...")
            time.sleep(PAUSA_SEGUNDOS)

    if destino.exists():
        log(f"Reemplazando copia previa: {destino.name}")
    temporal.replace(destino)

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
        except Exception as error:  # la fuente puede estar caida o bloquear
            log(f"No se pudo descargar el archivo del anio en curso: {error}")
            log("Se continua con la copia previa en data/Descargas.")
    copiar_a_raw()
    return inventario_raw()


if __name__ == "__main__":
    ejecutar()
