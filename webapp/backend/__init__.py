"""Backend local del proyecto SIPSA (descargas, consolidacion y limpieza)."""

from . import config, consolidar, descargas, limpieza, pipeline, utils

__all__ = ["config", "consolidar", "descargas", "limpieza", "pipeline", "utils"]
