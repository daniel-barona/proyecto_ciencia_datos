@echo off
echo ===================================
echo Pipeline SIPSA - %date% %time%
echo ===================================

cd /d C:\analisis_de_datos\proyecto_ciencia_datos
python -m webapp.backend.pipeline

echo.
echo Finalizado: %date% %time%
echo ===================================