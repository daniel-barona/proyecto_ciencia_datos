# -*- coding: utf-8 -*-
"""
04_modelo.py
------------
Motor de predicción SIPSA (SARIMA/ARIMA) refactorizado desde el notebook
04_modelo.ipynb para ser IMPORTADO por la aplicación (Streamlit/Flask/API).

Mantiene la MISMA ESTRUCTURA que ``modelo.py`` (dataclass ``Fase``, carga de
datos ``cargar_trusted``, selectores en cascada ``opciones_*`` y un
``ejecutar_pipeline`` que devuelve un dict), pero conserva el CÓDIGO del
notebook 04_modelo: backtest rolling-origin multi-horizonte, familia amplia de
candidatos SARIMA/ARIMA (con y sin deriva), control de "planitud" del
pronóstico, intervalos de confianza empíricos y persistencia con joblib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (para Streamlit / servidores)
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from config import TRUSTED_DIR
except ImportError:  # permite importar desde la raiz del proyecto
    try:
        from backend.config import TRUSTED_DIR
    except ImportError:  # fallback local si no existe modulo de configuracion
        TRUSTED_DIR = Path(__file__).resolve().parent / "data" / "trusted"

# joblib es OPCIONAL: solo se necesita para persistir el modelo entrenado.
try:
    import joblib

    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

# pmdarima es OPCIONAL: si no esta, se usa un grid search propio.
try:
    import pmdarima as pm
    from pmdarima.arima.utils import ndiffs, nsdiffs

    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False


# ======================================================================
# Variables globales del modelo (equivalentes a las del notebook)
# ======================================================================
M = 12            # periodicidad estacional (12 = mensual con ciclo anual)
USE_LOG = True    # transformacion logaritmica para estabilizar varianza
TEST_SIZE = None  # None = se calcula automaticamente
H_FUTURO = 12     # meses a pronosticar hacia el futuro

UMBRAL_PLANITUD = 0.005  # por debajo de esto un pronostico se considera "plano"
N_ORIGENES = 6           # origenes para el backtest rolling-origin

# Nombre por defecto del archivo trusted consolidado
TRUSTED_FILE = "SIPSA_2013_2026_trusted.xlsx"

# Parametros de ajuste de SARIMAX (comunes a todo el pipeline)
FIT_KW = dict(enforce_stationarity=True, enforce_invertibility=True)


# ======================================================================
# Estructura para almacenar la salida de cada fase
# ======================================================================
@dataclass
class Fase:
    """Contiene todo lo que produjo una fase del pipeline."""

    id: str                                     # ej. "Fase 1"
    titulo: str                                 # ej. "Creacion de la serie"
    logs: list = field(default_factory=list)    # lineas de texto
    tablas: list = field(default_factory=list)  # lista de (titulo, DataFrame)
    figuras: list = field(default_factory=list)  # lista de matplotlib.Figure

    def log(self, *partes) -> None:
        self.logs.append(" ".join(str(p) for p in partes))

    def tabla(self, titulo: str, df: pd.DataFrame) -> None:
        self.tablas.append((titulo, df))

    def figura(self, fig) -> None:
        self.figuras.append(fig)

    @property
    def texto(self) -> str:
        return "\n".join(self.logs)


# ======================================================================
# Carga de datos y selectores en cascada (reemplazan a los dropdowns)
# ======================================================================
def cargar_trusted(trusted_path: Optional[Path] = None) -> pd.DataFrame:
    """Carga el dataset trusted consolidado desde el backend local."""
    ruta = Path(trusted_path) if trusted_path else (TRUSTED_DIR / TRUSTED_FILE)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo trusted en: {ruta}\n"
            "Ejecuta primero el backend de descarga/consolidacion/limpieza "
            "para generar la capa trusted."
        )
    df = pd.read_excel(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _anio_referencia(df: pd.DataFrame) -> int:
    """Ultimo anio disponible en los datos (equivale al 2026 del notebook)."""
    return int(df["fecha"].dt.year.max())


def opciones_departamentos(df: pd.DataFrame) -> list:
    """'colombia' + departamentos con datos en el anio de referencia."""
    anio = _anio_referencia(df)
    deps = sorted(
        df.loc[df["fecha"].dt.year == anio, "departamento"].dropna().unique()
    )
    return ["colombia"] + deps


def requiere_municipio(departamento: str) -> bool:
    return departamento.lower() not in ["colombia", "bogota, d.c."]


def opciones_municipios(df: pd.DataFrame, departamento: str) -> list:
    anio = _anio_referencia(df)
    return sorted(
        df[(df["departamento"] == departamento) & (df["fecha"].dt.year == anio)]
        ["municipio"].dropna().unique().tolist()
    )


def opciones_mercados(
    df: pd.DataFrame, departamento: str, municipio: Optional[str] = None
) -> list:
    """Mercados disponibles. En modo 'colombia' no aplica (promedio nacional)."""
    anio = _anio_referencia(df)
    if departamento.lower() == "colombia":
        return []
    if departamento.lower() == "bogota, d.c.":
        base = df[(df["departamento"] == departamento) & (df["fecha"].dt.year == anio)]
    else:
        base = df[(df["municipio"] == municipio) & (df["fecha"].dt.year == anio)]
    return sorted(base["mercado"].dropna().unique().tolist())


def opciones_productos(df: pd.DataFrame, mercado: Optional[str]) -> list:
    anio = _anio_referencia(df)
    if mercado is None:  # modo nacional
        base = df[df["fecha"].dt.year == anio]
    else:
        base = df[(df["mercado"] == mercado) & (df["fecha"].dt.year == anio)]
    return sorted(base["producto"].dropna().unique().tolist())


# ======================================================================
# Funciones estadisticas (identicas en logica al notebook)
# ======================================================================
def preparar_serie(df, producto, mercado=None):
    """Construye la serie mensual del producto (promedio si mercado=None)."""
    base = df[df["producto"] == producto]
    if mercado is not None:
        base = base[base["mercado"] == mercado]

    serie = base[["fecha", "precio_promedio_kg"]].copy()
    serie = serie.sort_values("fecha")
    serie.set_index("fecha", inplace=True)
    serie.rename(columns={"precio_promedio_kg": "precio"}, inplace=True)
    serie = serie.groupby(serie.index).mean()
    serie = serie.asfreq("MS").interpolate("linear").ffill().bfill()
    return serie["precio"]


def fuerza_estacional(y, m=12):
    """Fs = max(0, 1 - Var(resid)/Var(resid+seasonal)) (Hyndman)."""
    if len(y) < 2 * m + 1:
        return np.nan
    try:
        res = STL(y, period=m, robust=True).fit()
    except Exception:
        return np.nan
    var_r = np.nanvar(res.resid)
    var_rs = np.nanvar(res.resid + res.seasonal)
    return float(max(0.0, 1 - var_r / var_rs)) if var_rs > 0 else 0.0


def verificar_registros(y, fase: Optional[Fase] = None, m=12, es_train=False):
    """Verifica tamano/estacionalidad y define limites de busqueda."""
    n = int(y.dropna().shape[0])
    ciclos = n / m
    nivel = (
        "INSUFICIENTE" if n < 24 else "MINIMO" if n < 36 else
        "ACEPTABLE" if n < 48 else "BUENO" if n < 72 else "OPTIMO"
    )

    Fs = fuerza_estacional(y, m)
    ciclos_ok = n >= 3 * m
    seasonal_ok = bool(ciclos_ok and np.isfinite(Fs) and Fs >= 0.30)
    seasonal_debil = bool(ciclos_ok and np.isfinite(Fs) and 0.15 <= Fs < 0.30)

    presupuesto = max(2, n // 10)
    max_p = 3 if n >= 60 else 2 if n >= 36 else 1
    max_q = max_p
    max_P = 1 if seasonal_ok else 0
    max_Q = 1 if (seasonal_ok and n >= 4 * m) else 0
    h_bt = int(min(m, max(6, n // 5), max(3, n - max(3 * m, 24))))

    info = dict(
        n=n, ciclos=round(ciclos, 2), nivel=nivel, apto=n >= 24, m=m,
        Fs=None if not np.isfinite(Fs) else round(Fs, 3),
        seasonal_ok=seasonal_ok, seasonal_debil=seasonal_debil,
        max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
        presupuesto=presupuesto, h_bt=h_bt,
    )

    if fase is not None:
        etiqueta = "train" if es_train else "serie"
        fase.log(f"Registros            : {n}")
        fase.log(f"Ciclos (m={m})        : {ciclos:.2f}")
        fase.log(f"Nivel                : {nivel}")
        fase.log(f"Fuerza estacional Fs : {info['Fs']}  (>=0.30 usa SARIMA, <0.15 no aporta)")
        fase.log(
            "Componente estacional: "
            + ("SI" if seasonal_ok else "NO -> ARIMA no estacional")
            + (" [debil]" if seasonal_debil else "")
        )
        fase.log(
            f"Limites busqueda     : p<={max_p} q<={max_q} P<={max_P} Q<={max_Q} "
            f"(max {presupuesto} params)"
        )
        fase.log(f"Horizonte backtest   : {h_bt} meses")
        if not info["apto"]:
            fase.log(f"ATENCION: {etiqueta} insuficiente para modelar (<24).")
    return info


def detectar_outliers(y_serie, umbral=3.5):
    """Detecta variaciones mes a mes atipicas (z-score robusto: mediana+MAD)."""
    diffs = y_serie.diff().dropna()
    mediana = diffs.median()
    mad = (diffs - mediana).abs().median()
    if mad == 0:
        return pd.Series(dtype=float)
    z = 0.6745 * (diffs - mediana) / mad
    return diffs[z.abs() > umbral]


def dividir_train_test(y, fase: Optional[Fase] = None, test_size=None, m=12, min_train=None):
    """Divide la serie de forma cronologica (sin mezclar el orden temporal)."""
    n = len(y)
    min_train = min_train or max(3 * m, 24)

    if test_size is None:
        test_size = int(min(m, max(6, n // 5), max(3, n - min_train)))
    test_size = (
        max(1, min(test_size, n - min_train)) if n > min_train else max(1, n // 5)
    )

    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    if fase is not None:
        fase.log(f"Serie total   : {n} observaciones")
        fase.log(
            f"Train         : {len(y_train)} observaciones "
            f"({y_train.index[0]:%Y-%m} a {y_train.index[-1]:%Y-%m})"
        )
        fase.log(
            f"Test          : {len(y_test)} observaciones "
            f"({y_test.index[0]:%Y-%m} a {y_test.index[-1]:%Y-%m})"
        )
    return y_train, y_test


def test_estacionariedad(x, fase: Optional[Fase] = None, nombre=""):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        if fase:
            fase.log(f"{nombre}: muestra insuficiente ({len(x)})")
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        adf = adfuller(x)[1]
        kps = kpss(x, nlags="auto")[1]
    if fase:
        fase.log(f"{nombre}: ADF p={adf:.3f}  KPSS p={kps:.3f}")
    return adf, kps


def estimar_dD(ys, info, fase: Optional[Fase] = None, m=12):
    """Estima d y D sobre el train (no se imponen)."""
    if HAS_PMDARIMA:
        d = int(min(2, ndiffs(ys, test="kpss", max_d=2)))
    else:
        adf = test_estacionariedad(ys)[0]
        d = 0 if (adf is not None and adf < 0.05) else 1
    D = 0
    if info["seasonal_ok"] and len(ys) >= 3 * m + m:
        if HAS_PMDARIMA:
            try:
                D = int(min(1, nsdiffs(ys, m=m, max_D=1, test="ocsb")))
            except Exception:
                D = 0
        else:
            D = 1
        if len(ys) - D * m - d < 2 * m:
            D = 0
    if fase is not None:
        fase.log(f"Diferenciacion estimada (train) -> d={d}, D={D} (m={m})")
    return d, D


# ======================================================================
# Fase 8/9: ajuste del modelo (auto_arima / grid AIC)
# ======================================================================
def _fit(ys, order, sorder, trend=None):
    return SARIMAX(
        ys, order=order, seasonal_order=sorder, trend=trend, **FIT_KW
    ).fit(disp=False)


def modelo_estable(fit):
    """True si todas las raices AR (y estacionales) estan fuera del circulo."""
    try:
        ar = np.concatenate(
            [np.atleast_1d(fit.arroots), np.atleast_1d(getattr(fit, "seasonalarroots", []))]
        )
        return bool(len(ar) == 0 or np.all(np.abs(ar) > 1.01))
    except Exception:
        return True


def ajustar_sarima(ys, info, d, D, fase: Optional[Fase] = None, m=12):
    """Ajusta SARIMA (auto_arima o grid AIC) solo con datos de train."""
    seasonal = info["seasonal_ok"] and (D > 0 or info["max_P"] + info["max_Q"] > 0)
    max_p, max_q = info["max_p"], info["max_q"]
    max_P = info["max_P"] if seasonal else 0
    max_Q = info["max_Q"] if seasonal else 0

    order, sorder = (1, d, 0), ((1, D, 0, m) if seasonal else (0, 0, 0, 0))
    if HAS_PMDARIMA:
        try:
            mdl = pm.auto_arima(
                ys, seasonal=seasonal, m=m if seasonal else 1,
                d=d, D=D if seasonal else 0,
                start_p=0, start_q=0, start_P=0, start_Q=0,
                max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                information_criterion="aicc",
                stationary=False, stepwise=True,
                suppress_warnings=True, error_action="ignore", trace=False,
            )
            order, sorder = mdl.order, (mdl.seasonal_order if seasonal else (0, 0, 0, 0))
        except Exception as e:
            if fase is not None:
                fase.log("auto_arima fallo, uso fallback:", e)
    else:
        mejor, best = None, np.inf
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        try:
                            f = _fit(
                                ys, (p, d, q),
                                (P, D, Q, m) if seasonal else (0, 0, 0, 0),
                            )
                            if f.aicc < best and modelo_estable(f):
                                best, mejor = f.aicc, (
                                    (p, d, q),
                                    (P, D, Q, m) if seasonal else (0, 0, 0, 0),
                                )
                        except Exception:
                            continue
        if mejor:
            order, sorder = mejor

    k = order[0] + order[2] + sorder[0] + sorder[2]
    if k > info["presupuesto"]:
        if fase is not None:
            fase.log(f"Modelo simplificado: k={k} > presupuesto={info['presupuesto']}")
        order = (1, d, 0)
        sorder = (1, D, 0, m) if seasonal else (0, 0, 0, 0)

    try:
        if not modelo_estable(_fit(ys, order, sorder)):
            if fase is not None:
                fase.log("Modelo inestable -> degradado a (0,d,1)")
            order = (0, d, 1)
            sorder = (0, D, 1, m) if seasonal else (0, 0, 0, 0)
    except Exception:
        order, sorder = (0, d, 1), (0, 0, 0, 0)
    return order, sorder


# ======================================================================
# Fase 10/11: metricas, predictores y backtest rolling-origin
# ======================================================================
def _inv_log(mu, var=None, use_log=True):
    """Vuelve a la escala original. Con log se aplica correccion de sesgo
    exp(mu + var/2) => se estima la MEDIA condicional, no la mediana."""
    mu = np.asarray(mu, float)
    if not use_log:
        return mu
    if var is None:
        return np.exp(mu)
    return np.exp(mu + np.asarray(var, float) / 2.0)


def _mase_denom(y_hist, m=12):
    yt = np.asarray(y_hist, float)
    lag = m if len(yt) > m + 2 else 1
    return np.mean(np.abs(yt[lag:] - yt[:-lag])) if len(yt) > lag else np.nan


def _metricas(y_true, y_pred, denom):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err / y_true)) * 100
    smape = np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100
    mase = mae / denom if denom and np.isfinite(denom) and denom > 0 else np.nan
    return dict(
        MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape, MASE=mase,
        Accuracy=max(0.0, 100 - mape),
    )


# ----------------------------------------------------------------------
# Predictores: cada uno recibe (y_hist, steps) y devuelve el pronostico
# en la ESCALA ORIGINAL. Asi baselines y modelos son comparables.
# ----------------------------------------------------------------------
def make_sarimax_predictor(order, sorder, trend=None, use_log=USE_LOG):
    def _p(y_hist, steps, want_full=False):
        ym = np.log(y_hist) if use_log else y_hist
        f = SARIMAX(ym, order=order, seasonal_order=sorder, trend=trend, **FIT_KW).fit(disp=False)
        fc = f.get_forecast(steps=steps)
        try:
            var = np.asarray(fc.var_pred_mean, float)
        except Exception:
            var = None
        punto = _inv_log(fc.predicted_mean.values, var, use_log)
        if not want_full:
            return punto
        ci = fc.conf_int(alpha=0.05)
        lo = np.exp(ci.iloc[:, 0].values) if use_log else ci.iloc[:, 0].values
        hi = np.exp(ci.iloc[:, 1].values) if use_log else ci.iloc[:, 1].values
        return punto, lo, hi, f
    return _p


def make_ets_predictor(damped=True, seasonal=False, m=12, use_log=USE_LOG):
    def _p(y_hist, steps, want_full=False):
        ym = np.log(y_hist) if use_log else y_hist
        kw = dict(trend="add", damped_trend=damped, initialization_method="estimated")
        if seasonal and len(ym) >= 2 * m + 1:
            kw.update(seasonal="add", seasonal_periods=m)
        f = ExponentialSmoothing(ym, **kw).fit(optimized=True)
        mu = np.asarray(f.forecast(steps), float)
        s2 = float(np.nanvar(np.asarray(f.resid, float)))
        punto = _inv_log(mu, np.full(steps, s2) * np.arange(1, steps + 1), use_log)
        return (punto, None, None, f) if want_full else punto
    return _p


def pred_naive(y_hist, steps, want_full=False):
    p = np.repeat(float(y_hist.iloc[-1]), steps)
    return (p, None, None, None) if want_full else p


def make_snaive(m=12):
    def _p(y_hist, steps, want_full=False):
        last = np.asarray(y_hist.iloc[-m:], float)
        p = np.array([last[i % len(last)] for i in range(steps)], float)
        return (p, None, None, None) if want_full else p
    return _p


def make_drift(ventana=None):
    """Naive con deriva: extiende la pendiente media reciente.
    Es la referencia MINIMA que un pronostico util debe batir y NUNCA es plano."""
    def _p(y_hist, steps, want_full=False):
        v = len(y_hist) if ventana is None else min(ventana, len(y_hist))
        z = y_hist.iloc[-v:]
        slope = (float(z.iloc[-1]) - float(z.iloc[0])) / max(1, len(z) - 1)
        p = float(y_hist.iloc[-1]) + slope * np.arange(1, steps + 1)
        return (p, None, None, None) if want_full else p
    return _p


def planitud(pred):
    """Mide que tan 'plano' es un pronostico: rango relativo respecto al nivel.
    ~0 => linea recta constante (sintoma del problema reportado)."""
    p = np.asarray(pred, float)
    return float((p.max() - p.min()) / max(1e-9, np.mean(np.abs(p))))


def backtest_rolling(y, predictor, h, min_train, max_origenes=6, m=12):
    """Reestima el modelo en varios origenes y pronostica el horizonte COMPLETO h.
    Devuelve (metricas_globales, perfil_por_horizonte, detalle, trayectorias)."""
    n = len(y)
    ultimos = n - h                       # ultimo origen con horizonte completo
    if ultimos < min_train:
        min_train = max(m + 2, int(n * 0.6))
    origenes = [o for o in range(min_train, ultimos + 1)]
    if not origenes:
        origenes = [max(m + 2, n - max(1, h // 2))]
    if len(origenes) > max_origenes:      # espaciados, siempre incluyendo el ultimo
        idx = np.unique(np.linspace(0, len(origenes) - 1, max_origenes).round().astype(int))
        origenes = [origenes[i] for i in idx]

    filas, err_h, trayectorias = [], {k: [] for k in range(1, h + 1)}, {}
    for o in origenes:
        y_hist = y.iloc[:o]
        steps = min(h, n - o)
        if steps < 1:
            continue
        y_real = y.iloc[o:o + steps]
        try:
            pred = np.asarray(predictor(y_hist, steps), float)
            if not np.all(np.isfinite(pred)):
                raise ValueError("pronostico no finito")
        except Exception:
            pred = np.repeat(float(y_hist.iloc[-1]), steps)
        denom = _mase_denom(y_hist, m)
        met = _metricas(y_real.values, pred, denom)
        met["origen"] = y_hist.index[-1].strftime("%Y-%m")
        met["h"] = steps
        filas.append(met)
        trayectorias[y_hist.index[-1]] = pd.Series(pred, index=y_real.index)
        for k in range(steps):
            if denom and np.isfinite(denom) and denom > 0:
                err_h[k + 1].append(abs(y_real.values[k] - pred[k]) / denom)

    detalle = pd.DataFrame(filas)
    glob = {c: float(np.nanmean(detalle[c])) for c in
            ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "Accuracy"]}
    perfil = pd.Series({k: (np.mean(v) if v else np.nan) for k, v in err_h.items()},
                       name="MASE_por_horizonte")
    return glob, perfil, detalle, trayectorias


def backtest_walkforward_1paso(y, n_train, predictor):
    """Reestima el modelo cada mes y pronostica UN solo paso adelante
    (grafica de referencia; no decide el ganador)."""
    n_test = len(y) - n_train
    preds = []
    for i in range(n_test):
        y_hist = y.iloc[: n_train + i]
        try:
            p = predictor(y_hist, 1)
            val = float(np.asarray(p, float)[0])
            if not np.isfinite(val):
                raise ValueError("pronostico no finito")
        except Exception:
            val = float(y_hist.iloc[-1])
        preds.append(val)
    return np.array(preds)


def construir_candidatos(order, seasonal_order, m=12):
    """Construye el diccionario de variantes candidatas y las referencias
    (Naive estacional / Deriva) a partir del order estimado sobre train."""
    order_ns = (order[0], order[1], order[2])
    usa_estacional = bool(seasonal_order[3]) and (
        seasonal_order[0] or seasonal_order[1] or seasonal_order[2]
    )
    # Con d>=1, trend='t' equivale a introducir DERIVA en el nivel: es lo que
    # impide que el pronostico se aplane.
    trend_deriva = "t" if order[1] >= 1 else "c"

    candidatos = {}
    # --- Familia SARIMA (ordenes estimados automaticamente sobre train) ---
    if usa_estacional:
        candidatos["SARIMA"] = dict(kind="sarimax", order=order, sorder=seasonal_order, trend=None)
        candidatos["SARIMA + deriva"] = dict(kind="sarimax", order=order, sorder=seasonal_order, trend=trend_deriva)
        candidatos["SARIMA(0,1,1)(0,1,1)"] = dict(kind="sarimax", order=(0, 1, 1), sorder=(0, 1, 1, m), trend=None)
        candidatos["SARIMA(1,1,1)(1,1,0)"] = dict(kind="sarimax", order=(1, 1, 1), sorder=(1, 1, 0, m), trend=None)
        candidatos["SARIMA(1,1,0)(0,1,1)"] = dict(kind="sarimax", order=(1, 1, 0), sorder=(0, 1, 1, m), trend=None)
    else:
        candidatos["ARIMA auto"] = dict(kind="sarimax", order=order, sorder=(0, 0, 0, 0), trend=None)
        candidatos["ARIMA auto + deriva"] = dict(kind="sarimax", order=order, sorder=(0, 0, 0, 0), trend=trend_deriva)

    # --- Familia ARIMA no estacional ---
    if usa_estacional:
        candidatos["ARIMA (sin estacional)"] = dict(kind="sarimax", order=order_ns, sorder=(0, 0, 0, 0), trend=None)
        candidatos["ARIMA (sin estacional) + deriva"] = dict(kind="sarimax", order=order_ns, sorder=(0, 0, 0, 0), trend=trend_deriva)

    candidatos["ARIMA(0,1,1)"] = dict(kind="sarimax", order=(0, 1, 1), sorder=(0, 0, 0, 0), trend=None)
    candidatos["ARIMA(0,1,1) + deriva"] = dict(kind="sarimax", order=(0, 1, 1), sorder=(0, 0, 0, 0), trend="t")
    candidatos["ARIMA(1,1,0) + deriva"] = dict(kind="sarimax", order=(1, 1, 0), sorder=(0, 0, 0, 0), trend="t")
    candidatos["ARIMA(1,1,1)"] = dict(kind="sarimax", order=(1, 1, 1), sorder=(0, 0, 0, 0), trend=None)
    candidatos["ARIMA(1,1,1) + deriva"] = dict(kind="sarimax", order=(1, 1, 1), sorder=(0, 0, 0, 0), trend="t")
    candidatos["ARIMA(2,1,1) + deriva"] = dict(kind="sarimax", order=(2, 1, 1), sorder=(0, 0, 0, 0), trend="t")
    candidatos["ARIMA(1,1,2) + deriva"] = dict(kind="sarimax", order=(1, 1, 2), sorder=(0, 0, 0, 0), trend="t")

    # Referencias (NO compiten): el naive estacional es el denominador del MASE.
    referencias = {"Naive estacional": make_snaive(m), "Deriva (12m)": make_drift(12)}
    return candidatos, referencias, order_ns, usa_estacional


def construir_predictor(spec, m=12, use_log=USE_LOG):
    k = spec["kind"]
    if k == "sarimax":
        return make_sarimax_predictor(spec["order"], spec["sorder"], spec.get("trend"), use_log)
    if k == "ets":
        return make_ets_predictor(spec.get("damped", True), spec.get("seasonal", False), m, use_log)
    if k == "naive":
        return pred_naive
    if k == "snaive":
        return make_snaive(m)
    if k == "drift":
        return make_drift(spec.get("ventana"))
    raise ValueError(k)


# ======================================================================
# Persistencia opcional del modelo (Fase 14 del notebook)
# ======================================================================
def guardar_modelo(resultado: dict, ruta="modelo_final.pkl") -> Optional[str]:
    """Persiste el modelo ganador con joblib. Devuelve la ruta o None."""
    if not HAS_JOBLIB:
        return None
    payload = resultado.get("modelo", {})
    joblib.dump(payload, ruta)
    return ruta


# ======================================================================
# Pipeline completo: devuelve las 15 fases en orden
# ======================================================================
def ejecutar_pipeline(
    df: pd.DataFrame,
    producto: str,
    mercado: Optional[str],
    m: int = M,
    use_log: bool = USE_LOG,
    test_size=TEST_SIZE,
    h_futuro: int = H_FUTURO,
    n_origenes: int = N_ORIGENES,
) -> dict:
    """Ejecuta todo el proceso de modelado y devuelve un dict con:
        - 'fases'       : lista de objetos Fase (1..15) en orden
        - 'forecast'    : DataFrame con el pronostico final
        - 'comparativa' : DataFrame con las metricas de las variantes
        - 'ganador'     : nombre de la variante ganadora
        - 'resumen'     : dict con datos clave para la UI
        - 'modelo'      : payload persistible (spec, fit_full, order, etc.)
    """
    fases: list[Fase] = []
    mercado_label = "Colombia (promedio nacional)" if mercado is None else mercado

    # ---------------- Fase 1: creacion de la serie ---------------------
    f1 = Fase("Fase 1", "Creación de la serie")
    y = preparar_serie(df, producto, mercado)
    f1.log(
        f"Serie construida ({mercado_label}): {len(y)} observaciones, "
        f"de {y.index[0]:%Y-%m} a {y.index[-1]:%Y-%m}"
    )
    f1.tabla("Serie mensual (precio por kg)", y.round(2).to_frame("precio"))
    fases.append(f1)

    # ---------------- Fase 2: verificacion tamano serie ----------------
    f2 = Fase("Fase 2", "Verificación del tamaño de la serie")
    info_serie = verificar_registros(y, f2, m=m)
    fases.append(f2)

    # ---------------- Fase 3: EDA express ------------------------------
    f3 = Fase("Fase 3", "EDA express y estacionalidad")
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    y.plot(ax=ax[0], title="Serie (nivel)"); ax[0].grid(alpha=.3)
    np.log(y).plot(ax=ax[1], title="Serie (log)"); ax[1].grid(alpha=.3)
    if len(y) >= 2 * m:
        seasonal_decompose(y, model="additive", period=m).seasonal.plot(
            ax=ax[2], title="Estacionalidad"
        )
        ax[2].grid(alpha=.3)
    fig.tight_layout()
    f3.figura(fig)
    if len(y) >= 2 * m:
        est = seasonal_decompose(y, model="additive", period=m).seasonal
        f3.log(
            f"Amplitud estacional = {est.max() - est.min():,.1f} "
            f"({(est.max() - est.min()) / y.mean() * 100:.1f}% del nivel medio)"
        )
    fases.append(f3)

    # ---------------- Fase 4: train/test -------------------------------
    f4 = Fase("Fase 4", "División Train / Test")
    y_train, y_test = dividir_train_test(y, f4, test_size=test_size, m=m)
    fig, axd = plt.subplots(figsize=(11, 4))
    axd.plot(y_train.index, y_train.values, "o-", label="Train", color="#1f77b4")
    axd.plot(y_test.index, y_test.values, "o-", label="Test", color="#d62728")
    axd.axvline(y_train.index[-1], color="gray", ls=":", lw=1)
    axd.set_title("División Train / Test de la serie")
    axd.legend(); axd.grid(alpha=.3); fig.tight_layout()
    f4.figura(fig)
    fases.append(f4)

    # ---------------- Fase 5: verificacion train -----------------------
    f5 = Fase("Fase 5", "Verificación del tamaño y estacionalidad (train)")
    info = verificar_registros(y_train, f5, m=m, es_train=True)
    fases.append(f5)

    # ---------------- Fase 6: EDA y outliers (train) -------------------
    f6 = Fase("Fase 6", "EDA y detección de valores atípicos (train)")
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    y_train.plot(ax=ax[0], title="Serie de Train (nivel)"); ax[0].grid(alpha=.3)
    np.log(y_train).plot(ax=ax[1], title="Serie de Train (log)"); ax[1].grid(alpha=.3)
    if len(y_train) >= 2 * m:
        seasonal_decompose(y_train, model="additive", period=m).seasonal.plot(
            ax=ax[2], title="Estacionalidad"
        )
        ax[2].grid(alpha=.3)
    fig.tight_layout()
    f6.figura(fig)

    atipicos_train = detectar_outliers(y_train)
    if len(atipicos_train):
        f6.log("Meses con variación atípica en TRAIN (posibles choques):")
        for fecha, delta in atipicos_train.items():
            f6.log(f"  {fecha:%Y-%m}: variación = {delta:+,.1f}")
    else:
        f6.log("No se detectaron variaciones atípicas relevantes en train.")

    atipicos_test = detectar_outliers(pd.concat([y_train.tail(m), y_test]))
    atipicos_test = atipicos_test[atipicos_test.index.isin(y_test.index)]
    if len(atipicos_test):
        f6.log("\nAVISO: el TEST tiene meses con variación atípica (el error puede subir):")
        for fecha, delta in atipicos_test.items():
            f6.log(f"  {fecha:%Y-%m}: variación = {delta:+,.1f}")
    fases.append(f6)

    # ---------------- Fase 7: estacionariedad y diferenciacion ---------
    f7 = Fase("Fase 7", "Estacionariedad y diferenciación (train)")
    y_mod_train = np.log(y_train) if use_log else y_train.copy()
    test_estacionariedad(y_mod_train, f7, "nivel (train)")
    test_estacionariedad(y_mod_train.diff(), f7, "d=1 (train)")
    D_ORDER, D_SEAS = estimar_dD(y_mod_train, info, f7, m)
    fases.append(f7)

    # ---------------- Fase 8: identificacion ACF/PACF ------------------
    f8 = Fase("Fase 8", "Identificación (ACF / PACF, train)")
    yd = y_mod_train.copy()
    for _ in range(D_ORDER):
        yd = yd.diff()
    for _ in range(D_SEAS):
        yd = yd.diff(m)
    yd = yd.dropna()
    n_eff = len(yd)
    lags = max(4, min(36, (n_eff // 2) - 1))
    f8.log(f"n efectivo tras diferenciar (train) = {n_eff} | lags = {lags}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(yd, lags=lags, ax=ax[0])
    plot_pacf(yd, lags=lags, method="ywm", ax=ax[1])
    fig.tight_layout()
    f8.figura(fig)
    if len(yd) > m:
        acf_m = pd.Series(yd).autocorr(lag=m)
        f8.log(
            f"ACF(lag {m}) = {acf_m:.3f}"
            + ("  <-- posible SOBRE-diferenciación estacional (revisa D)"
               if acf_m < -0.35 else "")
        )
    fases.append(f8)

    # ---------------- Fase 9: estimacion del modelo --------------------
    f9 = Fase("Fase 9", "Estimación del modelo (auto_arima / grid AIC)")
    order, seasonal_order = ajustar_sarima(y_mod_train, info, D_ORDER, D_SEAS, f9, m)
    f9.log(f"order = {order} | seasonal_order = {seasonal_order} | use_log = {use_log}")
    fases.append(f9)

    # ---------------- Fase 10: diagnostico de residuos -----------------
    f10 = Fase("Fase 10", "Diagnóstico de residuos (train)")
    fit_train = _fit(y_mod_train, order, seasonal_order)
    f10.log("Modelo estable (raíces AR fuera del círculo): " + str(modelo_estable(fit_train)))
    resid = pd.Series(fit_train.resid).replace([np.inf, -np.inf], np.nan).dropna()
    burn = order[1] + seasonal_order[1] * (seasonal_order[3] or 0)
    resid = resid.iloc[burn:]
    nr = len(resid)
    lags_r = max(2, min(24, (nr // 2) - 1))
    f10.log(f"residuos útiles = {nr} | lags diagnóstico = {lags_r}")
    if nr >= 8:
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        resid.plot(ax=ax[0, 0], title="Residuos"); ax[0, 0].grid(alpha=.3)
        ax[0, 1].hist(resid, bins=max(5, nr // 4)); ax[0, 1].set_title("Histograma")
        plot_acf(resid, lags=lags_r, ax=ax[1, 0])
        qqplot(resid, line="s", ax=ax[1, 1]); ax[1, 1].set_title("Q-Q")
        fig.tight_layout()
        f10.figura(fig)
        lb = acorr_ljungbox(resid, lags=[max(1, min(lags_r, 10))], return_df=True)
        f10.log("Ruido blanco (p>0.05): " + str(bool(lb["lb_pvalue"].iloc[0] > 0.05)))
        f10.log("Jarque-Bera p = " + str(round(stats.jarque_bera(resid)[1], 4)))
    else:
        f10.log("Muy pocos residuos para diagnóstico gráfico.")
    fases.append(f10)

    # ---------------- Fase 11: evaluacion en test (rolling-origin) -----
    f11 = Fase("Fase 11", "Evaluación en Test (backtest rolling-origin)")
    candidatos, referencias, order_ns, usa_estacional = construir_candidatos(
        order, seasonal_order, m
    )
    predictores = {n: construir_predictor(s, m, use_log) for n, s in candidatos.items()}

    H_TEST = len(y_test)
    H_EVAL = int(min(h_futuro, max(3, H_TEST)))
    MIN_TRAIN = max(2 * m, int(len(y) * 0.6))

    resultados, perfiles, curvas_test, detalles, trayectorias_wf = {}, {}, {}, {}, {}
    for nombre, pred_fn in predictores.items():
        try:
            glob, perfil, det, tray = backtest_rolling(
                y, pred_fn, H_EVAL, MIN_TRAIN, max_origenes=n_origenes, m=m
            )
            resultados[nombre] = glob
            perfiles[nombre] = perfil
            detalles[nombre] = det
            trayectorias_wf[nombre] = tray
        except Exception as e:
            f11.log(f"{nombre}: fallo en backtest ({e})")
            continue
        try:
            curva = np.asarray(pred_fn(y_train, H_TEST), float)
            curvas_test[nombre] = curva
            resultados[nombre]["Planitud"] = planitud(curva)
        except Exception:
            resultados[nombre]["Planitud"] = np.nan

    # Referencias: se evaluan aparte, no compiten por ser el ganador
    referencias_met = {}
    for nombre, pred_fn in referencias.items():
        try:
            g, _pf, _d, _tr = backtest_rolling(
                y, pred_fn, H_EVAL, MIN_TRAIN, max_origenes=n_origenes, m=m
            )
            referencias_met[nombre] = g
        except Exception:
            pass

    cols = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "Accuracy", "Planitud"]
    comparativa = pd.DataFrame(resultados).T[cols].sort_values("MASE")
    perfil_horizonte = pd.DataFrame(perfiles)

    n_orig = len(detalles[list(detalles)[0]]) if detalles else 0
    f11.log("Solo se estudian modelos de la familia ARIMA / SARIMA y sus variantes.")
    f11.log(f"=== BACKTEST ROLLING-ORIGIN (h={H_EVAL} meses, {n_orig} orígenes) ===")
    f11.tabla("Comparativa de variantes (rolling-origin)", comparativa.round(3))
    f11.tabla("MASE por horizonte (dónde se degrada cada modelo)", perfil_horizonte.round(3))

    # Walk-forward de 1 paso adelante (grafica de referencia, no decide ganador)
    n_train = len(y_train)
    curvas_walkforward = {}
    for nombre, pred_fn in predictores.items():
        try:
            curvas_walkforward[nombre] = backtest_walkforward_1paso(y, n_train, pred_fn)
        except Exception as e:
            f11.log(f"{nombre}: fallo en walk-forward 1 paso ({e})")
    f11.log(
        f"Walk-forward (1 paso adelante) calculado para {len(curvas_walkforward)} "
        f"variantes, sobre {len(y_test)} meses de test."
    )

    fig, axc = plt.subplots(figsize=(11.5, 5))
    axc.plot(y_train.index[-24:], y_train.values[-24:], "ko-", lw=1.5, alpha=.5,
             label="Train (últimos 24m)")
    axc.plot(y_test.index, y_test.values, "ko-", lw=2, label="Real (test)")
    for nom, pred in curvas_walkforward.items():
        axc.plot(y_test.index, pred, "--", marker="s", ms=4, alpha=.85, label=nom)
    axc.axvline(y_train.index[-1], color="gray", ls=":", lw=1)
    axc.set_title("Validación walk-forward (1 paso) — variantes SARIMA/ARIMA")
    axc.legend(fontsize=8, ncol=2); axc.grid(alpha=.3); fig.tight_layout()
    f11.figura(fig)
    fases.append(f11)

    # ---------------- Fase 12: seleccion de la ganadora ----------------
    f12 = Fase("Fase 12", "Selección de la variante ganadora")
    mov_real = planitud(y.iloc[-min(len(y), 24):].values)
    cand = comparativa.copy()
    cand["degenerado"] = (cand["Planitud"] < UMBRAL_PLANITUD) & (mov_real > 5 * UMBRAL_PLANITUD)

    crit = "MASE" if cand["MASE"].notna().any() else "MAE"
    utiles = cand[~cand["degenerado"]]

    if len(utiles):
        GANADOR = utiles[crit].idxmin()
        mejor_global = cand[crit].idxmin()
        if mejor_global != GANADOR:
            margen = (cand.loc[GANADOR, crit] - cand.loc[mejor_global, crit]) / max(
                1e-9, cand.loc[mejor_global, crit]
            )
            f12.log(
                f'AVISO: "{mejor_global}" tiene mejor {crit} pero produce un pronóstico '
                f'PLANO (variación {cand.loc[mejor_global, "Planitud"] * 100:.3f}% del nivel).'
            )
            f12.log(
                f"       Se descarta: la ventaja es de solo {margen * 100:.1f}% "
                "y no aporta información de trayectoria."
            )
    else:
        GANADOR = cand[crit].idxmin()
        f12.log("AVISO: todos los candidatos son planos; la serie parece una caminata aleatoria pura.")

    mase_g = float(cand.loc[GANADOR, "MASE"])
    acc_g = float(cand.loc[GANADOR, "Accuracy"])
    spec_g = candidatos[GANADOR]

    f12.log("=" * 68)
    f12.log(
        f"MEJOR VARIANTE: {GANADOR}   ({crit}={cand.loc[GANADOR, crit]:.3f} | "
        f"Accuracy={acc_g:.1f}% | variación del pronóstico="
        f"{cand.loc[GANADOR, 'Planitud'] * 100:.2f}%)"
    )
    f12.log("=" * 68)

    if np.isfinite(mase_g):
        if mase_g < 0.8:
            veredicto = "Aporta valor claro sobre el naive estacional (MASE<0.8)."
        elif mase_g < 1.0:
            veredicto = "Mejora marginal sobre el naive; útil pero con poco margen."
        else:
            veredicto = "NO supera al naive estacional: la serie es poco predecible a este horizonte."
        f12.log(f"MASE rolling-origin = {mase_g:.3f} -> {veredicto}")

    planos = cand.index[cand["degenerado"]].tolist()
    if planos:
        f12.log("Variantes descartadas por pronóstico plano: " + ", ".join(planos))
    fases.append(f12)

    # ---------------- Fase 13: reentrenamiento serie completa ----------
    f13 = Fase("Fase 13", "Reentrenamiento con la serie completa")
    y_mod_full = np.log(y) if use_log else y.copy()
    info_full = verificar_registros(y, None, m=m)

    TREND_FINAL = spec_g.get("trend")
    kind_final = spec_g["kind"]

    if kind_final == "sarimax":
        if GANADOR.startswith(("SARIMA", "ARIMA (sin estacional)")):
            d_f, D_f = estimar_dD(y_mod_full, info_full, None, m)
            if "sin estacional" in GANADOR:
                info_ns = dict(info_full)
                info_ns.update(seasonal_ok=False, max_P=0, max_Q=0)
                order_final, sorder_final = ajustar_sarima(y_mod_full, info_ns, d_f, 0, None, m)
            else:
                order_final, sorder_final = ajustar_sarima(y_mod_full, info_full, d_f, D_f, None, m)
            f13.log(f"Órdenes reestimados sobre la serie completa: {order_final} x {sorder_final}")
            if TREND_FINAL is not None:
                TREND_FINAL = "t" if order_final[1] >= 1 else "c"
        else:
            order_final, sorder_final = spec_g["order"], spec_g["sorder"]

        fit_full = SARIMAX(
            y_mod_full, order=order_final, seasonal_order=sorder_final,
            trend=TREND_FINAL, **FIT_KW
        ).fit(disp=False)

        if not modelo_estable(fit_full):
            f13.log("AVISO: modelo inestable con la serie completa -> fallback (0,1,1) con deriva")
            order_final, sorder_final, TREND_FINAL = (0, 1, 1), (0, 0, 0, 0), "t"
            fit_full = SARIMAX(
                y_mod_full, order=order_final, seasonal_order=sorder_final,
                trend=TREND_FINAL, **FIT_KW
            ).fit(disp=False)
        predictor_final = make_sarimax_predictor(order_final, sorder_final, TREND_FINAL, use_log)
    else:
        order_final, sorder_final, fit_full = None, None, None
        predictor_final = predictores[GANADOR]

    f13.log(
        f'Reentrenado "{GANADOR}" con la serie completa ({len(y)} meses): '
        f"order={order_final}, seasonal_order={sorder_final}, trend={TREND_FINAL}"
    )
    fases.append(f13)

    # ---------------- Fase 14: pronostico ------------------------------
    f14 = Fase("Fase 14", "Pronóstico a futuro")
    fut_idx = pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1), periods=h_futuro, freq="MS"
    )
    punto, lo, hi, _obj = predictor_final(y, h_futuro, want_full=True)
    punto = np.asarray(punto, float)

    # IC empirico cuando el modelo no lo provee (baselines / ETS)
    if lo is None or hi is None:
        perf = perfil_horizonte[GANADOR] if GANADOR in perfil_horizonte else None
        denom = _mase_denom(y, m)
        if perf is not None and np.isfinite(denom):
            esc = np.array([perf.get(k, np.nanmean(perf)) for k in range(1, h_futuro + 1)], float)
            esc = np.where(np.isfinite(esc), esc, np.nanmean(esc)) * denom * 1.96
        else:
            esc = np.full(h_futuro, np.nanstd(np.diff(y.values)) * 1.96) * np.sqrt(
                np.arange(1, h_futuro + 1)
            )
        lo, hi = punto - esc, punto + esc

    forecast_final = pd.DataFrame(
        {"pred": punto, "lo95": lo, "hi95": hi}, index=fut_idx
    )
    forecast_final["var_%_vs_mes_previo"] = forecast_final["pred"].pct_change().fillna(
        forecast_final["pred"].iloc[0] / float(y.iloc[-1]) - 1
    ) * 100
    forecast_final.index.name = "Mes"

    lo_h, hi_h = y.min() * 0.4, y.max() * 2.5
    fuera_rango = forecast_final[
        (forecast_final["pred"] < lo_h) | (forecast_final["pred"] > hi_h)
    ]
    f14.log(f"PRONÓSTICO {h_futuro} MESES  |  modelo: {GANADOR}  |  trend={TREND_FINAL}")
    f14.log(
        "Fuera de rango plausible: "
        + ("ninguno OK" if fuera_rango.empty
           else str(fuera_rango.index.strftime("%Y-%m").tolist()))
    )

    var_pred = planitud(punto)
    f14.log(
        f"Variación del pronóstico = {var_pred * 100:.2f}% del nivel "
        f"(serie real últimos 24m = {mov_real * 100:.2f}%)"
    )
    if var_pred < UMBRAL_PLANITUD:
        f14.log(
            "ATENCIÓN: el pronóstico sigue siendo prácticamente plano. Usa el IC 95% "
            "como rango de decisión."
        )
    else:
        f14.log("OK: el pronóstico tiene trayectoria (no es constante).")
    f14.tabla("Pronóstico detallado", forecast_final.round(2))

    fig, axf = plt.subplots(figsize=(12, 5))
    hist_plot = y.iloc[-min(len(y), 48):]
    axf.plot(hist_plot.index, hist_plot.values, "o-", color="#1f77b4", label="Histórico")
    axf.plot(forecast_final.index, forecast_final["pred"], "s--", color="#d62728",
             label=f"Pronóstico {h_futuro}m ({GANADOR})")
    axf.fill_between(forecast_final.index, forecast_final["lo95"], forecast_final["hi95"],
                     color="#d62728", alpha=.15, label="IC 95%")
    axf.axvline(y.index[-1], color="gray", ls=":", lw=1)
    axf.set_title(
        f"{producto} — {mercado_label}  |  próximos {h_futuro} meses  |  {GANADOR}  "
        f"(MASE={mase_g:.2f}, Accuracy={acc_g:.1f}%)"
    )
    axf.set_ylabel("Precio por kg"); axf.legend(); axf.grid(alpha=.3); fig.tight_layout()
    f14.figura(fig)
    fases.append(f14)

    # ---------------- Fase 15: resultados finales ----------------------
    f15 = Fase("Fase 15", "Resultados finales y resumen del proceso")
    f15.log("=" * 60)
    f15.log("RESUMEN DE RESULTADOS")
    f15.log("=" * 60)
    f15.log(
        f"Serie                  : {producto} en {mercado_label}  "
        f"(n={len(y)} meses, {len(y) / m:.1f} ciclos)"
    )
    f15.log(f"Train / Test           : {len(y_train)} / {len(y_test)} meses")
    f15.log(
        f"Estacionalidad         : Fs={info['Fs']} -> "
        + ("usada" if info["seasonal_ok"] else "NO usada (señal débil)")
    )
    f15.log(f"Modelo Fase 9 (train)  : SARIMA{order}x{seasonal_order}")
    f15.log("-" * 60)
    f15.log(
        f"MEJOR VARIANTE          : {GANADOR}   (MASE={mase_g:.3f}, "
        f"Accuracy={acc_g:.1f}%, MAE={comparativa.loc[GANADOR, 'MAE']:,.1f}, "
        f"sMAPE={comparativa.loc[GANADOR, 'sMAPE']:.1f}%)"
    )
    f15.log(f"Último observado {y.index[-1]:%Y-%m} : {y.iloc[-1]:,.2f}")
    f15.log(
        f"Pronóstico {forecast_final.index[0]:%Y-%m}       : "
        f"{forecast_final['pred'].iloc[0]:,.2f} "
        f"[{forecast_final['lo95'].iloc[0]:,.2f} ; {forecast_final['hi95'].iloc[0]:,.2f}]"
    )
    f15.log(
        f"Pronóstico a {h_futuro} meses ({forecast_final.index[-1]:%Y-%m}) : "
        f"{forecast_final['pred'].iloc[-1]:,.2f}"
    )
    f15.log("=" * 60)
    f15.tabla("Métricas en TEST (rolling-origin)", comparativa.round(3))

    tabla_pronostico = forecast_final.copy()
    tabla_pronostico.index = tabla_pronostico.index.strftime("%Y-%m")
    tabla_pronostico.index.name = "Mes"
    tabla_pronostico = tabla_pronostico.rename(
        columns={
            "pred": "Pronóstico",
            "lo95": "Límite inferior 95%",
            "hi95": "Límite superior 95%",
            "var_%_vs_mes_previo": "Variación % vs mes previo",
        }
    ).round(2)
    f15.tabla(
        f"Tabla de pronóstico — próximos {h_futuro} meses ({producto}, {mercado_label})",
        tabla_pronostico,
    )
    f15.figura(_figura_resumen(y, forecast_final, producto, mercado_label, GANADOR,
                               mase_g, acc_g, h_futuro))
    fases.append(f15)

    resumen = dict(
        producto=producto,
        mercado=mercado_label,
        ganador=GANADOR,
        mase=mase_g,
        accuracy=acc_g,
        ultimo_observado=float(y.iloc[-1]),
        ultima_fecha=y.index[-1].strftime("%Y-%m"),
        pred_prox_mes=float(forecast_final["pred"].iloc[0]),
        pred_prox_mes_fecha=forecast_final.index[0].strftime("%Y-%m"),
        pred_ultimo=float(forecast_final["pred"].iloc[-1]),
        pred_ultimo_fecha=forecast_final.index[-1].strftime("%Y-%m"),
        n_obs=len(y),
    )

    modelo = dict(
        ganador=GANADOR,
        spec=spec_g,
        fit_full=fit_full,           # None si el ganador es baseline/ETS
        order=order_final,
        seasonal_order=sorder_final,
        trend=TREND_FINAL,
        use_log=use_log,
        m=m,
        info=info_full,
        metricas_backtest=comparativa.loc[GANADOR].to_dict(),
        perfil_horizonte=perfil_horizonte[GANADOR].to_dict()
        if GANADOR in perfil_horizonte else None,
        producto=producto,
        mercado=mercado,
    )

    return dict(
        fases=fases,
        forecast=forecast_final,
        comparativa=comparativa,
        perfil_horizonte=perfil_horizonte,
        referencias=referencias_met,
        ganador=GANADOR,
        resumen=resumen,
        modelo=modelo,
    )


def _figura_resumen(y, forecast_final, producto, mercado_label, ganador,
                    mase_g, acc_g, h_futuro):
    """Grafica sintetica para encabezar la Fase 15."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_plot = y.iloc[-min(len(y), 48):]
    ax.plot(hist_plot.index, hist_plot.values, "o-", color="#1f77b4", label="Histórico")
    ax.plot(forecast_final.index, forecast_final["pred"], "s--", color="#16a34a",
            label=f"Pronóstico {h_futuro}m ({ganador})")
    ax.fill_between(forecast_final.index, forecast_final["lo95"], forecast_final["hi95"],
                    color="#16a34a", alpha=.15, label="IC 95%")
    ax.axvline(y.index[-1], color="gray", ls=":", lw=1)
    ax.set_title(
        f"{producto} — {mercado_label}  |  {ganador}  "
        f"(MASE={mase_g:.2f}, Accuracy={acc_g:.1f}%)"
    )
    ax.set_ylabel("Precio por kg"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    return fig
