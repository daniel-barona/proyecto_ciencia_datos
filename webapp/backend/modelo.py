# -*- coding: utf-8 -*-
"""
Motor de prediccion SIPSA (SARIMA/ARIMA) adaptado del notebook

Mantiene exactamente las 8 fases del notebook:
  Fase 1: Creacion de la serie
  Fase 2: Verificacion de la serie y division Train / Test
  Fase 3: Exploracion, estacionariedad e identificacion (train)
  Fase 4: Estimacion del modelo y diagnostico de residuos (train)
  Fase 5: Evaluacion en Test (backtest rolling-origin) + metricas
  Fase 6: Seleccion del modelo ganador y validacion visual
  Fase 7: Reentrenamiento con la serie completa
  Fase 8: Pronostico a futuro + resumen final

Optimizado para deploy: ajuste SARIMAX con lbfgs (rapido) y powell solo
como reintento; STL/descomposiciones reutilizadas entre fases.
"""

from __future__ import annotations

import gc
import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from config import TRUSTED_DIR
except ImportError:
    try:
        from backend.config import TRUSTED_DIR
    except ImportError:
        TRUSTED_DIR = Path(__file__).resolve().parent / "data" / "trusted"

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

try:
    import pmdarima as pm
    from pmdarima.arima.utils import ndiffs, nsdiffs
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False


# Constantes del modelo (mismas del notebook, optimizadas para deploy)

M = 12
TEST_SIZE = 12
H_FUTURO = 12
USE_LOG = True

UMBRAL_PLANITUD = 0.005
UMBRAL_ESTACION = 0.004
AMPLITUD_MINIMA = 0.01
TOLERANCIA = 0.10
N_ORIGENES = 1  # minimo para ahorrar CPU

TRUSTED_FILE = "SIPSA_2013_2026_trusted.parquet"

# Optimizacion: lbfgs converge mucho mas rapido que powell; powell queda
# solo como reintento unico cuando lbfgs no entrega parametros finitos.
FIT_KW = dict(enforce_stationarity=True, enforce_invertibility=True)
MAXITER = 50
MAXITER_FALLBACK = 100



# Fase dataclass


@dataclass
class Fase:
    id: str
    titulo: str
    logs: list = field(default_factory=list)
    tablas: list = field(default_factory=list)
    figuras: list = field(default_factory=list)

    def log(self, *partes) -> None:
        self.logs.append(" ".join(str(p) for p in partes))

    def tabla(self, titulo: str, df) -> None:
        self.tablas.append((titulo, df))

    def figura(self, fig) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        self.figuras.append(buf.getvalue())

    @property
    def texto(self) -> str:
        return "\n".join(self.logs)

# Carga de datos y selectores en cascada

def cargar_trusted(trusted_path: Optional[Path] = None) -> pd.DataFrame:
    ruta = Path(trusted_path) if trusted_path else (TRUSTED_DIR / TRUSTED_FILE)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo trusted en: {ruta}\n"
            "Ejecuta primero el backend de descarga/consolidacion/limpieza "
            "para generar la capa trusted."
        )
    df = pd.read_parquet(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _anio_referencia(df: pd.DataFrame) -> int:
    return int(df["fecha"].dt.year.max())


def opciones_departamentos(df: pd.DataFrame) -> list:
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
    if mercado is None:
        base = df[df["fecha"].dt.year == anio]
    else:
        base = df[(df["mercado"] == mercado) & (df["fecha"].dt.year == anio)]
    return sorted(base["producto"].dropna().unique().tolist())



# Funciones auxiliares

def preparar_serie(df, producto, mercado=None):
    base = df[df['producto'] == producto]
    if mercado is not None:
        base = base[base['mercado'] == mercado]
    serie = base[['fecha', 'precio_promedio_kg']].copy()
    serie = serie.sort_values('fecha')
    serie.set_index('fecha', inplace=True)
    serie.rename(columns={'precio_promedio_kg': 'precio'}, inplace=True)
    serie = serie.groupby(serie.index).mean()
    serie = serie.asfreq('MS').interpolate('linear').ffill().bfill()
    return serie['precio']


def fuerza_estacional(y, m=12, stl_res=None):
    if len(y) < 2 * m + 1:
        return np.nan
    try:
        res = stl_res if stl_res is not None else STL(y, period=m, robust=True).fit()
    except Exception:
        return np.nan
    var_r = np.nanvar(res.resid)
    var_rs = np.nanvar(res.resid + res.seasonal)
    return float(max(0.0, 1 - var_r / var_rs)) if var_rs > 0 else 0.0


def fuerza_tendencia(y, m=12, stl_res=None):
    if len(y) < 2 * m + 1:
        return np.nan
    try:
        res = stl_res if stl_res is not None else STL(y, period=m, robust=True).fit()
    except Exception:
        return np.nan
    var_r = np.nanvar(res.resid)
    var_rt = np.nanvar(res.resid + res.trend)
    return float(max(0.0, 1 - var_r / var_rt)) if var_rt > 0 else 0.0


def verificar_registros(y, m=12, etiqueta=''):
    n = int(y.dropna().shape[0])
    ciclos = n / m
    nivel = ('INSUFICIENTE' if n < 24 else 'MINIMO' if n < 36 else
             'ACEPTABLE' if n < 48 else 'BUENO' if n < 72 else 'OPTIMO')

    stl_res = None
    if len(y) >= 2 * m + 1:
        try:
            stl_res = STL(y, period=m, robust=True).fit()
        except Exception:
            stl_res = None
    Fs = fuerza_estacional(y, m, stl_res=stl_res)
    Ft = fuerza_tendencia(y, m, stl_res=stl_res)
    seasonal_ok = bool(n >= 2 * m + 1)
    seasonal_fuerte = bool(seasonal_ok and np.isfinite(Fs) and Fs >= 0.30)
    tendencia_ok = bool(np.isfinite(Ft) and Ft >= 0.30)

    presupuesto = max(4, n // 8)
    max_p = 3 if n >= 60 else 2 if n >= 36 else 1
    max_q = max_p
    max_P = 1 if seasonal_ok else 0
    max_Q = 1 if (seasonal_ok and n >= 3 * m) else 0
    h_bt = int(min(m, max(6, n // 5), max(3, n - max(3 * m, 24))))

    info = dict(
        n=n, ciclos=round(ciclos, 2), nivel=nivel, apto=n >= 24, m=m,
        Fs=None if not np.isfinite(Fs) else round(Fs, 3),
        Ft=None if not np.isfinite(Ft) else round(Ft, 3),
        seasonal_ok=seasonal_ok, seasonal_fuerte=seasonal_fuerte,
        seasonal_debil=bool(seasonal_ok and not seasonal_fuerte),
        tendencia_ok=tendencia_ok,
        max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
        presupuesto=presupuesto, h_bt=h_bt,
    )
    return info


def dividir_train_test(y, test_size=None, m=12, min_train=None):
    n = len(y)
    min_train = min_train or max(3 * m, 24)
    if test_size is None:
        test_size = int(min(m, max(6, n // 5), max(3, n - min_train)))
    test_size = max(1, min(test_size, n - min_train)) if n > min_train else max(1, n // 5)
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]
    return y_train, y_test


def detectar_outliers(y_serie, umbral=3.5):
    diffs = y_serie.diff().dropna()
    mediana = diffs.median()
    mad = (diffs - mediana).abs().median()
    if mad == 0:
        return pd.Series(dtype=float)
    z = 0.6745 * (diffs - mediana) / mad
    return diffs[z.abs() > umbral]


def test_estacionariedad(x, nombre=''):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InterpolationWarning)
        adf = adfuller(x)[1]
        kps = kpss(x, nlags='auto')[1]
    return adf, kps


def estimar_dD(ys, info, m=12):
    if HAS_PMDARIMA:
        d = int(min(2, ndiffs(ys, test='kpss', max_d=2)))
    else:
        adf = test_estacionariedad(ys, '_')[0]
        d = 0 if (adf is not None and adf < 0.05) else 1

    D = 0
    if info['seasonal_ok']:
        D = 1
        if HAS_PMDARIMA and not info['seasonal_fuerte']:
            try:
                D = int(min(1, nsdiffs(ys, m=m, max_D=1, test='ocsb')))
            except Exception:
                D = 1
        if len(ys) - D * m - d < int(1.5 * m):
            D = 0
    return d, D


def _sanear_sorder(sorder, D, m=12, seasonal=True):
    if not seasonal:
        return (0, 0, 0, 0)
    P, Dd, Q, s = (list(sorder) + [0, 0, 0, 0])[:4]
    s = m
    Dd = int(Dd if Dd else D)
    if P == 0 and Q == 0 and Dd == 0:
        Q = 1
    return (int(P), int(Dd), int(Q), int(s))


def _fit(ys, order, sorder, trend=None, maxiter=None):
    mi = MAXITER if maxiter is None else int(maxiter)
    kw = dict(FIT_KW, low_memory=True)
    try:
        f = SARIMAX(
            ys, order=order, seasonal_order=sorder, trend=trend, **kw
        ).fit(disp=False, maxiter=mi, method='lbfgs')
        if np.all(np.isfinite(f.params)):
            return f
    except Exception:
        pass
    return SARIMAX(
        ys, order=order, seasonal_order=sorder, trend=trend, **kw
    ).fit(disp=False, maxiter=MAXITER_FALLBACK, method='powell')


def modelo_estable(fit):
    try:
        ar = np.concatenate(
            [np.atleast_1d(fit.arroots), np.atleast_1d(getattr(fit, 'seasonalarroots', []))]
        )
        return bool(len(ar) == 0 or np.all(np.abs(ar) > 1.01))
    except Exception:
        return True


def ajustar_sarima(ys, info, d, D, m=12, verbose=True, fase=None):
    seasonal = bool(info['seasonal_ok'])
    max_p, max_q = info['max_p'], info['max_q']
    max_P = max(1, info['max_P']) if seasonal else 0
    max_Q = max(1, info['max_Q']) if seasonal else 0

    order, sorder = (1, d, 0), _sanear_sorder((1, D, 0, m), D, m, seasonal)
    if HAS_PMDARIMA:
        try:
            mdl = pm.auto_arima(
                ys, seasonal=seasonal, m=m if seasonal else 1,
                d=d, D=D if seasonal else 0,
                start_p=0, start_q=0, start_P=0, start_Q=1,
                max_p=min(1, max_p), max_q=min(1, max_q), max_P=min(1, max_P), max_Q=min(1, max_Q),
                information_criterion='aicc',
                stationary=False, stepwise=True, maxiter=30,
                suppress_warnings=True, error_action='ignore', trace=False,
            )
            order = mdl.order
            sorder = _sanear_sorder(mdl.seasonal_order, D, m, seasonal)
            if seasonal and sorder[0] == 0 and sorder[2] == 0 and sorder[1] == 0:
                D_forzado = max(1, D)
                sorder = (0, D_forzado, 1, m)
                if fase is not None:
                    fase.log(f'auto_arima sin terminos estacionales -> forzado sorder={sorder}')
        except Exception as e:
            if fase is not None:
                fase.log('auto_arima fallo, uso fallback:', e)
    else:
        mejor, best = None, np.inf
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        so = _sanear_sorder((P, D, Q, m), D, m, seasonal)
                        try:
                            f = _fit(ys, (p, d, q), so)
                            if f.aicc < best and modelo_estable(f):
                                best, mejor = f.aicc, ((p, d, q), so)
                        except Exception:
                            continue
        if mejor:
            order, sorder = mejor

    k = order[0] + order[2] + sorder[0] + sorder[2]
    if k > info['presupuesto']:
        if fase is not None:
            fase.log(f'Modelo simplificado: k={k} > presupuesto={info["presupuesto"]}')
        order = (1, d, 0)
        sorder = _sanear_sorder((1, D, 0, m), D, m, seasonal)

    return order, sorder


def _inv_log(mu, var=None, use_log=True):
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


def make_sarimax_predictor(order, sorder, trend=None, maxiter=None):
    _m = MAXITER if maxiter is None else int(maxiter)
    def _p(y_hist, steps, want_full=False):
        ym = np.log(y_hist) if USE_LOG else y_hist
        f = _fit(ym, order, sorder, trend, maxiter=_m)
        fc = f.get_forecast(steps=steps)
        try:
            var = np.asarray(fc.var_pred_mean, float)
        except Exception:
            var = None
        punto = _inv_log(fc.predicted_mean.values, var, USE_LOG)
        if not want_full:
            return punto
        ci = fc.conf_int(alpha=0.05)
        lo = np.exp(ci.iloc[:, 0].values) if USE_LOG else ci.iloc[:, 0].values
        hi = np.exp(ci.iloc[:, 1].values) if USE_LOG else ci.iloc[:, 1].values
        return punto, lo, hi, f
    return _p


def make_snaive(m=12):
    def _p(y_hist, steps, want_full=False):
        last = np.asarray(y_hist.iloc[-m:], float)
        p = np.array([last[i % len(last)] for i in range(steps)], float)
        return (p, None, None, None) if want_full else p
    return _p


def make_drift(ventana=None):
    def _p(y_hist, steps, want_full=False):
        v = len(y_hist) if ventana is None else min(ventana, len(y_hist))
        z = y_hist.iloc[-v:]
        slope = (float(z.iloc[-1]) - float(z.iloc[0])) / max(1, len(z) - 1)
        p = float(y_hist.iloc[-1]) + slope * np.arange(1, steps + 1)
        return (p, None, None, None) if want_full else p
    return _p


def backtest_rolling(y, predictor, h, min_train, max_origenes=6, m=12):
    n = len(y)
    ultimos = n - h
    if ultimos < min_train:
        min_train = max(m + 2, int(n * 0.6))
    origenes = [o for o in range(min_train, ultimos + 1)]
    if not origenes:
        origenes = [max(m + 2, n - max(1, h // 2))]
    if len(origenes) > max_origenes:
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
                raise ValueError('pronostico no finito')
        except Exception:
            pred = np.repeat(float(y_hist.iloc[-1]), steps)
        denom = _mase_denom(y_hist, m)
        met = _metricas(y_real.values, pred, denom)
        met['origen'] = y_hist.index[-1].strftime('%Y-%m')
        met['h'] = steps
        filas.append(met)
        trayectorias[y_hist.index[-1]] = pd.Series(pred, index=y_real.index)
        for k in range(steps):
            if denom and np.isfinite(denom) and denom > 0:
                err_h[k + 1].append(abs(y_real.values[k] - pred[k]) / denom)

    detalle = pd.DataFrame(filas)
    glob = {c: float(np.nanmean(detalle[c])) for c in
            ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'Accuracy']}
    perfil = pd.Series({k: (np.mean(v) if v else np.nan) for k, v in err_h.items()},
                       name='MASE_por_horizonte')
    return glob, perfil, detalle, trayectorias


def planitud(pred):
    p = np.asarray(pred, float)
    return float((p.max() - p.min()) / max(1e-9, np.mean(np.abs(p))))


def amplitud_estacional_pred(pred, m=12):
    p = np.asarray(pred, float)
    if len(p) < 3:
        return np.nan
    x = np.arange(len(p))
    coef = np.polyfit(x, p, 1)
    resid = p - np.polyval(coef, x)
    return float(np.std(resid) / max(1e-9, np.mean(np.abs(p))))


def pendiente_rel(pred):
    p = np.asarray(pred, float)
    if len(p) < 2:
        return np.nan
    x = np.arange(len(p))
    m_, b_ = np.polyfit(x, p, 1)
    return float(m_ * (len(p) - 1) / max(1e-9, np.mean(np.abs(p))))


def construir_predictor(spec, maxiter=None):
    k = spec['kind']
    if k == 'sarimax':
        return make_sarimax_predictor(spec['order'], spec['sorder'], spec.get('trend'), maxiter=maxiter)
    if k == 'snaive':
        return make_snaive(M)
    if k == 'drift':
        return make_drift(spec.get('ventana'))
    raise ValueError(k)


def backtest_walkforward_1paso(y, n_train, predictor):
    n_test = len(y) - n_train
    preds = []
    for i in range(n_test):
        y_hist = y.iloc[: n_train + i]
        try:
            p = predictor(y_hist, 1)
            val = float(np.asarray(p, float)[0])
            if not np.isfinite(val):
                raise ValueError('pronostico no finito')
        except Exception:
            val = float(y_hist.iloc[-1])
        preds.append(val)
    return np.array(preds)


def guardar_modelo(resultado: dict, ruta="modelo_final.pkl") -> Optional[str]:
    if not HAS_JOBLIB:
        return None
    payload = resultado.get("modelo", {})
    joblib.dump(payload, ruta)
    return ruta


def _figura_resumen(y, forecast_final, producto, mercado_label, ganador,
                    mase_g, acc_g, h_futuro):
    """Grafica sintetica para encabezar el resumen final (Fase 8)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hist_plot = y.iloc[-min(len(y), 48):]
    ax.plot(hist_plot.index, hist_plot.values, 'o-', color='#1f77b4', label='Historico')
    ax.plot(forecast_final.index, forecast_final['Pronostico'], 's--',
            color='#16a34a', label=f'Pronostico {h_futuro}m ({ganador})')
    ax.fill_between(
        forecast_final.index, forecast_final['IC_95_Lo'], forecast_final['IC_95_Hi'],
        color='#16a34a', alpha=.15, label='IC 95%'
    )
    ax.axvline(y.index[-1], color='gray', ls=':', lw=1)
    ax.set_title(
        f'{producto} - {mercado_label}  |  {ganador}  '
        f'(MASE={mase_g:.2f}, Accuracy={acc_g:.1f}%)'
    )
    ax.set_ylabel('Precio por kg')
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


# ======================================================================
# Pipeline completo - 8 fases exactas del notebook
# ======================================================================
def ejecutar_pipeline(
    df: pd.DataFrame,
    producto: str,
    mercado: Optional[str],
    m: int = M,
    test_size: int = TEST_SIZE,
    h_futuro: int = H_FUTURO,
    n_origenes: int = N_ORIGENES,
    backtest_maxiter: int = 30,
) -> dict:
    fases: list[Fase] = []
    mercado_label = 'Colombia (promedio nacional)' if mercado is None else mercado

    # ================================================================
    # Fase 1: Creacion de la serie
    # ================================================================
    f1 = Fase("Fase 1", "Creacion de la serie")
    y = preparar_serie(df, producto, mercado)
    f1.log(
        f'Serie construida ({mercado_label}): {len(y)} observaciones, '
        f'de {y.index[0]:%Y-%m} a {y.index[-1]:%Y-%m}'
    )
    f1.tabla("Serie mensual (precio por kg)", y.round(2).to_frame("precio"))
    fases.append(f1)

    # ================================================================
    # Fase 2: Verificacion de la serie y division Train / Test
    # ================================================================
    f2 = Fase("Fase 2", "Verificacion de la serie y division Train / Test")

    f2.log('--- Serie completa (referencia) ---')
    info_full_ref = verificar_registros(y, m=m, etiqueta='  ')
    f2.log(f"  Registros            : {info_full_ref['n']}")
    f2.log(f"  Periodo estacional m : {m}  (fijo, ciclo anual)")
    f2.log(f"  Ciclos (m={m})        : {info_full_ref['ciclos']:.2f}")
    f2.log(f"  Nivel                : {info_full_ref['nivel']}")
    f2.log(f"  Fuerza estacional Fs : {info_full_ref['Fs']}  ({'fuerte' if info_full_ref['seasonal_fuerte'] else 'moderada/debil'})")
    f2.log(f"  Fuerza tendencia  Ft : {info_full_ref['Ft']}  ({'con tendencia' if info_full_ref['tendencia_ok'] else 'sin tendencia clara'})")
    f2.log(f"  Bloque estacional    : {'SI (P,D,Q,12)' if info_full_ref['seasonal_ok'] else 'NO (menos de 2 ciclos)'}")
    f2.log(f"  Limites busqueda     : p<={info_full_ref['max_p']} q<={info_full_ref['max_q']} P<={info_full_ref['max_P']} Q<={info_full_ref['max_Q']} (max {info_full_ref['presupuesto']} params)")

    y_train, y_test = dividir_train_test(y, test_size=test_size, m=m)
    f2.log(f'  Serie total   : {len(y)} observaciones')
    f2.log(
        f'  Train         : {len(y_train)} observaciones '
        f'({y_train.index[0]:%Y-%m} a {y_train.index[-1]:%Y-%m})'
    )
    f2.log(
        f'  Test          : {len(y_test)} observaciones '
        f'({y_test.index[0]:%Y-%m} a {y_test.index[-1]:%Y-%m})'
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    y_train.plot(ax=ax, label='Train', color='steelblue')
    y_test.plot(ax=ax, label='Test', color='tomato')
    ax.set_title('Division Train / Test')
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    f2.figura(fig)

    f2.log('')
    f2.log('--- Train (rige la busqueda de ordenes; el test queda fuera para no filtrarlo) ---')
    info = verificar_registros(y_train, m=m, etiqueta='  ')
    f2.log(f"  Registros            : {info['n']}")
    f2.log(f"  Periodo estacional m : {m}  (fijo, ciclo anual)")
    f2.log(f"  Ciclos (m={m})        : {info['ciclos']:.2f}")
    f2.log(f"  Nivel                : {info['nivel']}")
    f2.log(f"  Fuerza estacional Fs : {info['Fs']}  ({'fuerte' if info['seasonal_fuerte'] else 'moderada/debil'})")
    f2.log(f"  Fuerza tendencia  Ft : {info['Ft']}  ({'con tendencia' if info['tendencia_ok'] else 'sin tendencia clara'})")
    f2.log(f"  Bloque estacional    : {'SI (P,D,Q,12)' if info['seasonal_ok'] else 'NO (menos de 2 ciclos)'}")
    f2.log(f"  Limites busqueda     : p<={info['max_p']} q<={info['max_q']} P<={info['max_P']} Q<={info['max_Q']} (max {info['presupuesto']} params)")
    f2.log(f"  Horizonte backtest   : {info['h_bt']} meses")
    fases.append(f2)

    # ================================================================
    # Fase 3: Exploracion, estacionariedad e identificacion (train)
    # ================================================================
    f3 = Fase("Fase 3", "Exploracion, estacionariedad e identificacion (train)")

    fig, ax = plt.subplots(3, 1, figsize=(10, 7))
    y_train.plot(ax=ax[0], title='Serie de Train (nivel)')
    ax[0].grid(alpha=.3)
    np.log(y_train).plot(ax=ax[1], title='Serie de Train (log)')
    ax[1].grid(alpha=.3)
    if len(y_train) >= 2 * m:
        decomp = seasonal_decompose(y_train, model='additive', period=m)
        decomp.seasonal.plot(ax=ax[2], title='Estacionalidad')
        ax[2].grid(alpha=.3)
    plt.tight_layout()
    f3.figura(fig)

    if len(y_train) >= 2 * m:
        est = decomp.seasonal
        f3.log(
            f'Amplitud estacional = {est.max() - est.min():,.1f} '
            f'({(est.max() - est.min()) / y_train.mean() * 100:.1f}% del nivel medio)'
        )

    atipicos_train = detectar_outliers(y_train)
    if len(atipicos_train):
        f3.log('Meses con variacion atipica en TRAIN (posibles choques puntuales):')
        for fecha, delta in atipicos_train.items():
            f3.log(f'  {fecha:%Y-%m}: variacion = {delta:+,.1f}')
    else:
        f3.log('No se detectaron variaciones atipicas relevantes en train.')

    atipicos_test = detectar_outliers(pd.concat([y_train.tail(m), y_test]))
    atipicos_test = atipicos_test[atipicos_test.index.isin(y_test.index)]
    if len(atipicos_test):
        f3.log('')
        f3.log('AVISO: el conjunto de TEST tiene meses con variacion atipica; es esperable que el error del modelo suba ahi:')
        for fecha, delta in atipicos_test.items():
            f3.log(f'  {fecha:%Y-%m}: variacion = {delta:+,.1f}')

    y_mod_train = np.log(y_train) if USE_LOG else y_train.copy()

    adf_val, kps_val = test_estacionariedad(y_mod_train, 'nivel (train)')
    f3.log(f'nivel (train): ADF p={adf_val:.3f}  KPSS p={kps_val:.3f}')
    adf_val, kps_val = test_estacionariedad(y_mod_train.diff(), 'd=1 (train)')
    f3.log(f'd=1 (train): ADF p={adf_val:.3f}  KPSS p={kps_val:.3f}')
    if len(y_mod_train) > m + 4:
        adf_val, kps_val = test_estacionariedad(y_mod_train.diff(m), 'D=1 estacional (train)')
        f3.log(f'D=1 estacional (train): ADF p={adf_val:.3f}  KPSS p={kps_val:.3f}')

    D_ORDER, D_SEAS = estimar_dD(y_mod_train, info, m=m)
    f3.log(f'Diferenciacion estimada (train) -> d={D_ORDER}, D={D_SEAS} (m={m})')
    if info['seasonal_ok'] and D_SEAS == 0:
        f3.log('  (D=0: la estacionalidad se modelara con terminos AR/MA estacionales P/Q en m=12)')

    yd = y_mod_train.copy()
    for _ in range(D_ORDER):
        yd = yd.diff()
    for _ in range(D_SEAS):
        yd = yd.diff(m)
    yd = yd.dropna()
    n_eff = len(yd)
    lags = max(4, min(36, (n_eff // 2) - 1))
    f3.log(f'n efectivo tras diferenciar (train) = {n_eff} | lags = {lags}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_acf(yd, lags=lags, ax=ax[0])
    plot_pacf(yd, lags=lags, method='ywm', ax=ax[1])
    plt.tight_layout()
    f3.figura(fig)

    if len(yd) > m:
        acf_m = pd.Series(yd).autocorr(lag=m)
        f3.log(
            f'ACF(lag {m}) = {acf_m:.3f}'
            + ('  <-- posible SOBRE-diferenciacion estacional (revisa D)' if acf_m < -0.35 else '')
        )
    fases.append(f3)

    # ================================================================
    # Fase 4: Estimacion del modelo y diagnostico de residuos (train)
    # ================================================================
    f4 = Fase("Fase 4", "Estimacion del modelo y diagnostico de residuos (train)")

    order, seasonal_order = ajustar_sarima(y_mod_train, info, D_ORDER, D_SEAS, m=m, fase=f4)
    f4.log(f'order = {order} | seasonal_order = {seasonal_order} | use_log = {USE_LOG}')
    if seasonal_order[3] != m:
        f4.log(f'AVISO: se forzo el periodo estacional a m = {m}')
        seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], m)
    if info['seasonal_ok'] and seasonal_order[0] == 0 and seasonal_order[2] == 0 and seasonal_order[1] == 0:
        seasonal_order = (0, max(1, D_SEAS), 1, m)
        f4.log(f'AVISO: modelo sin terminos estacionales -> corregido a seasonal_order={seasonal_order}')

    fit_train = _fit(y_mod_train, order, seasonal_order)
    if not modelo_estable(fit_train):
        f4.log('Modelo inestable -> degradado a (0,d,1) con bloque estacional (0,D,1)')
        order = (0, D_ORDER, 1)
        seasonal_order = _sanear_sorder((0, D_SEAS, 1, m), D_SEAS, m, bool(info['seasonal_ok']))
        fit_train = _fit(y_mod_train, order, seasonal_order)
    f4.log(f'Modelo estable (raices AR fuera del circulo): {modelo_estable(fit_train)}')

    resid = pd.Series(fit_train.resid).replace([np.inf, -np.inf], np.nan).dropna()
    burn = order[1] + seasonal_order[1] * (seasonal_order[3] or 0)
    resid = resid.iloc[burn:]
    nr = len(resid)
    lags_r = max(2, min(24, (nr // 2) - 1))
    f4.log(f'residuos utiles = {nr} | lags diagnostico = {lags_r}')

    if nr >= 8:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        resid.plot(ax=ax[0, 0], title='Residuos')
        ax[0, 0].grid(alpha=.3)
        ax[0, 1].hist(resid, bins=max(5, nr // 4))
        ax[0, 1].set_title('Histograma')
        plot_acf(resid, lags=lags_r, ax=ax[1, 0])
        qqplot(resid, line='s', ax=ax[1, 1])
        ax[1, 1].set_title('Q-Q')
        plt.tight_layout()
        f4.figura(fig)
        lb = acorr_ljungbox(resid, lags=[max(1, min(lags_r, 10))], return_df=True)
        f4.log(f'  Ljung-Box p={lb["lb_pvalue"].iloc[0]:.4f}')
        f4.log(f'  Ruido blanco (p>0.05): {bool(lb["lb_pvalue"].iloc[0] > 0.05)}')
        f4.log(f'  Jarque-Bera p = {round(stats.jarque_bera(resid)[1], 4)}')
    else:
        f4.log('Muy pocos residuos para diagnostico grafico.')
    fases.append(f4)

    # ================================================================
    # Fase 5: Evaluacion en Test (backtest rolling-origin) + metricas
    # ================================================================
    f5 = Fase("Fase 5", "Evaluacion en Test (backtest rolling-origin) + metricas")

    H_TEST = len(y_test)
    H_EVAL = int(min(h_futuro, max(3, H_TEST)))
    MIN_TRAIN = max(2 * m + 2, int(len(y) * 0.6))

    order_ns = (order[0], order[1], order[2])
    d_o = order[1]
    D_s = int(seasonal_order[1]) if seasonal_order[1] else (
        1 if info['seasonal_ok'] and len(y) >= 3 * m else 0
    )
    TREND_DERIVA = 't' if d_o >= 1 else 'c'

    def S(P, D_, Q):
        return (int(P), int(D_), int(Q), int(m))

    CANDIDATOS = {}
    if info['seasonal_ok']:
        CANDIDATOS['SARIMA auto'] = dict(kind='sarimax', order=order, sorder=S(*seasonal_order[:3]), trend=None)
        CANDIDATOS['SARIMA(0,1,1)(0,1,1)[12] + deriva'] = dict(kind='sarimax', order=(0, 1, 1), sorder=S(0, 1, 1), trend='t')
    CANDIDATOS['ARIMA auto (control)'] = dict(kind='sarimax', order=order_ns, sorder=(0, 0, 0, 0), trend=None)

    REFERENCIAS = {'Naive estacional': make_snaive(m), 'Deriva (12m)': make_drift(12)}

    PREDICTORES = {n: construir_predictor(s, maxiter=backtest_maxiter) for n, s in CANDIDATOS.items()}

    resultados, perfiles, curvas_test, detalles = {}, {}, {}, {}
    for nombre, pred_fn in PREDICTORES.items():
        try:
            glob, perfil, det, tray = backtest_rolling(
                y, pred_fn, H_EVAL, MIN_TRAIN, max_origenes=n_origenes, m=m
            )
            resultados[nombre] = glob
            perfiles[nombre] = perfil
            detalles[nombre] = det
            curvas_test[nombre] = list(tray.values())[-1] if tray else None
        except Exception as e:
            f5.log(f'{nombre}: fallo en backtest ({e})')
            continue
        try:
            ultima_tray = curvas_test[nombre]
            if ultima_tray is not None:
                resultados[nombre]['Planitud'] = planitud(ultima_tray.values)
                resultados[nombre]['Estacion'] = amplitud_estacional_pred(ultima_tray.values, m)
                resultados[nombre]['Tendencia'] = pendiente_rel(ultima_tray.values)
            else:
                raise ValueError
        except Exception:
            resultados[nombre]['Planitud'] = np.nan
            resultados[nombre]['Estacion'] = np.nan
            resultados[nombre]['Tendencia'] = np.nan
        resultados[nombre]['Estacional'] = bool(CANDIDATOS[nombre]['sorder'][3] == m)

    gc.collect()

    cols = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'Accuracy', 'Planitud', 'Estacion', 'Tendencia', 'Estacional']
    comparativa = pd.DataFrame(resultados).T[cols].sort_values('MASE')
    perfil_horizonte = pd.DataFrame(perfiles)

    n_orig = len(detalles[list(detalles)[0]]) if detalles else 0
    f5.log(f'Familia ARIMA / SARIMA con periodo estacional fijo m={m}.')
    f5.log(f'=== BACKTEST ROLLING-ORIGIN (h={H_EVAL} meses, {n_orig} origenes, '
           f'{len(CANDIDATOS)} candidatos) ===')
    f5.log('Planitud = variacion total del pronostico | Estacion = forma anual | Tendencia = pendiente relativa')
    f5.tabla("Comparativa de variantes (rolling-origin)", comparativa.round(3))
    f5.tabla("MASE por horizonte", perfil_horizonte.round(3))
    fases.append(f5)

    # ================================================================
    # Fase 6: Seleccion del modelo ganador y validacion visual
    # ================================================================
    f6 = Fase("Fase 6", "Seleccion del modelo ganador y validacion visual")

    mov_real = planitud(y.iloc[-min(len(y), 24):].values)

    cand = comparativa.copy()
    cand['degenerado'] = (cand['Planitud'] < UMBRAL_PLANITUD) & (mov_real > 5 * UMBRAL_PLANITUD)

    crit = 'MASE' if cand['MASE'].notna().any() else 'MAE'
    utiles = cand[~cand['degenerado']]
    if not len(utiles):
        utiles = cand
        f6.log('AVISO: todos los candidatos son planos; la serie parece una caminata aleatoria pura.')

    mejor_valor = float(utiles[crit].min())
    empate = utiles[utiles[crit] <= mejor_valor * (1 + TOLERANCIA)]

    preferidos = empate[(empate['Estacional'] == True) & (empate['Estacion'] >= UMBRAL_ESTACION)]
    if len(preferidos):
        GANADOR = preferidos[crit].idxmin()
    else:
        GANADOR = empate[crit].idxmin()

    if cand.loc[GANADOR, 'Estacion'] < AMPLITUD_MINIMA:
        con_amplitud = cand[(cand['Estacional'] == True) & (cand['Estacion'] >= AMPLITUD_MINIMA)]
        if len(con_amplitud):
            mejor_amp = con_amplitud[crit].idxmin()
            margen_amp = (cand.loc[mejor_amp, crit] - mejor_valor) / max(1e-9, mejor_valor)
            if margen_amp < 0.25:
                f6.log(
                    f'GANADOR "{GANADOR}" tenia amplitud={cand.loc[GANADOR, "Estacion"] * 100:.2f}% '
                    f'< {AMPLITUD_MINIMA * 100:.1f}% -> reemplazado por "{mejor_amp}" '
                    f'(amplitud={cand.loc[mejor_amp, "Estacion"] * 100:.2f}%, MASE +{margen_amp * 100:.1f}%)'
                )
                GANADOR = mejor_amp
        else:
            amplio = cand[crit] <= mejor_valor * 1.50
            con_amp_amplio = cand[amplio & (cand['Estacional'] == True) & (cand['Estacion'] >= AMPLITUD_MINIMA)]
            if len(con_amp_amplio):
                GANADOR = con_amp_amplio[crit].idxmin()
                f6.log(
                    f'GANADOR ajustado a "{GANADOR}" '
                    f'(tolerancia ampliada, amplitud={cand.loc[GANADOR, "Estacion"] * 100:.2f}%)'
                )

    mejor_global = cand[crit].idxmin()
    if mejor_global != GANADOR:
        margen = (cand.loc[GANADOR, crit] - cand.loc[mejor_global, crit]) / max(
            1e-9, cand.loc[mejor_global, crit]
        )
        f6.log(
            f'NOTA: "{mejor_global}" tiene el mejor {crit} bruto, pero se elige "{GANADOR}" '
            f'(solo {margen * 100:.1f}% peor) porque modela estacionalidad m={m} y tendencia.'
        )

    mase_g = float(cand.loc[GANADOR, 'MASE'])
    acc_g = float(cand.loc[GANADOR, 'Accuracy'])
    spec_g = CANDIDATOS[GANADOR]

    f6.log('=' * 74)
    f6.log(f'MEJOR VARIANTE: {GANADOR}')
    f6.log(
        f'  {crit}={cand.loc[GANADOR, crit]:.3f} | Accuracy={acc_g:.1f}% | '
        f'variacion={cand.loc[GANADOR, "Planitud"] * 100:.2f}% | '
        f'forma anual={cand.loc[GANADOR, "Estacion"] * 100:.2f}% | '
        f'tendencia={cand.loc[GANADOR, "Tendencia"] * 100:+.2f}%'
    )
    f6.log(
        f'  order={spec_g["order"]}  seasonal_order={spec_g["sorder"]}  trend={spec_g.get("trend")}'
    )
    f6.log('=' * 74)

    if np.isfinite(mase_g):
        if mase_g < 0.8:
            veredicto = 'Aporta valor claro sobre el naive estacional (MASE<0.8).'
        elif mase_g < 1.0:
            veredicto = 'Mejora marginal sobre el naive; util pero con poco margen.'
        else:
            veredicto = 'NO supera al naive estacional: la serie es poco predecible a este horizonte.'
        f6.log(f'MASE rolling-origin = {mase_g:.3f} -> {veredicto}')

    planos = cand.index[cand['degenerado']].tolist()
    if planos:
        f6.log('Variantes descartadas por pronostico plano: ' + ', '.join(planos))

    fig, axc = plt.subplots(figsize=(10, 4))
    axc.plot(y_train.index[-24:], y_train.values[-24:], 'ko-', lw=1.5, alpha=.5,
             label='Train (ult. 24m)')
    axc.plot(y_test.index, y_test.values, 'ko-', lw=2, label='Real (test)')
    if GANADOR in curvas_test and curvas_test[GANADOR] is not None:
        axc.plot(curvas_test[GANADOR].index, curvas_test[GANADOR].values, '--s', ms=4, alpha=.85, label=GANADOR)
    for nombre, pred_fn in REFERENCIAS.items():
        try:
            ref_pred = np.asarray(pred_fn(y_train, len(y_test)), float)
            axc.plot(y_test.index, ref_pred, ':', ms=3, alpha=.6, label=nombre)
        except Exception:
            pass
    axc.axvline(y_train.index[-1], color='gray', ls=':', lw=1)
    axc.set_title(f'Validacion en test -- modelo ganador: {GANADOR}')
    axc.legend(fontsize=8)
    axc.grid(alpha=.3)
    plt.tight_layout()
    f6.figura(fig)
    fases.append(f6)

    # ================================================================
    # Fase 7: Reentrenamiento con la serie completa
    # ================================================================
    f7 = Fase("Fase 7", "Reentrenamiento con la serie completa")

    y_mod_full = np.log(y) if USE_LOG else y.copy()
    info_full = info_full_ref  # ya calculado en la Fase 2 (evita otro STL robusto)

    order_final, sorder_final, TREND_FINAL = (
        spec_g['order'], spec_g['sorder'], spec_g.get('trend')
    )

    if info_full['seasonal_ok'] and sorder_final[3] not in (0, m):
        sorder_final = (sorder_final[0], sorder_final[1], sorder_final[2], m)
    if info_full['seasonal_ok']:
        if sorder_final[3] == 0:
            sorder_final = (sorder_final[0], sorder_final[1], sorder_final[2], m)
        if sorder_final[0] == 0 and sorder_final[2] == 0 and sorder_final[1] == 0:
            order_final = order_final if order_final[1] > 0 else (0, 1, 1)
            sorder_final = (0, 1, 1, m)
            TREND_FINAL = 't'
            f7.log(f'AVISO Fase 7: sin terminos estacionales -> forzado {order_final} {sorder_final} trend={TREND_FINAL}')

    def _ajustar(o, so, tr):
        return _fit(y_mod_full, o, so, tr)

    fit_full = _ajustar(order_final, sorder_final, TREND_FINAL)

    if not modelo_estable(fit_full):
        f7.log('AVISO: modelo inestable con la serie completa -> fallback estacional')
        if info_full['seasonal_ok']:
            order_final, sorder_final, TREND_FINAL = (0, 1, 1), (0, 1, 1, m), 't'
        else:
            order_final, sorder_final, TREND_FINAL = (0, 1, 1), (0, 0, 0, 0), 't'
        fit_full = _ajustar(order_final, sorder_final, TREND_FINAL)

    predictor_final = make_sarimax_predictor(order_final, sorder_final, TREND_FINAL)

    f7.log(
        f'Reentrenado "{GANADOR}" con la serie completa ({len(y)} meses): '
        f'order={order_final}, seasonal_order={sorder_final}, trend={TREND_FINAL}'
    )
    f7.log(
        f'Periodo estacional usado: m={sorder_final[3] if sorder_final[3] else "sin componente estacional"}'
    )
    fases.append(f7)

    # ================================================================
    # Fase 8: Pronostico a futuro
    # ================================================================
    f8 = Fase("Fase 8", "Pronostico a futuro")

    fut_idx = pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1), periods=h_futuro, freq='MS'
    )

    punto, lo, hi, _obj = predictor_final(y, h_futuro, want_full=True)
    punto = np.asarray(punto, float)

    if lo is None or hi is None:
        perf = perfil_horizonte[GANADOR] if GANADOR in perfil_horizonte else None
        denom = _mase_denom(y, m)
        if perf is not None and np.isfinite(denom):
            esc = np.array(
                [perf.get(k, np.nanmean(perf)) for k in range(1, h_futuro + 1)], float
            )
            esc = np.where(np.isfinite(esc), esc, np.nanmean(esc)) * denom * 1.96
        else:
            esc = (
                np.full(h_futuro, np.nanstd(np.diff(y.values)) * 1.96)
                * np.sqrt(np.arange(1, h_futuro + 1))
            )
        lo, hi = punto - esc, punto + esc

    forecast_final = pd.DataFrame({
        'Pronostico': punto,
        'IC_95_Lo': lo,
        'IC_95_Hi': hi,
    }, index=fut_idx)

    forecast_final['Cambio_$'] = forecast_final['Pronostico'].diff().fillna(
        forecast_final['Pronostico'].iloc[0] - float(y.iloc[-1])
    )

    forecast_final['Cambio_%'] = forecast_final['Pronostico'].pct_change().fillna(
        forecast_final['Pronostico'].iloc[0] / float(y.iloc[-1]) - 1
    ) * 100

    forecast_final.index.name = 'Mes'

    tabla_usuario = forecast_final[['Pronostico', 'Cambio_$', 'Cambio_%']].copy()
    tabla_usuario['Pronostico'] = tabla_usuario['Pronostico'].apply(lambda x: f'${x:,.0f}')
    tabla_usuario['Cambio_$'] = tabla_usuario['Cambio_$'].apply(lambda x: f'{x:+,.0f}')
    tabla_usuario['Cambio_%'] = tabla_usuario['Cambio_%'].apply(lambda x: f'{x:+.2f}%')

    f8.log(f'PRONOSTICO {h_futuro} MESES  |  modelo: {GANADOR}')
    f8.log(f'Producto: {producto}  |  Mercado: {mercado_label}')
    f8.log(f'Precio actual: ${float(y.iloc[-1]):,.0f}/kg')

    lo_h, hi_h = float(y.min()) * 0.4, float(y.max()) * 2.5
    fuera_rango = forecast_final[
        (forecast_final['Pronostico'] < lo_h) | (forecast_final['Pronostico'] > hi_h)
    ]
    f8.log(
        'Fuera de rango plausible: '
        + ('ninguno (OK)' if fuera_rango.empty
           else ', '.join(fuera_rango.index.strftime('%Y-%m')))
    )

    f8.tabla("Pronostico", tabla_usuario)

    tabla_detalle = forecast_final[['Pronostico', 'IC_95_Lo', 'IC_95_Hi']].copy()
    tabla_detalle.index = tabla_detalle.index.strftime('%Y-%m')
    tabla_detalle.columns = ['Pronostico ($/kg)', 'IC 95% inferior', 'IC 95% superior']
    f8.tabla('Pronostico detallado con intervalos', tabla_detalle.round(2))

    fig, axf = plt.subplots(figsize=(10, 4.5))
    hist_plot = y.iloc[-min(len(y), 48):]
    axf.plot(hist_plot.index, hist_plot.values, 'o-', color='#1f77b4', label='Historico')
    axf.plot(
        forecast_final.index, forecast_final['Pronostico'], 's--', color='#d62728',
        label=f'Pronostico {h_futuro}m'
    )
    axf.fill_between(
        forecast_final.index, forecast_final['IC_95_Lo'], forecast_final['IC_95_Hi'],
        color='#d62728', alpha=.15, label='IC 95%'
    )
    axf.axvline(y.index[-1], color='gray', ls=':', lw=1)
    axf.set_title(f'{producto} - {mercado_label}  |  proximos {h_futuro} meses')
    axf.set_ylabel('Precio por kg')
    axf.legend()
    axf.grid(alpha=.3)
    plt.tight_layout()
    f8.figura(fig)
    f8.figura(_figura_resumen(
        y, forecast_final, producto, mercado_label, GANADOR, mase_g, acc_g, h_futuro
    ))

    f8.log('')
    f8.log('=' * 60)
    f8.log('RESUMEN DE RESULTADOS')
    f8.log('=' * 60)
    f8.log(
        f'Serie                  : {producto} en {mercado_label} '
        f'(n={len(y)} meses, {len(y) / m:.1f} ciclos)'
    )
    f8.log(f'Train / Test           : {len(y_train)} / {len(y_test)} meses')
    f8.log(
        f'Estacionalidad         : Fs={info["Fs"]} -> '
        + ('usada' if info['seasonal_ok'] else 'NO usada (senal debil)')
    )
    f8.log(
        f'Modelo ganador         : {GANADOR} '
        f'(order={spec_g["order"]}, seasonal={spec_g["sorder"]})'
    )
    f8.log(
        f'Metricas test          : MASE={mase_g:.3f} | Accuracy={acc_g:.1f}% | '
        f'MAE={float(comparativa.loc[GANADOR, "MAE"]):,.1f}'
    )
    f8.log(f'Ultimo observado {y.index[-1]:%Y-%m} : ${float(y.iloc[-1]):,.2f}/kg')
    f8.log(
        f'Pronostico {forecast_final.index[0]:%Y-%m}       : '
        f'${punto[0]:,.2f} [${lo[0]:,.2f} ; ${hi[0]:,.2f}]'
    )
    f8.log(
        f'Pronostico a {h_futuro} meses ({forecast_final.index[-1]:%Y-%m}) : '
        f'${punto[-1]:,.2f}/kg'
    )
    f8.log('=' * 60)
    fases.append(f8)

    gc.collect()

    # ================================================================
    # Construccion del resumen y payload del modelo
    # ================================================================
    resumen = dict(
        producto=producto,
        mercado=mercado_label,
        ganador=GANADOR,
        mase=mase_g,
        accuracy=acc_g,
        ultimo_observado=float(y.iloc[-1]),
        ultima_fecha=y.index[-1].strftime('%Y-%m'),
        pred_prox_mes=float(forecast_final['Pronostico'].iloc[0]),
        pred_prox_mes_fecha=forecast_final.index[0].strftime('%Y-%m'),
        pred_ultimo=float(forecast_final['Pronostico'].iloc[-1]),
        pred_ultimo_fecha=forecast_final.index[-1].strftime('%Y-%m'),
        n_obs=len(y),
    )

    modelo = dict(
        ganador=GANADOR,
        spec=spec_g,
        fit_full=fit_full,
        order=order_final,
        seasonal_order=sorder_final,
        trend=TREND_FINAL,
        use_log=USE_LOG,
        m=m,
        info=info_full,
        metricas_backtest=comparativa.loc[GANADOR].to_dict(),
        perfil_horizonte=(
            perfil_horizonte[GANADOR].to_dict()
            if GANADOR in perfil_horizonte else None
        ),
        producto=producto,
        mercado=mercado,
    )

    return dict(
        fases=fases,
        forecast=forecast_final,
        comparativa=comparativa,
        ganador=GANADOR,
        resumen=resumen,
        modelo=modelo,
    )