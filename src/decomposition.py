"""
decomposition.py
================
Classical time series decomposition into Trend + Seasonality + Residual.

Implements STL-lite using:
  - Trend: centred moving average (Hodrick-Prescott optional)
  - Seasonality: periodic averaging
  - Residual: series − trend − seasonality
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DecompositionResult:
    original: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int
    model: str  # 'additive' or 'multiplicative'

    # Summary statistics
    @property
    def trend_strength(self) -> float:
        """
        Wang et al. (2006) trend strength: how much variance is explained by trend.
        Range [0, 1] — higher = stronger trend.
        """
        var_resid = self.residual.var()
        var_trend_plus_resid = (self.trend.dropna() + self.residual.dropna()).var()
        if var_trend_plus_resid < 1e-12:
            return 0.0
        return float(max(0, 1 - var_resid / var_trend_plus_resid))

    @property
    def seasonal_strength(self) -> float:
        """Seasonal strength: analogous to trend strength."""
        var_resid = self.residual.var()
        var_seas_plus_resid = (self.seasonal.dropna() + self.residual.dropna()).var()
        if var_seas_plus_resid < 1e-12:
            return 0.0
        return float(max(0, 1 - var_resid / var_seas_plus_resid))

    def summary(self) -> str:
        return (
            f"Decomposition ({self.model}, period={self.period})\n"
            f"  Trend strength   : {self.trend_strength:.3f}\n"
            f"  Seasonal strength: {self.seasonal_strength:.3f}\n"
            f"  Residual std     : {self.residual.std():.6f}\n"
        )


def classical_decompose(
    series: pd.Series,
    period: int = 252,
    model: str = "additive",
) -> DecompositionResult:
    """
    Classical decomposition using centred moving average trend.

    Parameters
    ----------
    series : pd.Series  — time series (prices or returns)
    period : int        — seasonal period (252 = annual for daily data)
    model  : str        — 'additive' (y = T + S + R) or 'multiplicative' (y = T × S × R)

    Returns
    -------
    DecompositionResult
    """
    y = series.copy()
    model = model.lower()

    # --- Step 1: Trend via centred moving average ---
    trend = _centred_ma(y, period)

    # --- Step 2: De-trend ---
    if model == "additive":
        detrended = y - trend
    else:
        detrended = y / trend.replace(0, np.nan)

    # --- Step 3: Seasonal component by averaging within each period ---
    seasonal_raw = detrended.groupby(np.arange(len(detrended)) % period).transform(
        "mean"
    )
    seasonal = pd.Series(seasonal_raw.values, index=y.index, name="seasonal")

    # Normalise seasonal: additive → mean=0, multiplicative → mean=1
    if model == "additive":
        seasonal -= seasonal.mean()
    else:
        seasonal /= seasonal.mean()

    # --- Step 4: Residual ---
    if model == "additive":
        residual = y - trend - seasonal
    else:
        residual = y / (trend * seasonal).replace(0, np.nan)

    return DecompositionResult(
        original=y,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
        model=model,
    )


def _centred_ma(series: pd.Series, window: int) -> pd.Series:
    """Centred (two-sided) moving average for even and odd windows."""
    if window % 2 == 1:
        trend = series.rolling(window=window, center=True, min_periods=1).mean()
    else:
        # For even window: average of two offset half-windows
        half = window // 2
        trend = (
            series.rolling(window=window, min_periods=1).mean().shift(-half)
            + series.rolling(window=window, min_periods=1).mean().shift(-half + 1)
        ) / 2
    trend.name = "trend"
    return trend


def hodrick_prescott(
    series: pd.Series,
    lamb: float = 1600.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Hodrick-Prescott filter.

    For daily financial data, typical lambda values:
      - 6.25       → annual
      - 129600     → daily (standard)
      - 1600       → quarterly convention

    Returns
    -------
    (trend, cycle) : both pd.Series
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    # Build second-difference matrix D
    # HP minimises: Σ(y-τ)² + λ Σ(Δ²τ)²
    # → (I + λ K'K) τ = y   where K = second-difference matrix
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    e = np.ones(n)
    D = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
    KtK = D.T @ D
    A = sparse.eye(n, format="csc") + lamb * KtK.tocsc()
    trend_arr = spsolve(A, y)
    cycle_arr = y - trend_arr

    trend = pd.Series(trend_arr, index=series.index, name="hp_trend")
    cycle = pd.Series(cycle_arr, index=series.index, name="hp_cycle")
    return trend, cycle
