"""
statistical_tests.py
====================
Professional-grade statistical tests for financial time series.

Implements ADF, KPSS, ACF, and PACF using only numpy/scipy,
without depending on statsmodels — maximum portability.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes for clean, typed results
# ---------------------------------------------------------------------------


@dataclass
class ADFResult:
    """Augmented Dickey-Fuller test result."""

    statistic: float
    pvalue: float
    n_lags: int
    n_obs: int
    critical_values: dict[str, float]
    is_stationary: bool  # reject H0 (unit root) at 5%
    interpretation: str

    def summary(self) -> str:
        sig = "✅ STATIONARY" if self.is_stationary else "❌ NON-STATIONARY"
        return (
            f"ADF Test Result: {sig}\n"
            f"  Statistic  : {self.statistic:.4f}\n"
            f"  p-value    : {self.pvalue:.4f}\n"
            f"  Lags used  : {self.n_lags}\n"
            f"  Critical values: { {k: f'{v:.3f}' for k,v in self.critical_values.items()} }\n"
            f"  {self.interpretation}"
        )


@dataclass
class KPSSResult:
    """KPSS stationarity test result."""

    statistic: float
    pvalue: float
    n_lags: int
    critical_values: dict[str, float]
    is_stationary: bool  # fail to reject H0 (stationary) at 5%
    interpretation: str

    def summary(self) -> str:
        sig = "✅ STATIONARY" if self.is_stationary else "❌ NON-STATIONARY"
        return (
            f"KPSS Test Result: {sig}\n"
            f"  Statistic  : {self.statistic:.4f}\n"
            f"  p-value    : {self.pvalue:.4f}\n"
            f"  Lags used  : {self.n_lags}\n"
            f"  Critical values: { {k: f'{v:.3f}' for k,v in self.critical_values.items()} }\n"
            f"  {self.interpretation}"
        )


@dataclass
class CorrelogramResult:
    """ACF / PACF result container."""

    lags: np.ndarray
    values: np.ndarray
    conf_band: float  # ±1.96/√n
    significant_lags: list[int]  # lags outside confidence band
    kind: str = "ACF"  # "ACF" or "PACF"


# ---------------------------------------------------------------------------
# ADF Test (custom implementation)
# ---------------------------------------------------------------------------


def adf_test(series: pd.Series | np.ndarray, max_lags: int | None = None) -> ADFResult:
    """
    Augmented Dickey-Fuller test.

    H0: series has a unit root (non-stationary)
    H1: series is stationary (no unit root)

    Reject H0 when t-statistic < critical value (or p-value < 0.05).

    Implementation follows Said & Dickey (1984).
    Lag selection via AIC (Akaike Information Criterion).
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)

    if max_lags is None:
        # Schwert (1989) rule of thumb
        max_lags = int(np.ceil(12 * (n / 100) ** 0.25))

    best_aic = np.inf
    best_lag = 0

    for lag in range(0, max_lags + 1):
        aic = _adf_aic(y, lag)
        if aic < best_aic:
            best_aic = aic
            best_lag = lag

    stat, pval, nobs = _adf_statistic(y, best_lag)

    # MacKinnon (1994) approximate critical values for constant+trend model
    cv = {
        "1%": -3.430,
        "5%": -2.862,
        "10%": -2.567,
    }

    is_stationary = pval < 0.05

    interp = (
        "The series IS stationary — safe to use directly in models."
        if is_stationary
        else (
            "The series has a unit root. Consider differencing (I(1) process) "
            "or log-transformation before modelling."
        )
    )

    return ADFResult(
        statistic=stat,
        pvalue=pval,
        n_lags=best_lag,
        n_obs=nobs,
        critical_values=cv,
        is_stationary=is_stationary,
        interpretation=interp,
    )


def _adf_statistic(y: np.ndarray, n_lags: int) -> tuple[float, float, int]:
    """Compute the ADF t-statistic for a given lag order."""
    dy = np.diff(y)
    n = len(dy)

    # Build regressor matrix: [Δy_{t-1}, Δy_{t-2}, …, Δy_{t-p}, y_{t-1}, 1]
    rows = n - n_lags
    X = np.ones((rows, n_lags + 2))

    # Lagged level (unit-root regressor)
    X[:, 0] = y[n_lags : n_lags + rows]

    # Augmentation lags
    for k in range(1, n_lags + 1):
        X[:, k] = dy[n_lags - k : n_lags - k + rows]

    # Constant already in last column
    Y = dy[n_lags : n_lags + rows]

    # OLS
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, rows

    Y_hat = X @ beta
    resid = Y - Y_hat
    sigma2 = np.sum(resid**2) / (rows - n_lags - 2)
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se_beta0 = np.sqrt(cov_beta[0, 0])

    t_stat = (beta[0] - 1) / se_beta0 if se_beta0 > 1e-10 else 0.0

    # MacKinnon (1994) response surface p-value approximation
    pval = _mackinnon_pvalue(t_stat, nobs=rows)

    return float(t_stat), float(pval), rows


def _adf_aic(y: np.ndarray, n_lags: int) -> float:
    """AIC for lag selection in ADF."""
    _, resid_ss, _, rows = _adf_regression_info(y, n_lags)
    k = n_lags + 2
    if rows <= k or resid_ss <= 0:
        return np.inf
    sigma2 = resid_ss / rows
    log_lik = -rows / 2 * (np.log(2 * np.pi * sigma2) + 1)
    return -2 * log_lik + 2 * k


def _adf_regression_info(y: np.ndarray, n_lags: int):
    dy = np.diff(y)
    n = len(dy)
    rows = n - n_lags
    if rows <= 0:
        return None, np.inf, None, rows

    X = np.ones((rows, n_lags + 2))
    X[:, 0] = y[n_lags : n_lags + rows]
    for k in range(1, n_lags + 1):
        X[:, k] = dy[n_lags - k : n_lags - k + rows]
    Y = dy[n_lags : n_lags + rows]

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return beta, float(np.sum(resid**2)), resid, rows


def _mackinnon_pvalue(t_stat: float, nobs: int) -> float:
    """
    Approximate MacKinnon (1994) p-value for ADF using
    a logistic sigmoid fit to the response surface.
    This is a practical approximation sufficient for finance.
    """
    # Response surface coefficients (intercept model, tau_1)
    # from MacKinnon (1994) Table 1
    cv_table = {
        0.01: -3.43035,
        0.025: -3.10726,
        0.05: -2.86154,
        0.10: -2.56677,
        0.90: 0.44053,
        0.975: 1.83990,
    }

    # Interpolate p-value
    cvs = np.array(list(cv_table.values()))
    ps = np.array(list(cv_table.keys()))
    # Sort by critical value
    order = np.argsort(cvs)
    cvs, ps = cvs[order], ps[order]

    if t_stat <= cvs[0]:
        return ps[0]
    if t_stat >= cvs[-1]:
        return 1.0
    return float(np.interp(t_stat, cvs, ps))


# ---------------------------------------------------------------------------
# KPSS Test (custom implementation)
# ---------------------------------------------------------------------------


def kpss_test(series: pd.Series | np.ndarray, regression: str = "c") -> KPSSResult:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test.

    H0: series is stationary (trend-stationary)
    H1: series has a unit root

    Reject H0 (conclude non-stationary) when statistic > critical value.
    This is the COMPLEMENT of ADF — using both provides stronger evidence.

    Parameters
    ----------
    series     : time series
    regression : 'c'  → test for level stationarity
                 'ct' → test for trend stationarity
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)

    # De-mean (or de-trend) the series
    t = np.arange(1, n + 1)
    if regression == "c":
        X = np.column_stack([np.ones(n)])
    else:
        X = np.column_stack([np.ones(n), t])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # Partial sums (cumulative sum of residuals)
    S = np.cumsum(resid)

    # Long-run variance estimate (Newey-West with Bartlett kernel)
    n_lags = int(np.ceil(4 * (n / 100) ** 0.25))
    sigma2 = _newey_west_variance(resid, n_lags)

    # KPSS statistic
    kpss_stat = np.sum(S**2) / (n**2 * sigma2)

    # Critical values (Kwiatkowski et al. 1992, Table 1)
    if regression == "c":
        cv = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    else:
        cv = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}

    # Approximate p-value by interpolation
    pval = _kpss_pvalue(kpss_stat, regression)

    is_stationary = kpss_stat < cv["5%"]

    interp = (
        "Cannot reject H0 — series IS stationary (KPSS)."
        if is_stationary
        else (
            "Reject H0 — series is NOT stationary (KPSS). "
            "Evidence of a unit root or structural break."
        )
    )

    return KPSSResult(
        statistic=float(kpss_stat),
        pvalue=float(pval),
        n_lags=n_lags,
        critical_values=cv,
        is_stationary=is_stationary,
        interpretation=interp,
    )


def _newey_west_variance(resid: np.ndarray, n_lags: int) -> float:
    """Newey-West heteroskedasticity and autocorrelation consistent variance."""
    n = len(resid)
    gamma0 = np.dot(resid, resid) / n
    long_run = gamma0

    for lag in range(1, n_lags + 1):
        gamma_lag = np.dot(resid[lag:], resid[:-lag]) / n
        bartlett_weight = 1 - lag / (n_lags + 1)
        long_run += 2 * bartlett_weight * gamma_lag

    return max(long_run, 1e-12)


def _kpss_pvalue(stat: float, regression: str) -> float:
    """Approximate p-value for KPSS by interpolation from critical values."""
    if regression == "c":
        cvs = np.array([0.347, 0.463, 0.574, 0.739])
        ps = np.array([0.10, 0.05, 0.025, 0.01])
    else:
        cvs = np.array([0.119, 0.146, 0.176, 0.216])
        ps = np.array([0.10, 0.05, 0.025, 0.01])

    if stat < cvs[0]:
        return 0.10
    if stat > cvs[-1]:
        return 0.01
    return float(np.interp(stat, cvs, ps))


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------


def acf(
    series: pd.Series | np.ndarray,
    n_lags: int = 40,
    alpha: float = 0.05,
) -> CorrelogramResult:
    """
    Sample Autocorrelation Function.

    ρ(k) = Cov(y_t, y_{t-k}) / Var(y_t)

    Confidence band: ±z_{α/2} / √n  (Bartlett's approximation)
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    y_dm = y - y.mean()

    gamma0 = np.dot(y_dm, y_dm) / n
    acf_values = np.zeros(n_lags + 1)
    acf_values[0] = 1.0

    for k in range(1, n_lags + 1):
        acf_values[k] = np.dot(y_dm[k:], y_dm[:-k]) / (n * gamma0)

    conf = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    lags = np.arange(n_lags + 1)
    sig_lags = [k for k in range(1, n_lags + 1) if abs(acf_values[k]) > conf]

    return CorrelogramResult(
        lags=lags,
        values=acf_values,
        conf_band=conf,
        significant_lags=sig_lags,
        kind="ACF",
    )


def pacf(
    series: pd.Series | np.ndarray,
    n_lags: int = 40,
    alpha: float = 0.05,
) -> CorrelogramResult:
    """
    Partial Autocorrelation Function via Yule-Walker equations.

    φ_{kk} = partial correlation between y_t and y_{t-k}
    after removing the linear effect of intermediate lags.
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)

    # Build ACF values for Yule-Walker
    acf_res = acf(series, n_lags=n_lags)
    r = acf_res.values[1:]  # r[0] = acf at lag 1, etc.

    pacf_values = np.zeros(n_lags + 1)
    pacf_values[0] = 1.0

    # Levinson-Durbin recursion
    phi = np.zeros(n_lags)
    phi[0] = r[0]
    pacf_values[1] = r[0]

    for k in range(1, n_lags):
        # New partial correlation coefficient
        num = r[k] - np.dot(phi[:k], r[k - 1 :: -1][:k])
        den = 1.0 - np.dot(phi[:k], r[:k])
        if abs(den) < 1e-12:
            break
        phi_kk = num / den
        pacf_values[k + 1] = phi_kk
        # Update previous coefficients
        phi_new = phi[:k] - phi_kk * phi[k - 1 :: -1][:k]
        phi[:k] = phi_new
        phi[k] = phi_kk

    conf = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    lags = np.arange(n_lags + 1)
    sig_lags = [k for k in range(1, n_lags + 1) if abs(pacf_values[k]) > conf]

    return CorrelogramResult(
        lags=lags,
        values=pacf_values,
        conf_band=conf,
        significant_lags=sig_lags,
        kind="PACF",
    )


# ---------------------------------------------------------------------------
# Stationarity diagnosis combining ADF + KPSS
# ---------------------------------------------------------------------------


def stationarity_diagnosis(series: pd.Series, label: str = "Series") -> dict:
    """
    Run ADF + KPSS and produce a joint verdict following the
    standard decision matrix:

    ADF: reject H0 | KPSS: fail to reject H0 → ✅ Stationary
    ADF: fail H0   | KPSS: reject H0         → ❌ Unit root (difference)
    ADF: reject H0 | KPSS: reject H0         → ⚠️  Trend-stationary (de-trend)
    ADF: fail H0   | KPSS: fail H0           → ⚠️  Inconclusive (structural break?)
    """
    adf = adf_test(series)
    kpss = kpss_test(series)

    adf_reject = adf.is_stationary  # ADF rejects unit-root H0
    kpss_reject = not kpss.is_stationary  # KPSS rejects stationarity H0

    if adf_reject and not kpss_reject:
        verdict = "✅ STATIONARY — both tests agree. Ready for ARMA/GARCH modelling."
    elif not adf_reject and kpss_reject:
        verdict = "❌ UNIT ROOT — first-difference the series before modelling."
    elif adf_reject and kpss_reject:
        verdict = "⚠️  TREND-STATIONARY — consider de-trending or using ARIMA with d=0."
    else:
        verdict = "⚠️  INCONCLUSIVE — possible structural break. Use Zivot-Andrews test."

    return {
        "label": label,
        "adf": adf,
        "kpss": kpss,
        "verdict": verdict,
    }
