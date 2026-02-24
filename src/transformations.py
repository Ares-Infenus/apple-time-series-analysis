"""
transformations.py
==================
Feature engineering for financial time series.
All transformations are pure functions (no side effects) — easy to unit-test.
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series, col: str = "log_return") -> pd.Series:
    """
    Compute continuously compounded (log) returns.
    r_t = ln(P_t / P_{t-1})

    Log-returns are additive across time, making them preferable
    to simple returns for statistical analysis.
    """
    lr = np.log(prices / prices.shift(1)).dropna()
    lr.name = col
    return lr


def simple_returns(prices: pd.Series) -> pd.Series:
    """Arithmetic (simple) returns: (P_t - P_{t-1}) / P_{t-1}"""
    r = prices.pct_change().dropna()
    r.name = "simple_return"
    return r


def rolling_statistics(
    series: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling mean, std (volatility), Sharpe proxy, and z-score
    for multiple window lengths.

    Parameters
    ----------
    series  : log returns or price series
    windows : list of integer window lengths (trading days)

    Returns
    -------
    pd.DataFrame with all rolling stats
    """
    if windows is None:
        windows = [5, 21, 63, 252]  # week, month, quarter, year

    frames = [series.rename("value")]
    for w in windows:
        roll = series.rolling(w, min_periods=max(w // 2, 2))
        frames.append(roll.mean().rename(f"mean_{w}d"))
        frames.append(roll.std().rename(f"vol_{w}d"))
        # Annualised volatility
        frames.append((roll.std() * np.sqrt(252)).rename(f"ann_vol_{w}d"))

    df = pd.concat(frames, axis=1)
    # Rolling z-score (useful for mean-reversion signals)
    for w in windows:
        df[f"zscore_{w}d"] = (series - df[f"mean_{w}d"]) / df[f"vol_{w}d"]
    return df


def realised_volatility(
    log_rets: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """
    Realised (historical) volatility: rolling standard deviation of log returns.
    Annualised by default (×√252).
    """
    rv = log_rets.rolling(window, min_periods=window // 2).std()
    if annualise:
        rv *= np.sqrt(252)
    rv.name = f"realised_vol_{window}d"
    return rv


def bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands: middle band ± n_std × rolling std.
    Useful as a visual volatility envelope.
    """
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + n_std * std,
            "bb_lower": mid - n_std * std,
            "bb_width": (2 * n_std * std) / mid,  # normalised width
        }
    )


def ewma_volatility(
    log_rets: pd.Series,
    span: int = 30,
) -> pd.Series:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics-style).
    Gives more weight to recent observations.
    """
    ewv = log_rets.ewm(span=span, adjust=False).std() * np.sqrt(252)
    ewv.name = f"ewma_vol_span{span}"
    return ewv


def descriptive_stats(series: pd.Series) -> pd.Series:
    """
    Full distributional summary: mean, std, skewness, excess kurtosis,
    min/max, VaR 5%, CVaR 5%, Jarque-Bera p-value.
    """
    from scipy import stats

    s = series.dropna()
    jb_stat, jb_p = stats.jarque_bera(s)

    var_5 = np.percentile(s, 5)
    cvar_5 = s[s <= var_5].mean()

    return pd.Series(
        {
            "n_obs": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "skewness": s.skew(),
            "excess_kurtosis": s.kurtosis(),
            "min": s.min(),
            "max": s.max(),
            "var_5pct": var_5,
            "cvar_5pct": cvar_5,
            "jb_stat": jb_stat,
            "jb_pvalue": jb_p,
            "is_normal_jb": jb_p > 0.05,
        }
    )
