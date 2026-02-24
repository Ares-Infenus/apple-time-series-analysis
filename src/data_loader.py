"""
data_loader.py
==============
Responsible for loading, validating, and persisting raw time-series data.
Generates realistic synthetic AAPL price data when network access is unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_aapl(
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic AAPL OHLCV time series using
    Geometric Brownian Motion (GBM) with stochastic volatility and
    fat-tail innovations — typical of equity markets.

    Parameters
    ----------
    start : str   ISO date for first trading day
    end   : str   ISO date for last trading day
    seed  : int   Random seed for reproducibility

    Returns
    -------
    pd.DataFrame  OHLCV DataFrame indexed by date
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # --- Stochastic Volatility (Heston-inspired) ---
    mu = 0.0003  # daily drift ≈ 7.5% annualised
    v0 = 0.0002  # initial variance
    kappa = 0.05  # mean-reversion speed
    theta = 0.0002  # long-run variance
    xi = 0.01  # vol-of-vol
    rho = -0.7  # leverage effect

    vol = np.zeros(n)
    vol[0] = v0
    for t in range(1, n):
        z_v = rng.standard_normal()
        vol[t] = np.abs(
            vol[t - 1]
            + kappa * (theta - vol[t - 1])
            + xi * np.sqrt(max(vol[t - 1], 1e-8)) * z_v
        )

    # --- Correlated returns (leverage effect) ---
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    z_s = rho * z1 + np.sqrt(1 - rho**2) * z2  # stock shock

    # Fat tails via Student-t scaling
    nu = 5
    t_scale = rng.chisquare(nu, n) / nu
    returns = mu + np.sqrt(vol / t_scale) * z_s

    # --- Regime shifts (market crashes / rallies) ---
    # COVID crash: Feb–Mar 2020
    crash_mask = (dates >= "2020-02-15") & (dates <= "2020-03-23")
    returns[crash_mask] -= 0.025
    # Post-COVID rally
    rally_mask = (dates >= "2020-03-24") & (dates <= "2020-08-31")
    returns[rally_mask] += 0.003
    # 2022 rate-hike bear market
    bear_mask = (dates >= "2022-01-01") & (dates <= "2022-10-15")
    returns[bear_mask] -= 0.001

    # --- Price path ---
    s0 = 157.0  # approx AAPL close Jan 2019
    close = s0 * np.exp(np.cumsum(returns))

    # --- OHLCV construction ---
    intraday_vol = np.sqrt(vol) * close * 0.6
    high = close + np.abs(rng.normal(0, intraday_vol))
    low = close - np.abs(rng.normal(0, intraday_vol))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    volume = (rng.lognormal(mean=np.log(8e7), sigma=0.4, size=n)).astype(int)

    df = pd.DataFrame(
        {
            "Open": np.round(open_, 4),
            "High": np.round(high, 4),
            "Low": np.round(low, 4),
            "Close": np.round(close, 4),
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_data(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load AAPL data from cache or generate fresh synthetic data.

    Priority:
        1. Processed parquet cache   → fastest
        2. Raw CSV cache             → fast
        3. Synthetic generation      → always works

    Returns
    -------
    pd.DataFrame  validated OHLCV data
    """
    parquet_path = PROCESSED_DIR / "aapl_ohlcv.parquet"
    csv_path = RAW_DIR / "aapl_ohlcv.csv"

    if not force_regenerate and parquet_path.with_suffix(".csv").exists():
        logger.info("Loading from parquet cache: %s", parquet_path)
        df = pd.read_csv(parquet_path, index_col="Date", parse_dates=True)
        return _validate(df)

    if not force_regenerate and csv_path.exists():
        logger.info("Loading from CSV cache: %s", csv_path)
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df.to_csv(parquet_path.with_suffix(".csv"))
        return _validate(df)

    logger.info("Generating synthetic AAPL data (2019-2024)…")
    df = generate_synthetic_aapl()
    df.to_csv(csv_path)
    df.to_csv(parquet_path.with_suffix(".csv"))
    logger.info("Saved %d rows → %s", len(df), csv_path)
    return _validate(df)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity checks on OHLCV data."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        logger.warning("Found %d NaN values — forward-filling.", n_nulls)
        df = df.ffill().bfill()

    # High >= Low sanity
    bad = (df["High"] < df["Low"]).sum()
    if bad > 0:
        logger.warning("%d rows where High < Low — swapping.", bad)
        mask = df["High"] < df["Low"]
        df.loc[mask, ["High", "Low"]] = df.loc[mask, ["Low", "High"]].values

    logger.info(
        "Data validated: %d rows | %s → %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )
    return df
