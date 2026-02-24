"""
visualization.py
================
Professional, publication-quality charts for financial time series analysis.
All figures exported to reports/figures/ automatically.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── Style constants ────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "primary": "#1f4e79",  # deep navy
    "secondary": "#2e86ab",  # medium blue
    "accent": "#f18f01",  # amber
    "danger": "#c73e1d",  # red
    "success": "#3b1f2b",  # dark
    "neutral": "#6c757d",
    "bg": "#f8f9fa",
    "grid": "#dee2e6",
}


def _apply_style(
    ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, linestyle="--", linewidth=0.5, color=COLORS["grid"], alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#adb5bd")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#2d3436")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=COLORS["neutral"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLORS["neutral"])
    ax.tick_params(labelsize=8, colors=COLORS["neutral"])


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── 1. Price & Volume Overview ──────────────────────────────────────────────


def plot_price_overview(df: pd.DataFrame, ticker: str = "AAPL") -> Path:
    """Candlestick-style OHLCV overview with Bollinger Bands."""
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    fig.suptitle(
        f"{ticker} — Price & Volume Overview",
        fontsize=15,
        fontweight="bold",
        y=0.98,
        color="#2d3436",
    )

    ax_price, ax_vol, ax_ret = axes
    close = df["Close"]

    # Rolling stats for Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    # Price
    ax_price.fill_between(
        close.index,
        ma20 - 2 * std20,
        ma20 + 2 * std20,
        alpha=0.12,
        color=COLORS["secondary"],
        label="BB ±2σ",
    )
    ax_price.plot(
        close.index, close, color=COLORS["primary"], linewidth=1.2, label="Close"
    )
    ax_price.plot(
        close.index,
        ma20,
        color=COLORS["accent"],
        linewidth=1,
        linestyle="--",
        label="MA(20)",
    )
    _apply_style(ax_price, f"{ticker} Closing Price", ylabel="USD ($)")
    ax_price.legend(fontsize=8, loc="upper left")
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # Volume
    ax_vol.bar(
        df.index, df["Volume"] / 1e6, color=COLORS["secondary"], alpha=0.6, width=1
    )
    _apply_style(ax_vol, ylabel="Volume (M)")

    # Log returns
    lr = np.log(close / close.shift(1)).dropna()
    colors = [COLORS["danger"] if r < 0 else COLORS["secondary"] for r in lr]
    ax_ret.bar(lr.index, lr * 100, color=colors, alpha=0.7, width=1)
    ax_ret.axhline(0, color=COLORS["neutral"], linewidth=0.7)
    _apply_style(ax_ret, ylabel="Log Return (%)", xlabel="Date")

    plt.tight_layout()
    return _save(fig, "01_price_overview")


# ── 2. Return Distribution ──────────────────────────────────────────────────


def plot_return_distribution(log_rets: pd.Series, ticker: str = "AAPL") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{ticker} — Return Distribution Analysis",
        fontsize=14,
        fontweight="bold",
        color="#2d3436",
    )

    # Histogram + KDE vs Normal
    ax = axes[0]
    data = log_rets.dropna()
    ax.hist(
        data,
        bins=80,
        density=True,
        color=COLORS["secondary"],
        alpha=0.5,
        label="Empirical",
    )
    x = np.linspace(data.min(), data.max(), 300)
    mu, sigma = data.mean(), data.std()
    ax.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        color=COLORS["danger"],
        linewidth=2,
        label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})",
    )
    kde_x = np.linspace(data.min(), data.max(), 300)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), color=COLORS["primary"], linewidth=2, label="KDE")
    _apply_style(ax, "Return Distribution", xlabel="Log Return", ylabel="Density")
    ax.legend(fontsize=7)

    # QQ Plot
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color=COLORS["secondary"])
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color=COLORS["danger"],
        linewidth=1.5,
        label=f"R²={r**2:.4f}",
    )
    _apply_style(
        ax,
        "Q-Q Plot (vs Normal)",
        xlabel="Theoretical Quantiles",
        ylabel="Sample Quantiles",
    )
    ax.legend(fontsize=8)

    # Rolling skewness & kurtosis
    ax = axes[2]
    roll_kurt = data.rolling(63).kurt()
    roll_skew = data.rolling(63).skew()
    ax.plot(
        roll_kurt.index,
        roll_kurt,
        color=COLORS["primary"],
        linewidth=1,
        label="Excess Kurtosis (63d)",
    )
    ax.plot(
        roll_skew.index,
        roll_skew,
        color=COLORS["accent"],
        linewidth=1,
        label="Skewness (63d)",
    )
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
    _apply_style(ax, "Rolling Moments (63-day)", xlabel="Date", ylabel="Value")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, "02_return_distribution")


# ── 3. Rolling Volatility ───────────────────────────────────────────────────


def plot_rolling_volatility(log_rets: pd.Series, ticker: str = "AAPL") -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))

    windows = [
        (21, COLORS["secondary"]),
        (63, COLORS["accent"]),
        (252, COLORS["primary"]),
    ]
    labels = ["21-day (monthly)", "63-day (quarterly)", "252-day (annual)"]

    for (w, c), lbl in zip(windows, labels):
        vol = log_rets.rolling(w).std() * np.sqrt(252) * 100
        ax.plot(vol.index, vol, color=c, linewidth=1.2, label=lbl, alpha=0.85)

    # EWMA volatility
    ewma_vol = log_rets.ewm(span=30).std() * np.sqrt(252) * 100
    ax.plot(
        ewma_vol.index,
        ewma_vol,
        color=COLORS["danger"],
        linewidth=1.5,
        linestyle="--",
        label="EWMA (span=30)",
        alpha=0.9,
    )

    ax.fill_between(ewma_vol.index, 0, ewma_vol, alpha=0.07, color=COLORS["danger"])

    _apply_style(
        ax,
        f"{ticker} — Annualised Volatility Regimes",
        xlabel="Date",
        ylabel="Annualised Vol (%)",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    return _save(fig, "03_rolling_volatility")


# ── 4. Stationarity Tests Visual ───────────────────────────────────────────


def plot_stationarity(series: pd.Series, diag: dict, ticker: str = "AAPL") -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{ticker} — Stationarity Analysis",
        fontsize=14,
        fontweight="bold",
        color="#2d3436",
    )

    label = diag["label"]
    adf = diag["adf"]
    kpss = diag["kpss"]

    # Original series
    ax = axes[0, 0]
    ax.plot(series.index, series, color=COLORS["primary"], linewidth=0.9)
    ma = series.rolling(63).mean()
    ax.plot(
        ma.index,
        ma,
        color=COLORS["accent"],
        linewidth=1.5,
        linestyle="--",
        label="MA(63d)",
    )
    _apply_style(ax, f"{label} — Time Plot", ylabel="Value")
    ax.legend(fontsize=8)

    # Rolling mean & std to visually check stationarity
    ax = axes[0, 1]
    roll_mean = series.rolling(63).mean()
    roll_std = series.rolling(63).std()
    ax.plot(
        roll_mean.index,
        roll_mean,
        color=COLORS["secondary"],
        linewidth=1.2,
        label="Rolling Mean (63d)",
    )
    ax2 = ax.twinx()
    ax2.plot(
        roll_std.index,
        roll_std,
        color=COLORS["accent"],
        linewidth=1.2,
        label="Rolling Std (63d)",
        linestyle="--",
    )
    ax2.set_ylabel("Std", fontsize=8, color=COLORS["accent"])
    _apply_style(ax, "Rolling Statistics", ylabel="Mean")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ADF result card
    ax = axes[1, 0]
    ax.axis("off")
    adf_color = "#d4edda" if adf.is_stationary else "#f8d7da"
    adf_text = (
        f"ADF TEST\n\n"
        f"Statistic:  {adf.statistic:.4f}\n"
        f"p-value:    {adf.pvalue:.4f}\n"
        f"Lags:       {adf.n_lags}\n\n"
        f"CV 1%:   {adf.critical_values['1%']:.3f}\n"
        f"CV 5%:   {adf.critical_values['5%']:.3f}\n"
        f"CV 10%:  {adf.critical_values['10%']:.3f}\n\n"
        + ("✅ Stationary" if adf.is_stationary else "❌ Unit Root")
    )
    ax.text(
        0.1,
        0.5,
        adf_text,
        transform=ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(facecolor=adf_color, edgecolor="#adb5bd", boxstyle="round,pad=0.8"),
    )
    ax.set_title("ADF Result", fontsize=11, fontweight="bold", color="#2d3436")

    # KPSS result card
    ax = axes[1, 1]
    ax.axis("off")
    kpss_color = "#d4edda" if kpss.is_stationary else "#f8d7da"
    kpss_text = (
        f"KPSS TEST\n\n"
        f"Statistic:  {kpss.statistic:.4f}\n"
        f"p-value:    {kpss.pvalue:.4f}\n"
        f"Lags (NW):  {kpss.n_lags}\n\n"
        f"CV 10%:  {kpss.critical_values['10%']:.3f}\n"
        f"CV 5%:   {kpss.critical_values['5%']:.3f}\n"
        f"CV 1%:   {kpss.critical_values['1%']:.3f}\n\n"
        + ("✅ Stationary" if kpss.is_stationary else "❌ Non-Stationary")
    )
    ax.text(
        0.1,
        0.5,
        kpss_text,
        transform=ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(facecolor=kpss_color, edgecolor="#adb5bd", boxstyle="round,pad=0.8"),
    )
    ax.set_title("KPSS Result", fontsize=11, fontweight="bold", color="#2d3436")

    # Verdict banner
    verdict_text = diag["verdict"]
    fig.text(
        0.5,
        0.01,
        verdict_text,
        ha="center",
        fontsize=10,
        color="#2d3436",
        style="italic",
        bbox=dict(facecolor="#fff3cd", edgecolor="#ffc107", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, "04_stationarity_tests")


# ── 5. ACF / PACF Correlogram ──────────────────────────────────────────────


def plot_correlogram(acf_res, pacf_res, title: str = "AAPL Log Returns") -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Correlogram — {title}", fontsize=14, fontweight="bold", color="#2d3436"
    )

    for ax, res in zip(axes, [acf_res, pacf_res]):
        lags = res.lags
        vals = res.values
        cb = res.conf_band

        # Stems
        ax.vlines(lags, 0, vals, color=COLORS["primary"], linewidth=1.2, alpha=0.8)
        ax.scatter(lags, vals, s=20, color=COLORS["primary"], zorder=5)

        # Confidence band
        ax.axhline(cb, color=COLORS["danger"], linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(-cb, color=COLORS["danger"], linestyle="--", linewidth=1, alpha=0.8)
        ax.fill_between(lags, -cb, cb, alpha=0.08, color=COLORS["secondary"])
        ax.axhline(0, color=COLORS["neutral"], linewidth=0.7)

        # Highlight significant lags
        sig = res.significant_lags
        if sig:
            ax.scatter(
                sig,
                vals[sig],
                s=40,
                color=COLORS["danger"],
                zorder=6,
                label=f"Significant ({len(sig)} lags)",
            )
            ax.legend(fontsize=8, loc="upper right")

        _apply_style(ax, res.kind, xlabel="Lag", ylabel="Correlation")
        ax.set_xlim(-0.5, lags[-1] + 0.5)
        ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    return _save(fig, "05_acf_pacf_correlogram")


# ── 6. Decomposition ────────────────────────────────────────────────────────


def plot_decomposition(dec_result, ticker: str = "AAPL") -> Path:
    from src.decomposition import DecompositionResult

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"{ticker} — Time Series Decomposition ({dec_result.model.title()})\n"
        f"Trend strength: {dec_result.trend_strength:.3f} | "
        f"Seasonal strength: {dec_result.seasonal_strength:.3f}",
        fontsize=13,
        fontweight="bold",
        color="#2d3436",
    )

    components = [
        (dec_result.original, "Original", COLORS["primary"]),
        (dec_result.trend, "Trend", COLORS["secondary"]),
        (dec_result.seasonal, "Seasonality", COLORS["accent"]),
        (dec_result.residual, "Residual", COLORS["neutral"]),
    ]

    for ax, (comp, name, color) in zip(axes, components):
        ax.plot(comp.index, comp, color=color, linewidth=0.9)
        if name == "Residual":
            ax.fill_between(comp.index, comp, 0, alpha=0.3, color=color)
            ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
        _apply_style(ax, ylabel=name)

    axes[-1].set_xlabel("Date", fontsize=9, color=COLORS["neutral"])
    plt.tight_layout()
    return _save(fig, "06_decomposition")


# ── 7. Volatility Clustering (ARCH effects) ────────────────────────────────


def plot_volatility_clustering(log_rets: pd.Series, ticker: str = "AAPL") -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{ticker} — Volatility Clustering & ARCH Effects",
        fontsize=14,
        fontweight="bold",
        color="#2d3436",
    )

    # Returns²
    sq_rets = log_rets**2
    abs_rets = log_rets.abs()

    ax = axes[0, 0]
    ax.plot(log_rets.index, log_rets, color=COLORS["primary"], linewidth=0.6, alpha=0.8)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.5)
    _apply_style(ax, "Log Returns", ylabel="Return")

    ax = axes[0, 1]
    ax.plot(sq_rets.index, sq_rets, color=COLORS["danger"], linewidth=0.6, alpha=0.8)
    _apply_style(ax, "Squared Returns (Volatility Proxy)", ylabel="r²")

    # ACF of squared returns (ARCH test visual)
    ax = axes[1, 0]
    from src.statistical_tests import acf as compute_acf

    acf_sq = compute_acf(sq_rets.dropna(), n_lags=30)
    lags = acf_sq.lags
    cb = acf_sq.conf_band
    ax.vlines(lags, 0, acf_sq.values, color=COLORS["danger"], linewidth=1.2)
    ax.scatter(lags, acf_sq.values, s=20, color=COLORS["danger"], zorder=5)
    ax.axhline(cb, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.axhline(-cb, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7)
    _apply_style(ax, "ACF of Squared Returns (ARCH Test)", xlabel="Lag", ylabel="ACF")

    # Scatter: r_t vs r_{t-1} (leverage / nonlinearity)
    ax = axes[1, 1]
    r = log_rets.dropna()
    ax.scatter(r.values[:-1], r.values[1:], s=3, alpha=0.3, color=COLORS["secondary"])
    m, b, r_val, _, _ = stats.linregress(r.values[:-1], r.values[1:])
    x_line = np.linspace(r.min(), r.max(), 100)
    ax.plot(
        x_line,
        m * x_line + b,
        color=COLORS["danger"],
        linewidth=1.5,
        label=f"r={r_val:.3f}",
    )
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.5, linestyle="--")
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.5, linestyle="--")
    _apply_style(
        ax, "r_t vs r_{t-1} (Serial Correlation)", xlabel="r_{t-1}", ylabel="r_t"
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, "07_volatility_clustering")


# ── 8. Heatmap: Monthly Returns Calendar ───────────────────────────────────


def plot_monthly_returns_heatmap(prices: pd.Series, ticker: str = "AAPL") -> Path:
    monthly = prices.resample("ME").last().pct_change() * 100
    monthly_df = monthly.to_frame("ret")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="Year", columns="Month", values="ret")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="#dee2e6",
        cbar_kws={"label": "Return (%)"},
        annot_kws={"size": 8},
    )
    ax.set_title(
        f"{ticker} — Monthly Return Calendar (%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#2d3436",
    )
    ax.set_xlabel("Month", fontsize=9)
    ax.set_ylabel("Year", fontsize=9)
    plt.tight_layout()
    return _save(fig, "08_monthly_return_heatmap")


# ── 9. Drawdown Analysis ────────────────────────────────────────────────────


def plot_drawdown(prices: pd.Series, ticker: str = "AAPL") -> Path:
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"{ticker} — Drawdown Analysis", fontsize=14, fontweight="bold", color="#2d3436"
    )

    axes[0].plot(prices.index, prices, color=COLORS["primary"], linewidth=1)
    axes[0].plot(
        cum_max.index,
        cum_max,
        color=COLORS["accent"],
        linewidth=1,
        linestyle="--",
        label="All-time High",
    )
    _apply_style(axes[0], ylabel="Price (USD)")
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    axes[1].fill_between(
        drawdown.index,
        drawdown,
        0,
        where=drawdown < 0,
        alpha=0.6,
        color=COLORS["danger"],
        label="Drawdown",
    )
    axes[1].plot(drawdown.index, drawdown, color=COLORS["danger"], linewidth=0.8)
    axes[1].axhline(
        -10, color=COLORS["neutral"], linewidth=0.7, linestyle=":", label="-10% level"
    )
    axes[1].axhline(
        -20, color="#6c757d", linewidth=0.7, linestyle=":", label="-20% (bear market)"
    )
    _apply_style(axes[1], ylabel="Drawdown (%)", xlabel="Date")
    axes[1].legend(fontsize=8)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    return _save(fig, "09_drawdown_analysis")


# ── 10. Statistical Summary Dashboard ──────────────────────────────────────


def plot_summary_dashboard(stats_dict: dict, ticker: str = "AAPL") -> Path:
    """Single-page statistical summary — ideal for reports."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    title = f"{ticker} Statistical Summary"
    ax.set_title(title, fontsize=16, fontweight="bold", color=COLORS["primary"], pad=20)

    rows = []
    formatters = {
        "n_obs": lambda v: f"{int(v):,}",
        "mean": lambda v: f"{v:.6f}",
        "std": lambda v: f"{v:.6f}",
        "skewness": lambda v: f"{v:.4f}",
        "excess_kurtosis": lambda v: f"{v:.4f}",
        "min": lambda v: f"{v:.6f}",
        "max": lambda v: f"{v:.6f}",
        "var_5pct": lambda v: f"{v:.6f}",
        "cvar_5pct": lambda v: f"{v:.6f}",
        "jb_stat": lambda v: f"{v:.2f}",
        "jb_pvalue": lambda v: f"{v:.4f}",
        "is_normal_jb": lambda v: "Yes" if v else "No",
    }
    labels = {
        "n_obs": "Observations",
        "mean": "Mean Daily Return",
        "std": "Std Dev (Daily)",
        "skewness": "Skewness",
        "excess_kurtosis": "Excess Kurtosis",
        "min": "Min Return",
        "max": "Max Return",
        "var_5pct": "VaR (5%)",
        "cvar_5pct": "CVaR / ES (5%)",
        "jb_stat": "Jarque-Bera Stat",
        "jb_pvalue": "JB p-value",
        "is_normal_jb": "Normal (JB test)?",
    }

    for key, label in labels.items():
        val = stats_dict.get(key, "N/A")
        fmt_val = formatters[key](val) if val != "N/A" else "N/A"
        rows.append([label, fmt_val])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Zebra rows
    for i in range(1, len(rows) + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(color)

    plt.tight_layout()
    return _save(fig, "10_statistical_summary")
