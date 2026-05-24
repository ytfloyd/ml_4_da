"""Factor library for the MLAT-2e crypto walkthrough.

Each factor is a function with the signature:

    def factor_name(bars: pd.DataFrame) -> pd.Series

where `bars` is a *single-symbol* OHLCV DataFrame indexed by `ts`, with
columns `open, high, low, close, volume`. The function returns a `pd.Series`
indexed by the same `ts`.

Cross-asset factors (e.g. beta to BTC) take an additional `ref_rets`
argument — a Series of reference returns indexed by `ts`.

The orchestrator `compute_factor_panel()` applies all registered factors
across a multi-symbol bars DataFrame and returns a long DataFrame with
MultiIndex (symbol, ts) and one column per factor.

This is a teaching library — clarity over performance. For production-grade
factor compilation across thousands of factors, use `trend_crypto.alphas`.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd


# ── Momentum & trend ─────────────────────────────────────────────────

def mom_20(bars: pd.DataFrame) -> pd.Series:
    """20-day return."""
    return bars["close"].pct_change(20)


def mom_60(bars: pd.DataFrame) -> pd.Series:
    """60-day return."""
    return bars["close"].pct_change(60)


def mom_120(bars: pd.DataFrame) -> pd.Series:
    """120-day return."""
    return bars["close"].pct_change(120)


def mom_252(bars: pd.DataFrame) -> pd.Series:
    """252-day (annual) return."""
    return bars["close"].pct_change(252)


def mom_12_1(bars: pd.DataFrame) -> pd.Series:
    """Classic '12-1' momentum: 220-day return excluding the most recent 20 days."""
    return bars["close"].shift(20).pct_change(220)


def trend_strength_60(bars: pd.DataFrame) -> pd.Series:
    """(close − 60d MA) / (60d MA × 60d daily-return σ). A vol-normalized trend score."""
    ma = bars["close"].rolling(60).mean()
    vol = bars["close"].pct_change().rolling(60).std()
    return (bars["close"] - ma) / (ma * vol.replace(0, np.nan))


def dist_above_ma_50(bars: pd.DataFrame) -> pd.Series:
    """Percentage distance above the 50-day moving average."""
    ma = bars["close"].rolling(50).mean()
    return (bars["close"] - ma) / ma


# ── Reversal ─────────────────────────────────────────────────────────

def rev_1(bars: pd.DataFrame) -> pd.Series:
    """1-day reversal: negative of yesterday's return."""
    return -bars["close"].pct_change(1)


def rev_5(bars: pd.DataFrame) -> pd.Series:
    """Short-term (5-day) reversal."""
    return -bars["close"].pct_change(5)


# ── Volatility ───────────────────────────────────────────────────────

def vol_20(bars: pd.DataFrame) -> pd.Series:
    """20-day annualized realized volatility (daily returns)."""
    return bars["close"].pct_change().rolling(20).std() * np.sqrt(365)


def vol_60(bars: pd.DataFrame) -> pd.Series:
    """60-day annualized realized volatility."""
    return bars["close"].pct_change().rolling(60).std() * np.sqrt(365)


def vol_ratio_20_60(bars: pd.DataFrame) -> pd.Series:
    """Short-term over long-term vol — > 1 means vol is rising (regime change)."""
    return vol_20(bars) / vol_60(bars).replace(0, np.nan)


def downside_vol_60(bars: pd.DataFrame) -> pd.Series:
    """60-day annualized vol of negative returns only."""
    rets = bars["close"].pct_change()
    downside = rets.where(rets < 0, 0.0)
    return downside.rolling(60).std() * np.sqrt(365)


# ── Microstructure / volume ──────────────────────────────────────────

def dvol_zscore_20(bars: pd.DataFrame) -> pd.Series:
    """Z-score of dollar volume vs trailing 20-day mean/std."""
    dv = bars["close"] * bars["volume"]
    mu = dv.rolling(20).mean()
    sd = dv.rolling(20).std().replace(0, np.nan)
    return (dv - mu) / sd


def amihud_20(bars: pd.DataFrame) -> pd.Series:
    """Amihud illiquidity proxy: rolling mean of |return| / dollar volume.
    Higher = more illiquid (price moves more per dollar traded)."""
    rets = bars["close"].pct_change().abs()
    dv = bars["close"] * bars["volume"]
    return (rets / dv.replace(0, np.nan)).rolling(20).mean()


def vol_price_corr_20(bars: pd.DataFrame) -> pd.Series:
    """Rolling 20-day correlation between returns and log-volume.
    Positive = volume confirms price moves; negative = volume disconfirms."""
    rets = bars["close"].pct_change()
    log_vol = np.log(bars["volume"].replace(0, np.nan))
    return rets.rolling(20).corr(log_vol)


# ── Drawdown ─────────────────────────────────────────────────────────

def drawdown_252(bars: pd.DataFrame) -> pd.Series:
    """Current drawdown from the trailing 252-day high (negative number)."""
    high = bars["close"].rolling(252, min_periods=20).max()
    return (bars["close"] - high) / high


# ── Cross-asset (require a reference return series) ──────────────────

def beta_to(bars: pd.DataFrame, ref_rets: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of symbol returns to a reference return series.

    `bars` must be a single-symbol OHLCV DataFrame indexed by `ts` (not
    MultiIndex). `ref_rets` must be a `ts`-indexed Series."""
    sym_rets = bars["close"].pct_change()
    df = pd.concat([sym_rets.rename("s"), ref_rets.rename("r")], axis=1, join="inner").dropna()
    if df.empty or len(df) < window:
        return pd.Series(index=bars.index, dtype=float)
    cov = df["s"].rolling(window).cov(df["r"])
    var = df["r"].rolling(window).var().replace(0, np.nan)
    return (cov / var).reindex(bars.index)


def rel_strength_to(bars: pd.DataFrame, ref_rets: pd.Series, window: int = 60) -> pd.Series:
    """Compound symbol return over the window minus compound reference return.
    Positive = symbol outperformed the reference.

    `bars` must be a single-symbol OHLCV DataFrame indexed by `ts` (not
    MultiIndex). `ref_rets` must be a `ts`-indexed Series."""
    def cumret(r):
        return (1 + r).prod() - 1

    sym_rets = bars["close"].pct_change()
    sym_cum = sym_rets.rolling(window).apply(cumret, raw=False)
    ref_cum = ref_rets.rolling(window).apply(cumret, raw=False)
    aligned = pd.concat([sym_cum.rename("s"), ref_cum.rename("r")], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(index=bars.index, dtype=float)
    return (aligned["s"] - aligned["r"]).reindex(bars.index)


# ── Registries & orchestrator ────────────────────────────────────────

# Single-symbol factors (no reference series needed).
PER_SYMBOL_FACTORS: Mapping[str, Callable[[pd.DataFrame], pd.Series]] = {
    # momentum / trend
    "mom_20":            mom_20,
    "mom_60":            mom_60,
    "mom_120":           mom_120,
    "mom_252":           mom_252,
    "mom_12_1":          mom_12_1,
    "trend_strength_60": trend_strength_60,
    "dist_above_ma_50":  dist_above_ma_50,
    # reversal
    "rev_1":             rev_1,
    "rev_5":             rev_5,
    # volatility
    "vol_20":            vol_20,
    "vol_60":            vol_60,
    "vol_ratio_20_60":   vol_ratio_20_60,
    "downside_vol_60":   downside_vol_60,
    # microstructure
    "dvol_zscore_20":    dvol_zscore_20,
    "amihud_20":         amihud_20,
    "vol_price_corr_20": vol_price_corr_20,
    # drawdown
    "drawdown_252":      drawdown_252,
}


def compute_factor_panel(
    bars: pd.DataFrame,
    factors: Mapping[str, Callable[[pd.DataFrame], pd.Series]] = PER_SYMBOL_FACTORS,
    ref_symbol: str = "BTC-USDC",
    cross_asset_window: int = 60,
) -> pd.DataFrame:
    """Compute a panel of factor values across all (symbol, ts) in `bars`.

    Parameters
    ----------
    bars                : MultiIndex DataFrame (symbol, ts), columns = OHLCV
    factors             : mapping of name → per-symbol factor function
    ref_symbol          : symbol whose returns are used as the reference for
                          cross-asset factors (beta_to_ref, rel_strength_to_ref).
                          Set to None to skip cross-asset factors.
    cross_asset_window  : window length for cross-asset factors

    Returns
    -------
    DataFrame with MultiIndex (symbol, ts), one column per factor.
    """
    if bars.empty:
        return pd.DataFrame(index=bars.index)

    panel_cols: dict[str, pd.Series] = {}

    # Per-symbol factors.
    for name, fn in factors.items():
        panel_cols[name] = (
            bars.groupby(level="symbol", group_keys=False).apply(fn)
        )

    # Cross-asset factors against `ref_symbol`. We do these manually (not via
    # groupby.apply) so that beta_to / rel_strength_to receive ts-indexed
    # Series, which is what they need for alignment with ref_rets.
    if ref_symbol is not None and ref_symbol in bars.index.get_level_values("symbol"):
        ref_rets = bars.xs(ref_symbol)["close"].pct_change()
        beta_pieces, rel_pieces = [], []
        for sym in bars.index.get_level_values("symbol").unique():
            g = bars.xs(sym)
            b = beta_to(g, ref_rets, cross_asset_window)
            r = rel_strength_to(g, ref_rets, cross_asset_window)
            b.index = pd.MultiIndex.from_product([[sym], b.index], names=["symbol", "ts"])
            r.index = pd.MultiIndex.from_product([[sym], r.index], names=["symbol", "ts"])
            beta_pieces.append(b)
            rel_pieces.append(r)
        panel_cols[f"beta_to_ref_{cross_asset_window}"]         = pd.concat(beta_pieces)
        panel_cols[f"rel_strength_to_ref_{cross_asset_window}"] = pd.concat(rel_pieces)

    return pd.DataFrame(panel_cols)
