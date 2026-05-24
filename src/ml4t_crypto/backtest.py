"""Adapter around the `trend_crypto` backtest engine.

`trend_crypto` (sister repo at `~/Dropbox/NRT/nrt_dev/trend_crypto`) is installed
into this venv as an editable package via:

    pip install -e ../trend_crypto

This module re-exports the pieces our MLAT-2e chapters reach for, and adds a
couple of pandas-friendly helpers so notebooks don't all need to know the
underlying polars-based API.

What lives where in trend_crypto:

    backtest.engine            BacktestEngine            (single-asset)
    backtest.portfolio_engine  PortfolioEngine           (multi-asset)
    backtest.metrics           sharpe, sortino, calmar, max_drawdown
    strategy.base              TargetWeightStrategy, PortfolioStrategy
    strategy.buy_and_hold      BuyAndHoldStrategy
    strategy.ma_crossover_long_only  MACrossoverLongOnlyStrategy
    strategy.ma_cross_vol_hysteresis MACrossVolHysteresis

We import lazily so that `import ml4t_crypto` still works even if trend_crypto
hasn't been installed yet (it'll just raise a helpful error when you try to
use the adapter).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backtest.engine import BacktestEngine  # noqa: F401
    from backtest.portfolio_engine import PortfolioEngine  # noqa: F401


_HELP = (
    "trend_crypto isn't importable. From the ml_4_da repo, run:\n"
    "    pip install -e ../trend_crypto\n"
    "(both repos live under ~/Dropbox/NRT/nrt_dev/.)"
)


def _import_trend_crypto():
    try:
        import backtest  # noqa: F401  — trend_crypto's top-level package
        import strategy  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(_HELP) from e


def get_engines():
    """Return (BacktestEngine, PortfolioEngine) classes."""
    _import_trend_crypto()
    from backtest.engine import BacktestEngine
    from backtest.portfolio_engine import PortfolioEngine
    return BacktestEngine, PortfolioEngine


def get_metrics():
    """Return the trend_crypto.backtest.metrics module."""
    _import_trend_crypto()
    from backtest import metrics
    return metrics


def get_strategy_base():
    """Return (TargetWeightStrategy, PortfolioStrategy) ABCs."""
    _import_trend_crypto()
    from strategy.base import TargetWeightStrategy, PortfolioStrategy
    return TargetWeightStrategy, PortfolioStrategy


# ── pandas-friendly perf summary ──────────────────────────────────────
#
# Most chapter notebooks just want: "given a daily/hourly returns Series,
# what's the Sharpe, Sortino, Calmar, max drawdown?". The trend_crypto
# helpers want polars Series and a NAV (not returns), so we wrap.

def perf_stats(
    returns: pd.Series,
    *,
    periods_per_year: int = 365,
    risk_free: float = 0.0,
) -> pd.Series:
    """Compute a small perf summary from a returns Series.

    Parameters
    ----------
    returns          : pandas Series of period returns (e.g. daily simple returns)
    periods_per_year : 365 for daily crypto bars, 8760 for hourly, 252 for daily TradFi
    risk_free        : *per-bar* risk-free rate (defaults to 0 — fine for crypto).
                       NOTE: trend_crypto.backtest.metrics expects per-bar, not
                       annualized. If you want an annualized 5% rf with daily
                       bars, pass 0.05 / 365.

    Returns
    -------
    A pandas Series with: total_return, ann_return, ann_vol, sharpe, sortino,
    calmar, max_drawdown.
    """
    metrics = get_metrics()
    import polars as pl

    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    nav = (1 + r).cumprod()
    nav_pl = pl.Series(nav.values)
    r_pl = pl.Series(r.values)

    total_return = float(nav.iloc[-1] - 1)
    ann_return = float((1 + total_return) ** (periods_per_year / len(r)) - 1)
    ann_vol = float(r.std() * (periods_per_year ** 0.5))

    return pd.Series({
        "total_return": total_return,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       metrics.sharpe_ratio(r_pl, periods_per_year=periods_per_year, risk_free=risk_free),
        "sortino":      metrics.sortino_ratio(r_pl, periods_per_year=periods_per_year, risk_free=risk_free),
        "calmar":       metrics.calmar_ratio(r_pl, nav_pl, periods_per_year=periods_per_year),
        "max_drawdown": metrics.max_drawdown(nav_pl),
    })


__all__ = [
    "get_engines",
    "get_metrics",
    "get_strategy_base",
    "perf_stats",
]
