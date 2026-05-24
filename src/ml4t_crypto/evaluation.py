"""Factor-evaluation primitives.

Building blocks the chapters use to ask 'is this factor any good?':

- forward_returns(bars, horizon)  — per-symbol forward return Series
- factor_ic(factor_panel, fwd_rets) — daily cross-sectional Spearman IC per factor
- factor_ic_at_horizons(factor_panel, bars, horizons) — IC across multiple horizons
- factor_turnover(factor_panel) — share of names whose decile rank changes per day
- decile_backtest(factor, bars, ...) — top-vs-bottom decile long-short equity curve

All functions are pandas-friendly and cost-unaware: this module is for
characterizing signals, not pricing them. Costs and proper portfolio
construction come in Ch 5 onward.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats


# ── Forward returns ──────────────────────────────────────────────────

def forward_returns(bars: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Per-symbol log-style forward return over `horizon` bars.

    `close[t+horizon] / close[t] - 1`, computed per-symbol so that the
    last `horizon` bars of each symbol come out NaN (no peek).
    """
    closes = bars["close"]
    fwd = closes.groupby(level="symbol").transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    fwd.name = f"fwd_ret_{horizon}"
    return fwd


# ── Information coefficient ──────────────────────────────────────────

def _spearman_ic_at_date(factor_vals: pd.Series, fwd_vals: pd.Series) -> float:
    """Spearman rank correlation between a factor cross-section and forward returns
    on a single date. Returns NaN if there aren't enough non-null pairs."""
    df = pd.concat([factor_vals, fwd_vals], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def factor_ic(
    factor_panel: pd.DataFrame,
    fwd_rets: pd.Series,
    min_names: int = 5,
) -> pd.DataFrame:
    """Daily cross-sectional IC for each factor.

    Parameters
    ----------
    factor_panel : DataFrame MultiIndex (symbol, ts), columns = factors
    fwd_rets     : Series MultiIndex (symbol, ts)
    min_names    : minimum number of non-null (factor, fwd) pairs required
                   on a date to compute IC; otherwise the date is NaN.

    Returns
    -------
    DataFrame indexed by ts, columns = factors, values = daily Spearman IC.
    """
    out: dict[str, pd.Series] = {}
    fwd = fwd_rets.rename("__fwd__")
    for factor_name in factor_panel.columns:
        df = pd.concat([factor_panel[factor_name].rename("f"), fwd], axis=1).dropna()
        if df.empty:
            out[factor_name] = pd.Series(dtype=float)
            continue
        ics: dict[pd.Timestamp, float] = {}
        for d, group in df.groupby(level="ts"):
            if len(group) >= min_names:
                ics[d] = _spearman_ic_at_date(group["f"], group["__fwd__"])
        out[factor_name] = pd.Series(ics, dtype=float).sort_index()
    return pd.DataFrame(out)


def factor_ic_at_horizons(
    factor_panel: pd.DataFrame,
    bars: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 20),
) -> pd.DataFrame:
    """Mean IC for each factor at multiple forward-return horizons.

    Returns
    -------
    DataFrame indexed by factor, columns = horizons, values = mean daily IC.
    """
    out = {}
    for h in horizons:
        fwd = forward_returns(bars, horizon=h)
        ic_panel = factor_ic(factor_panel, fwd)
        out[h] = ic_panel.mean()
    return pd.DataFrame(out)


# ── Turnover ─────────────────────────────────────────────────────────

def factor_turnover(
    factor_panel: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.Series:
    """For each factor, the fraction of names whose quantile bin changes day-over-day.

    Higher turnover = faster-moving signal = more trading = more cost-sensitive
    in a real strategy. A factor with turnover near 0 is suspiciously persistent
    (possibly look-ahead or stale). A factor at ~1/n_quantiles is well-mixed.
    """
    out = {}
    for factor in factor_panel.columns:
        s = factor_panel[factor]
        # Cross-sectional ranking per date → quantile bin.
        try:
            ranks = s.groupby(level="ts").transform(
                lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
            )
        except ValueError:
            # qcut fails when there's not enough variation; skip.
            out[factor] = float("nan")
            continue
        # Did each symbol's bin change vs yesterday?
        bin_panel = ranks.unstack(level="symbol")
        changed = bin_panel.diff().ne(0)
        out[factor] = float(changed.mean().mean())
    return pd.Series(out, name="bin_turnover").sort_values(ascending=False)


# ── Decile (or n-tile) backtest ──────────────────────────────────────

def decile_backtest(
    factor: pd.Series,
    bars: pd.DataFrame,
    n_quantiles: int = 10,
    rebal_days: int = 5,
    long_high: bool = True,
) -> dict:
    """Long-short equally-weighted backtest of top vs bottom quantile of a factor.

    Parameters
    ----------
    factor       : Series MultiIndex (symbol, ts) — factor values
    bars         : DataFrame MultiIndex (symbol, ts) — for next-day return calc
    n_quantiles  : 10 = deciles
    rebal_days   : rebalance every N bars (5 = weekly)
    long_high    : if True, long the top quantile / short the bottom. Set False
                   for factors where 'lower is better' (e.g. low-vol anomaly).

    Returns
    -------
    dict with:
        equity         : pd.Series — daily NAV (starts at 1.0)
        long_short_ret : pd.Series — daily L/S returns
        long_ret       : pd.Series — daily long-leg returns
        short_ret      : pd.Series — daily short-leg returns (positive = winning)
        n_long         : pd.Series — # symbols long each day
        n_short        : pd.Series — # symbols short each day
        ann_return     : float
        sharpe         : float
        max_dd         : float
    """
    # 1-day forward return is what a position held overnight earns.
    fwd1 = forward_returns(bars, horizon=1)

    # Quantile bins per date.
    bins = factor.groupby(level="ts").transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
    )

    # Sample dates for rebalance (every `rebal_days`).
    all_dates = factor.index.get_level_values("ts").unique().sort_values()
    rebal_dates = all_dates[::rebal_days]

    # Build target-weight series held between rebalances.
    top_label = n_quantiles - 1 if long_high else 0
    bot_label = 0 if long_high else n_quantiles - 1

    long_ret_l, short_ret_l, ls_ret_l, n_long_l, n_short_l, dates_l = [], [], [], [], [], []

    for i, d in enumerate(rebal_dates):
        # Names in top / bottom quantile as of d.
        bins_today = bins.xs(d, level="ts", drop_level=True).dropna()
        longs  = bins_today[bins_today == top_label].index
        shorts = bins_today[bins_today == bot_label].index
        if len(longs) == 0 or len(shorts) == 0:
            continue

        # Hold these positions until the next rebalance.
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        hold_dates = all_dates[(all_dates >= d) & (all_dates < next_d)]

        for hd in hold_dates:
            try:
                rets_today = fwd1.xs(hd, level="ts", drop_level=True)
            except KeyError:
                continue
            l = rets_today.reindex(longs).dropna()
            s = rets_today.reindex(shorts).dropna()
            if l.empty or s.empty:
                continue
            long_ret_l.append(l.mean())
            short_ret_l.append(-s.mean())  # short returns = -realized
            ls_ret_l.append(l.mean() - s.mean())
            n_long_l.append(len(l))
            n_short_l.append(len(s))
            dates_l.append(hd)

    if not dates_l:
        return {"equity": pd.Series(dtype=float), "ann_return": float("nan"),
                "sharpe": float("nan"), "max_dd": float("nan")}

    long_ret  = pd.Series(long_ret_l, index=dates_l, name="long_ret")
    short_ret = pd.Series(short_ret_l, index=dates_l, name="short_ret")
    ls_ret    = pd.Series(ls_ret_l, index=dates_l, name="ls_ret")

    equity = (1 + ls_ret).cumprod()
    ann_return = float((1 + ls_ret.mean()) ** 365 - 1) if not ls_ret.empty else float("nan")
    sharpe = (
        float(ls_ret.mean() / ls_ret.std() * np.sqrt(365))
        if ls_ret.std() > 0
        else float("nan")
    )
    rolling_max = equity.cummax()
    max_dd = float(((equity / rolling_max) - 1).min())

    return {
        "equity":         equity,
        "long_short_ret": ls_ret,
        "long_ret":       long_ret,
        "short_ret":      short_ret,
        "n_long":         pd.Series(n_long_l, index=dates_l),
        "n_short":        pd.Series(n_short_l, index=dates_l),
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "max_dd":         max_dd,
    }


def all_decile_backtests(
    factor_panel: pd.DataFrame,
    bars: pd.DataFrame,
    n_quantiles: int = 10,
    rebal_days: int = 5,
    long_high_overrides: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Run decile_backtest across every factor in the panel. Summary table.

    `long_high_overrides` lets you flip the direction for factors where
    "low is better" (e.g. low-vol anomaly, low-amihud anomaly). Pass
    {'vol_60': False, 'amihud_20': False} to short the high values.
    """
    overrides = dict(long_high_overrides or {})
    rows = []
    for f in factor_panel.columns:
        res = decile_backtest(
            factor_panel[f],
            bars,
            n_quantiles=n_quantiles,
            rebal_days=rebal_days,
            long_high=overrides.get(f, True),
        )
        rows.append({
            "factor":     f,
            "long_high":  overrides.get(f, True),
            "ann_return": res["ann_return"],
            "sharpe":     res["sharpe"],
            "max_dd":     res["max_dd"],
        })
    return pd.DataFrame(rows).set_index("factor").sort_values("sharpe", ascending=False)
