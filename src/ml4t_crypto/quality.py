"""Data-quality flagging and the canonical eligible-universe helper.

Two responsibilities:

1. **Per-symbol QC** — `data_quality(symbol)` returns a small report of
   crypto-typical pathologies (zero-volume runs, suspicious price jumps,
   long flat windows). Report-only — we never auto-exclude. Notebooks and
   strategies decide whether to act on the flags.

2. **The universe filter every later chapter will call** —
   `eligible_universe(as_of_date, ...)` returns the symbols that pass a
   parameterized set of filters as of a given date. Point-in-time by
   construction (uses only data ≤ `as_of_date`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_TABLES
from .data import connect, load_bars
from .universe import quote_currency_of


# ── Per-symbol data quality ───────────────────────────────────────────

@dataclass(frozen=True)
class QualityReport:
    symbol: str
    granularity: str
    n_bars: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    coverage_pct: float          # actual bars / expected bars over the window
    zero_volume_pct: float       # share of bars with volume == 0
    flat_close_pct: float        # share of bars with close == prior close (no change)
    max_single_bar_return: float # max |close pct-change| seen
    n_jumps_over_50pct: int      # number of bars with |return| > 0.5
    n_jumps_over_20pct: int      # number of bars with |return| > 0.2
    median_dollar_volume: float  # median close * volume
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["notes"] = "; ".join(self.notes) if self.notes else ""
        return d


def _expected_bars(first: pd.Timestamp, last: pd.Timestamp, granularity: str) -> int:
    """Crude expected-bar count assuming no gaps."""
    sec_per_bar = {"1m": 60, "1h": 3600, "4h": 14400, "1d": 86400}[granularity]
    delta = (last - first).total_seconds()
    return int(delta / sec_per_bar) + 1


def data_quality(symbol: str, granularity: str = "1d") -> QualityReport:
    """Compute a QC report for a single symbol-quote pair.

    This is the diagnostic function. It produces flags but never excludes
    anything — callers decide what thresholds matter for their strategy.
    """
    bars = load_bars(granularity, symbols=[symbol])
    if bars.empty:
        raise ValueError(f"No bars for {symbol!r} at {granularity!r}")
    df = bars.xs(symbol)

    n = len(df)
    first, last = df.index.min(), df.index.max()
    expected = _expected_bars(first, last, granularity)
    coverage = n / expected if expected > 0 else 0.0

    zero_vol = (df["volume"] == 0).mean()
    flat = (df["close"].diff() == 0).mean()

    rets = df["close"].pct_change()
    abs_rets = rets.abs()
    max_ret = float(abs_rets.max()) if abs_rets.notna().any() else 0.0
    n50 = int((abs_rets > 0.5).sum())
    n20 = int((abs_rets > 0.2).sum())

    dv = (df["close"] * df["volume"]).replace(0, np.nan)
    med_dv = float(dv.median()) if dv.notna().any() else 0.0

    notes: list[str] = []
    if coverage < 0.9:
        notes.append(f"coverage {coverage:.1%}")
    if zero_vol > 0.05:
        notes.append(f"zero-vol bars {zero_vol:.1%}")
    if flat > 0.2:
        notes.append(f"flat-close bars {flat:.1%}")
    if n50 > 0:
        notes.append(f"{n50} bars with |ret|>50%")
    if med_dv < 10_000:
        notes.append(f"median \\$-vol {med_dv:,.0f}")

    return QualityReport(
        symbol=symbol,
        granularity=granularity,
        n_bars=n,
        first_date=first,
        last_date=last,
        coverage_pct=coverage,
        zero_volume_pct=zero_vol,
        flat_close_pct=flat,
        max_single_bar_return=max_ret,
        n_jumps_over_50pct=n50,
        n_jumps_over_20pct=n20,
        median_dollar_volume=med_dv,
        notes=tuple(notes),
    )


def universe_quality(
    symbols: Optional[Iterable[str]] = None,
    granularity: str = "1d",
) -> pd.DataFrame:
    """Run data_quality across many symbols. Returns a DataFrame, one row per
    symbol. If `symbols` is None, runs across every symbol in the lake's
    daily table — a few hundred to a thousand depending on granularity."""
    if symbols is None:
        table = DEFAULT_TABLES[granularity]
        with connect() as con:
            symbols = [r[0] for r in con.execute(
                f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
            ).fetchall()]

    rows = []
    for s in symbols:
        try:
            rows.append(data_quality(s, granularity).to_dict())
        except Exception as e:
            rows.append({"symbol": s, "granularity": granularity, "notes": f"ERROR: {e}"})
    return pd.DataFrame(rows).set_index("symbol")


# ── The canonical eligible-universe helper ────────────────────────────

def eligible_universe(
    as_of_date: str,
    min_history_days: int = 365,
    min_dollar_volume: float = 0.0,
    dollar_volume_window_days: int = 20,
    quotes: Sequence[str] = ("USDC", "USD"),
    granularity: str = "1d",
    dedupe_tokens: bool = True,
) -> pd.DataFrame:
    """Symbols passing the standard set of universe filters *as of* a given date.

    Point-in-time by construction: every computation uses only data with
    `ts <= as_of_date`. Designed to be called once per rebalance.

    Parameters
    ----------
    as_of_date                  : the "today" of the filter (ISO date)
    min_history_days            : require at least this many bars before as_of_date
    min_dollar_volume           : require trailing-window median dollar volume ≥ this
    dollar_volume_window_days   : how many bars back to look for the volume check
    quotes                      : whitelist of quote currencies in priority order
                                  (default: USDC > USD)
    granularity                 : OHLCV table to query
    dedupe_tokens               : if True (default), keep one row per token using
                                  the highest-priority quote currency available.
                                  If False, return one row per (token, quote) pair.

    Returns
    -------
    DataFrame with columns: symbol, token, n_bars_to_date, first_date, last_date,
    median_dollar_volume_recent. Sorted by median_dollar_volume_recent desc.
    """
    table = DEFAULT_TABLES[granularity]

    # The volume window we sample over to enforce the liquidity floor.
    # We use the last `dollar_volume_window_days` of bars up to as_of_date.
    sql = f"""
    WITH ts_filter AS (
      SELECT * FROM {table} WHERE ts <= ?
    ),
    counts AS (
      SELECT symbol,
             COUNT(*) AS n_bars,
             MIN(ts)::DATE AS first_date,
             MAX(ts)::DATE AS last_date
      FROM ts_filter
      GROUP BY symbol
    ),
    recent AS (
      SELECT symbol,
             MEDIAN(close * volume) AS median_dollar_volume_recent
      FROM (
        SELECT symbol, close, volume,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
        FROM ts_filter
      )
      WHERE rn <= ?
      GROUP BY symbol
    )
    SELECT c.symbol, c.n_bars AS n_bars_to_date, c.first_date, c.last_date,
           COALESCE(r.median_dollar_volume_recent, 0) AS median_dollar_volume_recent
    FROM counts c
    LEFT JOIN recent r USING (symbol)
    WHERE c.n_bars >= ?
      AND COALESCE(r.median_dollar_volume_recent, 0) >= ?
    ORDER BY median_dollar_volume_recent DESC
    """

    with connect() as con:
        df = con.execute(sql, [as_of_date, dollar_volume_window_days,
                               min_history_days, min_dollar_volume]).fetchdf()

    if quotes is not None:
        wanted = set(quotes)
        df = df[df["symbol"].apply(lambda s: quote_currency_of(s) in wanted)].reset_index(drop=True)

    df["token"] = df["symbol"].str.rsplit("-", n=1).str[0]

    if dedupe_tokens and not df.empty:
        # Keep the row whose quote currency is highest-priority per `quotes`.
        priority = {q: i for i, q in enumerate(quotes)}
        df["_pri"] = df["symbol"].apply(lambda s: priority.get(quote_currency_of(s), 999))
        df = (
            df.sort_values(["token", "_pri"])
              .drop_duplicates(subset=["token"], keep="first")
              .drop(columns=["_pri"])
              .sort_values("median_dollar_volume_recent", ascending=False)
              .reset_index(drop=True)
        )

    cols = ["symbol", "token", "n_bars_to_date", "first_date", "last_date",
            "median_dollar_volume_recent"]
    return df[cols]
