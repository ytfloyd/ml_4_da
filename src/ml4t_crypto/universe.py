"""Universe helpers.

There are two kinds of universe in this project:

1. **The full universe** — every symbol the lake has bars for, with no
   filtering. This is the default for research notebooks. Helpers:
       available_symbols_on(date)
       symbols_with_history(min_days)
       symbol_listing_dates()
       quote_currency_of(symbol)
       split_by_quote(symbols)

2. **The curated reference universe** — top-50 by 20-day ADV with a $10M ADV
   floor, with point-in-time membership. Useful as a benchmark, *not* the
   default. Helpers:
       load_curated_universe()
       load_curated_membership()
       curated_symbols_on(date)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import pandas as pd

from .config import (
    CURATED_MEMBERSHIP_TABLE,
    CURATED_UNIVERSE_TABLE,
    DEFAULT_TABLES,
    QUOTE_CURRENCIES,
)
from .data import connect


# ── Full-universe helpers (the default) ───────────────────────────────

def available_symbols_on(
    date: str,
    granularity: str = "1d",
    quotes: Optional[Iterable[str]] = None,
) -> list[str]:
    """Symbols that have at least one bar on the given date.

    Parameters
    ----------
    date         : ISO date string, e.g. "2024-01-15"
    granularity  : "1m", "1h", "4h", or "1d"
    quotes       : optional whitelist of quote currencies (e.g. {"USDC", "USD"})
                   to restrict to. If None, all quote currencies are returned.
    """
    table = DEFAULT_TABLES[granularity]
    with connect() as con:
        rows = con.execute(
            f"SELECT DISTINCT symbol FROM {table} WHERE ts::DATE = ? ORDER BY symbol",
            [date],
        ).fetchall()
    syms = [r[0] for r in rows]
    if quotes is not None:
        wanted = set(quotes)
        syms = [s for s in syms if quote_currency_of(s) in wanted]
    return syms


def symbols_with_history(
    min_days: int,
    granularity: str = "1d",
    end_date: Optional[str] = None,
    quotes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Symbols that have at least `min_days` of bars on the given granularity.

    Returns a DataFrame with columns: symbol, first_date, last_date, n_bars.
    Filtered to `end_date` (inclusive) if provided.
    """
    table = DEFAULT_TABLES[granularity]
    where = ""
    params: list = []
    if end_date is not None:
        where = "WHERE ts <= ?"
        params.append(end_date)
    with connect() as con:
        df = con.execute(f"""
            SELECT symbol,
                   MIN(ts)::DATE AS first_date,
                   MAX(ts)::DATE AS last_date,
                   COUNT(*) AS n_bars
            FROM {table}
            {where}
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY n_bars DESC
        """, [*params, min_days]).fetchdf()
    if quotes is not None:
        wanted = set(quotes)
        df = df[df["symbol"].apply(lambda s: quote_currency_of(s) in wanted)].reset_index(drop=True)
    return df


def symbol_listing_dates(granularity: str = "1d") -> pd.DataFrame:
    """First and last bar date for every symbol in the lake.

    Columns: symbol, first_date, last_date, n_bars.
    """
    return symbols_with_history(min_days=1, granularity=granularity)


def quote_currency_of(symbol: str) -> str:
    """Extract the quote currency from a symbol like 'BTC-USDC' → 'USDC'.
    Returns 'other' if the symbol doesn't split on '-' or the quote isn't
    one of the recognized quote currencies."""
    if "-" not in symbol:
        return "other"
    quote = symbol.rsplit("-", 1)[-1].upper()
    return quote if quote in QUOTE_CURRENCIES else "other"


def split_by_quote(symbols: Iterable[str]) -> dict[str, list[str]]:
    """Group symbols by quote currency."""
    out: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        out[quote_currency_of(s)].append(s)
    return dict(out)


# ── Curated reference universe (not the default) ──────────────────────

def load_curated_universe(
    start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    """Load the curated top-50 ∩ \\$10M ADV universe bars (with vwap & adv20_usd).

    This is the institutional-liquidity baseline. *Not* the default research
    universe — for that, query the bars_1d_clean / bars_1h_clean tables
    directly via load_bars()."""
    where, params = [], []
    if start is not None:
        where.append("ts >= ?"); params.append(start)
    if end is not None:
        where.append("ts <= ?"); params.append(end)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    q = f"SELECT * FROM {CURATED_UNIVERSE_TABLE} {clause} ORDER BY symbol, ts"
    with connect() as con:
        df = con.execute(q, params).fetchdf()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index(["symbol", "ts"]).sort_index()


def load_curated_membership(
    start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    """Load point-in-time membership for the curated universe."""
    where, params = [], []
    if start is not None:
        where.append("ts >= ?"); params.append(start)
    if end is not None:
        where.append("ts <= ?"); params.append(end)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    q = f"SELECT * FROM {CURATED_MEMBERSHIP_TABLE} {clause} ORDER BY ts, rn"
    with connect() as con:
        df = con.execute(q, params).fetchdf()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def curated_symbols_on(date: str) -> list[str]:
    """Symbols in the curated reference universe on a given date."""
    with connect() as con:
        rows = con.execute(
            f"SELECT symbol FROM {CURATED_MEMBERSHIP_TABLE} "
            f"WHERE ts::DATE = ? ORDER BY rn",
            [date],
        ).fetchall()
    return [r[0] for r in rows]
