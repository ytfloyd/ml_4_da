"""Thin wrappers around the DuckDB lake.

Notebooks should do:

    from ml4t_crypto import load_bars
    df = load_bars("1d", symbols=["BTC-USD", "ETH-USD"], start="2020-01-01")

For tokens that exist across multiple quote currencies (e.g. BTC-USD and
BTC-USDC), use `stitched_bars("BTC", "1d")` to get a single continuous series.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import duckdb
import pandas as pd

from .config import DEFAULT_TABLES, LAKE_PATH


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a connection to the lake. Defaults to read-only — the lake is the
    source of truth and notebooks should not mutate it."""
    return duckdb.connect(str(LAKE_PATH), read_only=read_only)


def list_tables() -> list[str]:
    with connect() as con:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def table_range(table: str) -> pd.DataFrame:
    """Quick summary: first/last ts, distinct symbols, row count."""
    with connect() as con:
        return con.execute(f"""
            SELECT
              MIN(ts)::DATE AS first_date,
              MAX(ts)::DATE AS last_date,
              COUNT(DISTINCT symbol) AS symbols,
              COUNT(*) AS rows
            FROM {table}
        """).fetchdf()


def load_bars(
    granularity: str = "1d",
    symbols: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    table: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load OHLCV bars from the lake.

    Parameters
    ----------
    granularity : one of "1m", "1h", "4h", "1d"
    symbols     : optional list of tickers (e.g. ["BTC-USD", "ETH-USD"])
    start, end  : ISO date strings, inclusive
    table       : override the default table for that granularity
    columns     : subset of columns to return (always includes symbol/ts)

    Returns
    -------
    DataFrame indexed by (symbol, ts), columns are the requested OHLCV fields.
    """
    if table is None:
        if granularity not in DEFAULT_TABLES:
            raise ValueError(f"Unknown granularity {granularity!r}; expected one of {list(DEFAULT_TABLES)}")
        table = DEFAULT_TABLES[granularity]

    cols = list(columns) if columns is not None else ["open", "high", "low", "close", "volume"]
    select = ", ".join(["symbol", "ts", *cols])

    where = []
    params: list = []
    if symbols is not None:
        placeholders = ",".join(["?"] * len(symbols))
        where.append(f"symbol IN ({placeholders})")
        params.extend(symbols)
    if start is not None:
        where.append("ts >= ?")
        params.append(start)
    if end is not None:
        where.append("ts <= ?")
        params.append(end)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    query = f"SELECT {select} FROM {table} {where_clause} ORDER BY symbol, ts"

    with connect() as con:
        df = con.execute(query, params).fetchdf()

    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index(["symbol", "ts"]).sort_index()


# ── Token-level helpers (base currency, e.g. "BTC", "ETH") ────────────

def base_of(symbol: str) -> str:
    """'BTC-USDC' → 'BTC'. Returns the symbol unchanged if no '-' present."""
    return symbol.rsplit("-", 1)[0] if "-" in symbol else symbol


def list_tokens(granularity: str = "1d") -> list[str]:
    """Distinct base tokens (e.g. 'BTC', 'ETH') in the lake at this granularity.
    Deduplicates across quote currencies."""
    table = DEFAULT_TABLES[granularity]
    with connect() as con:
        rows = con.execute(
            f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
        ).fetchall()
    return sorted({base_of(r[0]) for r in rows})


# ── Stitched bars (cross-quote-currency, single continuous series) ────

def stitched_bars(
    token: str,
    granularity: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    quote_pref: Sequence[str] = ("USDC", "USD"),
) -> pd.DataFrame:
    """Continuous OHLCV for a token, stitched across quote currencies.

    The rule: on each timestamp, use the *highest-preference quote* that has
    data on that bar. With the default `quote_pref=("USDC", "USD")`, USDC bars
    are used wherever they exist; USD fills in everywhere else (typically the
    pre-USDC era).

    This means for most tokens you get:
        - USD bars from the first listing date up to (first USDC date - 1 bar)
        - USDC bars from the first USDC date onward

    The result has an extra `quote_source` column so you can see exactly where
    the splice happened.

    Parameters
    ----------
    token       : base currency, e.g. "BTC", "ETH", "SOL"
    granularity : "1m", "1h", "4h", "1d"
    start, end  : ISO date strings, inclusive
    quote_pref  : tuple of quote currencies, in priority order. Default
                  ("USDC", "USD") matches the Coinbase USD → USDC migration.

    Returns
    -------
    DataFrame indexed by ts with columns: open, high, low, close, volume,
    quote_source. Empty DataFrame if the token doesn't exist for any of the
    requested quotes.
    """
    candidates = [f"{token}-{q}" for q in quote_pref]
    table = DEFAULT_TABLES[granularity]

    where = ["symbol IN (" + ",".join(["?"] * len(candidates)) + ")"]
    params: list = list(candidates)
    if start is not None:
        where.append("ts >= ?"); params.append(start)
    if end is not None:
        where.append("ts <= ?"); params.append(end)
    clause = "WHERE " + " AND ".join(where)

    with connect() as con:
        raw = con.execute(
            f"SELECT symbol, ts, open, high, low, close, volume "
            f"FROM {table} {clause} ORDER BY ts",
            params,
        ).fetchdf()

    if raw.empty:
        return raw

    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)

    # Priority ordering: lower number = higher preference.
    priority = {sym: i for i, sym in enumerate(candidates)}
    raw["_pri"] = raw["symbol"].map(priority)

    # Keep one row per ts: the lowest-priority-number (= most preferred) symbol.
    raw = raw.sort_values(["ts", "_pri"]).drop_duplicates(subset=["ts"], keep="first")

    raw = raw.rename(columns={"symbol": "quote_source"})
    raw["quote_source"] = raw["quote_source"].str.rsplit("-", n=1).str[-1]
    raw = raw.drop(columns=["_pri"]).set_index("ts").sort_index()
    return raw[["open", "high", "low", "close", "volume", "quote_source"]]
