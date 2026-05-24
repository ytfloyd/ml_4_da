"""ml4t_crypto — shared utilities for the MLAT-2e crypto walkthrough."""
from .backtest import get_engines, get_metrics, get_strategy_base, perf_stats
from .config import (
    CURATED_MEMBERSHIP_TABLE,
    CURATED_UNIVERSE_TABLE,
    DEFAULT_TABLES,
    LAKE_PATH,
    QUOTE_CURRENCIES,
)
from .data import (
    base_of,
    connect,
    list_tables,
    list_tokens,
    load_bars,
    stitched_bars,
    table_range,
)
from .quality import data_quality, eligible_universe, universe_quality, QualityReport
from .universe import (
    # Full-universe (default) helpers
    available_symbols_on,
    quote_currency_of,
    split_by_quote,
    symbol_listing_dates,
    symbols_with_history,
    # Curated reference universe (not the default)
    curated_symbols_on,
    load_curated_membership,
    load_curated_universe,
)

__all__ = [
    # Lake basics
    "LAKE_PATH",
    "DEFAULT_TABLES",
    "QUOTE_CURRENCIES",
    "CURATED_UNIVERSE_TABLE",
    "CURATED_MEMBERSHIP_TABLE",
    "base_of",
    "connect",
    "list_tables",
    "list_tokens",
    "load_bars",
    "stitched_bars",
    "table_range",
    # Data-quality + canonical universe filter
    "data_quality",
    "eligible_universe",
    "universe_quality",
    "QualityReport",
    # Full universe (the default)
    "available_symbols_on",
    "quote_currency_of",
    "split_by_quote",
    "symbol_listing_dates",
    "symbols_with_history",
    # Curated reference universe
    "load_curated_universe",
    "load_curated_membership",
    "curated_symbols_on",
    # Backtest adapters
    "perf_stats",
    "get_engines",
    "get_metrics",
    "get_strategy_base",
]
