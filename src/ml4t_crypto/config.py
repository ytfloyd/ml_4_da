"""Paths and constants for the crypto data lake.

Universe philosophy
-------------------
This project's default research universe is the **full** Coinbase OHLCV lake —
every symbol the exchange has ever listed, no liquidity filter, no rank cap.
That's why `DEFAULT_TABLES` points at the raw `*_clean` tables. We do this
deliberately: limiting the universe is a *research choice*, not a default.
Each chapter that needs to construct a tradeable universe will do so visibly,
with its own filters and rationale.

The lake also contains a *curated* universe (`bars_1d_usd_universe_clean_top50_adv10m`):
top-50-by-ADV ∩ ADV20 ≥ \\$10M, with point-in-time membership. We keep it as a
named reference — useful for comparing our own filtering against an
institutional-liquidity baseline — but it is *not* the default.
"""
from __future__ import annotations

import os
from pathlib import Path

# Override with COINBASE_LAKE env var if you need to point elsewhere.
LAKE_PATH = Path(
    os.environ.get(
        "COINBASE_LAKE",
        Path.home() / "Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb",
    )
).expanduser()

# Default granularity → table. We use the *_clean variants throughout
# (de-duplicated, gap-aware). These cover the FULL universe — 850 symbols
# across 7 quote currencies — with no liquidity filter applied.
DEFAULT_TABLES = {
    "1m":  "candles_1m",
    "1h":  "bars_1h_clean",
    "4h":  "bars_4h_clean",
    "1d":  "bars_1d_clean",
}

# Curated reference universe — top-50 by 20-day ADV, with a $10M ADV floor.
# NOT the default research universe — see module docstring.
CURATED_UNIVERSE_TABLE   = "bars_1d_usd_universe_clean_top50_adv10m"
CURATED_MEMBERSHIP_TABLE = "bars_1d_usd_universe_clean_top50_adv10m_membership"

# Recognized quote currencies in the lake, in rough order of trading relevance.
QUOTE_CURRENCIES = ("USDC", "USD", "USDT", "EUR", "GBP", "BTC", "ETH")
