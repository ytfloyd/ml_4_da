# ml_4_da — Machine Learning for Algorithmic Trading, on Coinbase crypto

A pair-programmed walkthrough of Stefan Jansen's *Machine Learning for Algorithmic
Trading* (2e), adapted to a Coinbase OHLCV data lake rather than the book's
US-equities + SEC-filings universe.

## Data

A single DuckDB file at `~/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb`
(~45 GB) with OHLCV bars at 1-minute, 1-hour, 4-hour, and daily granularity covering
the full Coinbase universe — **850 symbols across 7 quote currencies** (USDC, USD,
USDT, EUR, GBP, BTC, ETH) — from 2015-01 to 2026-05.

Key tables:

| Table | Notes |
|---|---|
| `candles_1m`, `bars_1h`, `bars_4h`, `bars_1d` | Raw OHLCV |
| `*_clean` | De-duplicated, gap-handled. **Default research universe.** |
| `bars_1d_usd_universe_clean_top50_adv10m` | *Reference* curated daily universe (top-50 ∩ ADV20 ≥ \$10M), with `vwap` and `adv20_usd` |
| `bars_1d_usd_universe_clean_top50_adv10m_membership` | Point-in-time membership for the curated universe |

### Universe philosophy

We default to the **full** universe — every symbol the lake has bars for, no
liquidity filter, no rank cap. The curated `top50_adv10m` universe is a *reference*
(useful as a baseline to compare our own filtering against), not the default. Any
chapter that needs a tradeable subset will construct its filters visibly, with
rationale.

The available universe grew from 4 symbols/day in 2015 to ~830 symbols/day today.
Symbol availability is itself a research variable: most chapters will filter by
quote currency (`USDC ∪ USD` is a reasonable "tradeable in dollars" set), minimum
history, and per-chapter liquidity thresholds.

## Track

Trading-focused subset of MLAT 2e — skipping the NLP-heavy chapters (14-18) and
SEC-filings alt-data (3, 20) that don't translate to crypto. Numbering follows
the book.

| Ch | Topic | Crypto angle |
|---|---|---|
| 1 | ML for Trading workflow | Frame the ML4T loop against our lake |
| 2 | Market & fundamental data | Hourly/daily OHLCV + PIT universe |
| 4 | Alpha factors | Momentum, vol, microstructure |
| 5 | Strategy evaluation | Vectorbt with crypto-specific costs |
| 6 | ML workflow | Purged/embargoed CV |
| 7 | Linear models | Ridge / Lasso on factors |
| 8 | ML4T end-to-end | First full pipeline |
| 9 | Time-series models | ARIMA, GARCH, cointegration |
| 10 | Bayesian ML | Stochastic vol, Bayesian Sharpe |
| 11 | Trees & boosting | LightGBM on factors |
| 13 | Unsupervised | Regime detection, clusters |
| 19 | RNNs | LSTM on hourly returns |
| 21 | Generative / autoencoders | Anomaly detection |
| 22 | Deep RL | DQN/PPO trading agent |

## Layout

```
ml_4_da/
├── src/ml4t_crypto/    # Reusable data + utility code (DuckDB loaders, universe helpers)
├── chapters/           # One folder per chapter; notebooks + scratch
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                     # makes `import ml4t_crypto` work in notebooks
pip install -e ../trend_crypto       # makes `from backtest.engine import ...` work
jupyter lab
```

### About `trend_crypto`

This repo leans on the sister repo `~/Dropbox/NRT/nrt_dev/trend_crypto` for its
backtesting engine. Installing it editable means notebooks can do:

```python
from ml4t_crypto.backtest import get_engines, perf_stats
BacktestEngine, PortfolioEngine = get_engines()

from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from backtest.metrics import sharpe_ratio
```

Edits to `trend_crypto/src/...` are picked up immediately — no resync. Use the
`ml4t_crypto.backtest` adapter when you want a pandas-friendly interface;
import directly from `backtest.*` / `strategy.*` when you want the native API.