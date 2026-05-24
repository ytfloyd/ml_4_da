# Chapter 2 — Market Data: Anatomy, Stitching, Quality, Universe

The book's Chapter 2 covers market microstructure, data sources, and storage
formats. Since our lake is already built, we shift the chapter toward the
things you actually need to know to use it safely.

## Goals for the crypto version

1. Verify the anatomy of an OHLCV bar — invariants hold, granularities are consistent.
2. Walk through hourly-vs-daily-vs-1-minute trade-offs.
3. Explain raw vs `_clean` tables and why `_clean` is the default.
4. Understand the USD ↔ USDC migration and the `stitched_bars()` abstraction.
5. Introduce `data_quality()` and `universe_quality()` — report-only QC, never auto-excludes.
6. Introduce `eligible_universe(date, ...)` — the canonical universe filter that every later chapter reuses.

Notebook: `01_anatomy.ipynb`
