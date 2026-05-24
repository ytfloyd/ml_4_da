# Chapter 4 — Financial Feature Engineering: Alpha Factors

The book's chapter on engineering predictive factors. Our crypto walk:

1. **A 19-factor library** in `ml4t_crypto.features` — momentum (7), reversal (2), volatility (4), microstructure (3), drawdown (1), cross-asset (2).
2. **Compute the panel** across the eligible universe.
3. **Sanity-check** — coverage, distributions, inter-factor rank correlations.
4. **IC analysis** — daily cross-sectional Spearman rank correlation at 1, 5, and 20-day forward horizons.
5. **Turnover analysis** — decile bin-change fraction per day.
6. **Decile L/S backtests** — naïve, cost-free, weekly rebal. Lower-bound on usefulness.

Runtime: ~60–90s for the full pipeline on a ~70-symbol universe over 4 years.

Notebook: `01_factor_library_and_evaluation.ipynb`
