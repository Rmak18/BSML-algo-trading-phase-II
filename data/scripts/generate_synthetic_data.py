#!/usr/bin/env python3
"""
Regenerate data/ALL_backtest.csv from the synthetic price model.

Paper Section 4.1: Cholesky-decomposed covariance structure.
Paper Section 5: exactly 1,538 trading days.

Run from repo root:
    PYTHONPATH=src python3 data/scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from bsml.policies.data_generator import generate_long_format, N_DAYS_PAPER

UNIVERSE = ['SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'EEM', 'GLD', 'TLT', 'XLF', 'EFA']
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "ALL_backtest.csv"

config = {
    'universe': UNIVERSE,
    'n_days': N_DAYS_PAPER,   # 1,538 trading days (paper Section 5)
    'start_date': '2017-01-03',
    'seed': 42,
}

print(f"Generating {N_DAYS_PAPER} days of synthetic prices for {len(UNIVERSE)} ETFs …")
long_df = generate_long_format(config)

long_df.to_csv(OUT_PATH, index=False)
print(f"Written: {OUT_PATH}")

counts = long_df.groupby('symbol').size()
print(f"Rows per symbol: {counts.min()} min, {counts.max()} max")
print(f"Date range: {long_df['date'].min().date()} → {long_df['date'].max().date()}")
print(f"Total rows: {len(long_df)}")

assert (counts == N_DAYS_PAPER).all(), f"Row count mismatch: {counts.to_dict()}"
assert counts.min() >= 253, "Still too short for 252-day TSMOM lookback"
print(f"✓ All {len(UNIVERSE)} symbols have {N_DAYS_PAPER} rows — TSMOM will produce signals")
