#!/usr/bin/env python3
"""
NOTE: This project uses SYNTHETIC data only.
      Do NOT download real market data from yfinance or any other source.

To regenerate data/ALL_backtest.csv, run the synthetic data generator:
    PYTHONPATH=src python3 data/scripts/generate_synthetic_data.py

The generator uses a Cholesky-decomposed covariance structure (paper Section 4.1)
to produce 1,538 trading days of correlated synthetic ETF prices.

Paper universe (Section 4):
    SPY, QQQ, IVV, VOO, VTI, EEM, GLD, TLT, XLF, EFA
"""
