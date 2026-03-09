"""
Data Generator Module
Generates realistic synthetic ETF price data with proper correlations.

Paper Section 4.1 — Cholesky-decomposed covariance structure:
  1. Construct correlation matrix from empirical SPY correlations
  2. Compute Cholesky decomposition: Σ = LL^T
  3. Generate uncorrelated returns: Z ~ N(0, I)
  4. Apply correlation: R = Z × L^T
  5. Scale by individual μ, σ parameters per ETF
  6. Compound to price series: P_t = 100 · exp(cumsum(r_t))
"""

import numpy as np
import pandas as pd

# ── ETF characteristics (annual return, annual volatility, correlation to SPY) ─
ETF_PARAMS = {
    'SPY':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 1.00},
    'QQQ':  {'mu': 0.15, 'sigma': 0.22, 'corr_spy': 0.85},
    'IVV':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 0.99},
    'VOO':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 0.99},
    'VTI':  {'mu': 0.13, 'sigma': 0.17, 'corr_spy': 0.95},
    'EEM':  {'mu': 0.06, 'sigma': 0.24, 'corr_spy': 0.65},
    'GLD':  {'mu': 0.08, 'sigma': 0.15, 'corr_spy': -0.10},
    'TLT':  {'mu': 0.02, 'sigma': 0.14, 'corr_spy': -0.30},
    'XLF':  {'mu': 0.10, 'sigma': 0.20, 'corr_spy': 0.90},
    'EFA':  {'mu': 0.09, 'sigma': 0.18, 'corr_spy': 0.85},
}

N_DAYS_PAPER = 1538   # paper Section 5: exactly 1,538 trading days (~6 years)


def generate_etf_prices(config: dict) -> pd.DataFrame:
    """
    Generate synthetic ETF prices with realistic correlations.

    Parameters
    ----------
    config : dict
        Must contain 'universe' (list of ticker strings).
        Date range determined by priority:
          1. config['n_days']                    → periods starting at start_date
          2. config['start_date'] + config['end_date']
          3. default: N_DAYS_PAPER=1538 days from 2017-01-03

    Returns
    -------
    pd.DataFrame
        Wide format: columns = ['date'] + universe tickers, rows = trading days.
    """
    universe = config['universe']
    n_etfs = len(universe)

    # ── Build date range ─────────────────────────────────────────────────────
    if 'n_days' in config:
        date_range = pd.date_range(
            start=config.get('start_date', '2017-01-03'),
            periods=int(config['n_days']),
            freq='B',
        )
    elif 'start_date' in config and 'end_date' in config:
        date_range = pd.date_range(
            start=config['start_date'],
            end=config['end_date'],
            freq='B',
        )
    else:
        date_range = pd.date_range(
            start='2017-01-03',
            periods=N_DAYS_PAPER,
            freq='B',
        )
    n_days = len(date_range)

    # ── Correlation matrix via SPY factor model ──────────────────────────────
    corr_matrix = np.ones((n_etfs, n_etfs))
    for i, etf_i in enumerate(universe):
        for j, etf_j in enumerate(universe):
            if i != j:
                corr_matrix[i, j] = (
                    ETF_PARAMS[etf_i]['corr_spy'] *
                    ETF_PARAMS[etf_j]['corr_spy']
                )

    # ── Cholesky decomposition ───────────────────────────────────────────────
    L = np.linalg.cholesky(corr_matrix)

    # ── Correlated returns ───────────────────────────────────────────────────
    seed = config.get('seed', 42)
    rng = np.random.default_rng(seed)
    raw_returns = rng.standard_normal((n_days, n_etfs))
    correlated_returns = raw_returns @ L.T

    # ── Scale and compound to prices ─────────────────────────────────────────
    prices_df = pd.DataFrame(index=date_range, columns=universe, dtype=float)
    for i, etf in enumerate(universe):
        params = ETF_PARAMS[etf]
        daily_mu = params['mu'] / 252
        daily_sigma = params['sigma'] / np.sqrt(252)
        returns = daily_mu + daily_sigma * correlated_returns[:, i]
        prices_df[etf] = 100.0 * np.exp(np.cumsum(returns))

    prices_df['date'] = prices_df.index
    prices_df = prices_df.reset_index(drop=True)
    prices_df = prices_df[['date'] + list(universe)]

    return prices_df


def generate_long_format(config: dict) -> pd.DataFrame:
    """
    Generate synthetic prices in long format (date, symbol, price) for the loader.
    """
    wide = generate_etf_prices(config)
    universe = config['universe']
    long = wide.melt(id_vars='date', value_vars=universe,
                     var_name='symbol', value_name='price')
    long['date'] = pd.to_datetime(long['date'])
    long['price'] = pd.to_numeric(long['price'], errors='coerce')
    long = long.dropna(subset=['price'])
    long = long.sort_values(['symbol', 'date']).reset_index(drop=True)
    return long


if __name__ == '__main__':
    import json

    with open('../config.json', 'r') as f:
        config = json.load(f)[0]

    prices = generate_etf_prices(config)
    print(prices.head())
    print(f"\nGenerated {len(prices)} days of data")
    print(f"\nCorrelation matrix:")
    print(prices[config['universe']].corr().round(2))
