# cost/models.py — Transaction cost model (Section 4 of paper)
#
# Six components, all wired into every backtest run:
#   1. Commission         : $0.0035 / share
#   2. Exchange/clearing  : 0.5 bps of notional
#   3. Spread cost        : 0.5 × quoted spread
#   4. Temporary impact   : 7 bps × participation_rate^0.6   (power law)
#   5. Permanent impact   : 2 bps × participation_rate^0.5   (power law)
#   6. Slippage floor     : 1 bps per order
#   (Short borrow: 1.5 % annual, prorated; applied to short legs only)
#
# Aggregate ≈ 5 bps per unit turnover on the baseline strategy.
import pandas as pd
import numpy as np
from typing import Dict
import yaml


def load_cost_config(path: str) -> Dict:
    """Load cost configuration from YAML."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_costs(
    trades: pd.DataFrame,
    cost_config: Dict,
    prices: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply all six transaction-cost components to a trades DataFrame.

    Parameters
    ----------
    trades       : DataFrame with at minimum [date, symbol, side, qty, price]
    cost_config  : dict loaded from costs.yaml
    prices       : optional DataFrame with [date, symbol, adv, spread_bps]
                   for symbol-level ADV and spread data

    Returns
    -------
    trades copy with additional columns:
        notional, participation_rate,
        cost_commission, cost_exchange, cost_spread,
        cost_temp_impact, cost_perm_impact, cost_slippage, cost_borrow,
        total_cost, cost_bps, net_price
    """
    trades = trades.copy()

    # ── Cost parameters ───────────────────────────────────────────────────────
    commission_per_share = cost_config.get('commission_per_share', 0.0035)
    exchange_fee_bps     = cost_config.get('exchange_fee_bps', 0.5)
    spread_factor        = cost_config.get('spread_factor', 0.5)      # 0.5 × spread
    temp_impact_bps      = cost_config.get('temp_impact_bps', 7)      # bps coefficient
    perm_impact_bps      = cost_config.get('perm_impact_bps', 2)
    slippage_bps         = cost_config.get('slippage_bps', 1)
    short_borrow_annual  = cost_config.get('short_borrow_annual', 0.015)

    # ── Notional ─────────────────────────────────────────────────────────────
    trades['notional'] = trades['qty'].abs() * trades['price']

    # ── Participation rate ────────────────────────────────────────────────────
    # ADV proxy: if real ADV data is available in prices, merge it; otherwise
    # assume each trade is 1 % of ADV (conservative for liquid large-cap ETFs).
    if prices is not None and 'adv' in prices.columns:
        trades = trades.merge(
            prices[['date', 'symbol', 'adv']], on=['date', 'symbol'], how='left'
        )
        trades['adv'] = trades['adv'].fillna(trades['notional'] * 100)
    else:
        trades['adv'] = trades['notional'] * 100   # → participation = 1 %

    trades['participation_rate'] = (
        (trades['notional'] / trades['adv']).clip(lower=1e-6, upper=1.0)
    )

    # ── 1. Commission: $0.0035 per share ─────────────────────────────────────
    trades['cost_commission'] = trades['qty'].abs() * commission_per_share

    # ── 2. Exchange / clearing: 0.5 bps of notional ──────────────────────────
    trades['cost_exchange'] = trades['notional'] * (exchange_fee_bps / 10_000)

    # ── 3. Spread: 0.5 × quoted spread ───────────────────────────────────────
    if prices is not None and 'spread_bps' in prices.columns:
        trades = trades.merge(
            prices[['date', 'symbol', 'spread_bps']], on=['date', 'symbol'], how='left'
        )
        trades['spread_bps'] = trades['spread_bps'].fillna(10)
    else:
        trades['spread_bps'] = 10   # 10 bps is typical for liquid ETFs

    trades['cost_spread'] = (
        trades['notional'] * (trades['spread_bps'] * spread_factor / 10_000)
    )

    # ── 4. Temporary market impact: 7 bps × participation^0.6 ────────────────
    trades['cost_temp_impact'] = (
        trades['notional']
        * (temp_impact_bps / 10_000)
        * (trades['participation_rate'] ** 0.6)
    )

    # ── 5. Permanent market impact: 2 bps × participation^0.5 ────────────────
    trades['cost_perm_impact'] = (
        trades['notional']
        * (perm_impact_bps / 10_000)
        * (trades['participation_rate'] ** 0.5)
    )

    # ── 6. Slippage floor: 1 bps per order ───────────────────────────────────
    trades['cost_slippage'] = trades['notional'] * (slippage_bps / 10_000)

    # ── Short borrow costs (prorated at 1-day holding period) ────────────────
    is_short = trades['side'].str.upper().isin(['SELL', 'SHORT'])
    trades['cost_borrow'] = np.where(
        is_short,
        trades['notional'] * short_borrow_annual / 252,   # 1 trading day
        0.0,
    )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    cost_columns = [
        'cost_commission', 'cost_exchange', 'cost_spread',
        'cost_temp_impact', 'cost_perm_impact',
        'cost_slippage', 'cost_borrow',
    ]
    trades['total_cost'] = trades[cost_columns].sum(axis=1)
    trades['cost_bps']   = (trades['total_cost'] / trades['notional'].clip(lower=1e-8)) * 10_000

    # ── Net execution price ───────────────────────────────────────────────────
    cost_per_unit = trades['total_cost'] / trades['qty'].abs().clip(lower=1e-8)
    trades['net_price'] = np.where(
        trades['side'].str.upper() == 'BUY',
        trades['price'] + cost_per_unit,
        trades['price'] - cost_per_unit,
    )

    # Drop helper columns not needed downstream
    trades = trades.drop(columns=['adv'], errors='ignore')

    return trades


def decompose_implementation_shortfall(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose IS into four components (Table 7 of paper).

    Components
    ----------
    is_spread_bps      : bid-ask spread cost
    is_market_impact_bps : temporary + permanent market impact
    is_adverse_sel_bps  : adverse selection (approx from slippage)
    is_timing_bps       : remaining IS after the above three

    All values in basis points; positive = cost (bad for the trader).

    Parameters
    ----------
    trades : DataFrame produced by apply_costs()

    Returns
    -------
    trades with four additional IS-decomposition columns
    """
    trades = trades.copy()

    ref = 'ref_price'
    if ref not in trades.columns:
        ref = 'price'

    ref_px = trades[ref]

    # 1. Spread component
    trades['is_spread_bps'] = (
        trades['cost_spread'] / ref_px.clip(lower=1e-8)
    ) * 10_000

    # 2. Market impact (temp + perm)
    trades['is_market_impact_bps'] = (
        (trades['cost_temp_impact'] + trades['cost_perm_impact'])
        / ref_px.clip(lower=1e-8)
    ) * 10_000

    # 3. Adverse selection ≈ slippage component
    trades['is_adverse_sel_bps'] = (
        trades['cost_slippage'] / ref_px.clip(lower=1e-8)
    ) * 10_000

    # 4. Total IS and timing residual
    sign = np.where(trades['side'].str.upper() == 'BUY', 1, -1)
    total_is_bps = sign * (trades['net_price'] - ref_px) / ref_px.clip(lower=1e-8) * 10_000
    trades['impl_shortfall_bps'] = total_is_bps
    trades['is_timing_bps'] = (
        total_is_bps
        - trades['is_spread_bps']
        - trades['is_market_impact_bps']
        - trades['is_adverse_sel_bps']
    )

    return trades


def compute_implementation_shortfall(
    trades: pd.DataFrame,
    benchmark_price: str = 'arrival_price',
) -> pd.DataFrame:
    """
    Compute raw implementation shortfall in basis points.

    IS_bps > 0 means the trade cost more than the benchmark (bad).
    """
    trades = trades.copy()
    if benchmark_price not in trades.columns:
        benchmark_price = 'ref_price'

    ref_px = trades[benchmark_price]
    trades['impl_shortfall'] = np.where(
        trades['side'].str.upper() == 'BUY',
        trades['net_price'] - ref_px,
        ref_px - trades['net_price'],
    )
    trades['impl_shortfall_bps'] = (
        trades['impl_shortfall'] / ref_px.clip(lower=1e-8)
    ) * 10_000

    return trades
