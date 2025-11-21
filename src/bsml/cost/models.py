# cost/models.py - Transaction cost model
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
    prices: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Apply realistic transaction costs to trades.
    
    Cost components:
    1. Commission (per share)
    2. Exchange fees (bps of notional)
    3. Spread cost (bid-ask spread)
    4. Temporary market impact
    5. Permanent market impact
    6. Short borrow costs (for short positions)
    7. Slippage
    
    Args:
        trades: DataFrame with [date, symbol, side, qty, price]
        cost_config: Dictionary with cost parameters
        prices: Optional price data for spread estimation
        
    Returns:
        DataFrame with added columns: [total_cost, cost_bps, net_price]
    """
    trades = trades.copy()
    
    # Extract cost parameters
    commission_per_share = cost_config.get('commission_per_share', 0.0035)
    exchange_fee_bps = cost_config.get('exchange_fee_bps', 0.5)
    spread_factor = cost_config.get('spread_factor', 0.5)
    temp_impact_bps = cost_config.get('temp_impact_bps', 7)
    perm_impact_bps = cost_config.get('perm_impact_bps', 2)
    slippage_bps = cost_config.get('slippage_bps', 1)
    short_borrow_annual = cost_config.get('short_borrow_annual', 0.015)
    
    # Calculate notional value
    trades['notional'] = trades['qty'] * trades['price']
    
    # 1. Commission
    trades['cost_commission'] = trades['qty'] * commission_per_share
    
    # 2. Exchange fees
    trades['cost_exchange'] = trades['notional'] * (exchange_fee_bps / 10000)
    
    # 3. Spread cost (half-spread crossing)
    # Estimate spread as % of price (use bid-ask data if available)
    if prices is not None and 'spread_bps' in prices.columns:
        # Merge actual spreads
        trades = trades.merge(
            prices[['date', 'symbol', 'spread_bps']], 
            on=['date', 'symbol'],
            how='left'
        )
        trades['spread_bps'] = trades['spread_bps'].fillna(10)  # Default 10bps
    else:
        # Estimate spread based on volatility proxy
        trades['spread_bps'] = 10  # Conservative estimate
    
    trades['cost_spread'] = trades['notional'] * (
        trades['spread_bps'] * spread_factor / 10000
    )
    
    # 4. Temporary market impact (square root model)
    # Impact proportional to sqrt(trade_size / ADV)
    # Simplified: use fixed impact in bps
    trades['cost_temp_impact'] = trades['notional'] * (temp_impact_bps / 10000)
    
    # 5. Permanent market impact
    trades['cost_perm_impact'] = trades['notional'] * (perm_impact_bps / 10000)
    
    # 6. Slippage (additional adverse selection)
    trades['cost_slippage'] = trades['notional'] * (slippage_bps / 10000)
    
    # 7. Short borrow costs (for shorts, annualized rate prorated)
    trades['is_short'] = trades['side'].str.upper().isin(['SELL', 'SHORT'])
    trades['holding_days'] = 1  # Assume 1-day holding for simplicity
    trades['cost_borrow'] = np.where(
        trades['is_short'],
        trades['notional'] * short_borrow_annual * (trades['holding_days'] / 365),
        0
    )
    
    # Total cost
    cost_columns = [
        'cost_commission', 'cost_exchange', 'cost_spread',
        'cost_temp_impact', 'cost_perm_impact', 
        'cost_slippage', 'cost_borrow'
    ]
    trades['total_cost'] = trades[cost_columns].sum(axis=1)
    
    # Cost in basis points
    trades['cost_bps'] = (trades['total_cost'] / trades['notional']) * 10000
    
    # Net execution price (after costs)
    trades['net_price'] = np.where(
        trades['side'].str.upper() == 'BUY',
        trades['price'] + (trades['total_cost'] / trades['qty']),
        trades['price'] - (trades['total_cost'] / trades['qty'])
    )
    
    # Clean up temporary columns
    trades = trades.drop(['is_short', 'holding_days'], axis=1, errors='ignore')
    
    return trades


def compute_implementation_shortfall(
    trades: pd.DataFrame,
    benchmark_price: str = 'arrival_price'
) -> pd.DataFrame:
    """
    Compute implementation shortfall for trades.
    
    IS = (Execution Price - Benchmark Price) / Benchmark Price
    
    For buys: positive IS = paid more than benchmark (bad)
    For sells: positive IS = received less than benchmark (bad)
    
    Args:
        trades: DataFrame with execution data
        benchmark_price: Column name for benchmark ('arrival_price', 'ref_price', etc.)
        
    Returns:
        DataFrame with added 'impl_shortfall_bps' column
    """
    trades = trades.copy()
    
    if benchmark_price not in trades.columns:
        # Use ref_price as fallback
        benchmark_price = 'ref_price'
    
    # Calculate shortfall
    trades['impl_shortfall'] = np.where(
        trades['side'].str.upper() == 'BUY',
        trades['net_price'] - trades[benchmark_price],
        trades[benchmark_price] - trades['net_price']
    )
    
    # Convert to basis points
    trades['impl_shortfall_bps'] = (
        trades['impl_shortfall'] / trades[benchmark_price]
    ) * 10000
    
    return trades
