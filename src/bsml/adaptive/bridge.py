"""
P7 Bridge Module - Converts trades to adversary classifier input format

Updated Week 4: Support for multiple prediction windows
- Original: "Will trade occur tomorrow?" (24h window)
- New: "Will trade occur in next 4 hours?" (4h window)

Owner: P7
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def prepare_adversary_data(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    prediction_window_hours: float = 4.0,
    auto_detect_window: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convert trades to adversary prediction format.
    
    Prediction task: "Will a trade occur in the next X hours?"
    
    Args:
        trades: DataFrame with columns ['date', 'symbol', 'side', 'qty', 'price', 'ref_price']
        prices: DataFrame with columns ['date', 'symbol', 'price'] + technical indicators
        prediction_window_hours: Hours to look ahead for trade prediction (default: 4.0)
        auto_detect_window: If True, automatically determine optimal window based on data
        verbose: Print diagnostic info
    
    Returns:
        DataFrame with features + 'label' column (1 if trade occurs within window, 0 otherwise)
    """
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Ensure datetime
    trades = trades.copy()
    prices = prices.copy()
    trades['date'] = pd.to_datetime(trades['date'])
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Auto-detect optimal window if requested
    if auto_detect_window:
        prediction_window_hours = _auto_detect_window(trades, verbose=verbose)
    
    if verbose:
        print(f"\n[Bridge] Prediction window: {prediction_window_hours:.1f} hours")
    
    # Sort by date
    trades = trades.sort_values('date').reset_index(drop=True)
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Get unique symbols
    symbols = trades['symbol'].unique()
    
    all_obs = []
    
    for symbol in symbols:
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_prices) == 0:
            continue
        
        # Merge prices with technical indicators
        symbol_prices = symbol_prices.sort_values('date').reset_index(drop=True)
        
        # For each price observation, label if trade occurs within window
        symbol_prices['label'] = 0
        
        for idx, row in symbol_prices.iterrows():
            obs_time = row['date']
            window_end = obs_time + pd.Timedelta(hours=prediction_window_hours)
            
            # Check if any trade occurs in [obs_time, window_end)
            trades_in_window = symbol_trades[
                (symbol_trades['date'] >= obs_time) & 
                (symbol_trades['date'] < window_end)
            ]
            
            if len(trades_in_window) > 0:
                symbol_prices.at[idx, 'label'] = 1
        
        # Add signal column (synonym for label, for backward compatibility)
        symbol_prices['signal'] = symbol_prices['label']
        
        all_obs.append(symbol_prices)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True)
    
    if verbose:
        n_positive = result['label'].sum()
        n_total = len(result)
        print(f"[Bridge] Total observations: {n_total}")
        print(f"[Bridge] Positive (trade within {prediction_window_hours:.1f}h): {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"[Bridge] Negative (no trade): {n_total - n_positive} ({(n_total-n_positive)/n_total*100:.1f}%)")
    
    return result


def _auto_detect_window(trades: pd.DataFrame, verbose: bool = False) -> float:
    """
    Automatically detect optimal prediction window based on trade gaps.
    
    Logic:
    - Calculate time gaps between consecutive trades
    - Use median gap as the prediction window
    - Clamp to reasonable range [1h, 48h]
    
    Args:
        trades: Trades DataFrame
        verbose: Print diagnostic info
    
    Returns:
        Optimal window in hours
    """
    trades = trades.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    gaps = []
    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_trades = symbol_trades.sort_values('date')
        
        if len(symbol_trades) < 2:
            continue
        
        trade_times = symbol_trades['date'].values
        time_diffs = np.diff(trade_times).astype('timedelta64[h]').astype(float)
        gaps.extend(time_diffs)
    
    if len(gaps) == 0:
        if verbose:
            print("[Bridge] Auto-detect: No gaps found, using default 4.0h")
        return 4.0
    
    # Use median gap as window
    median_gap = np.median(gaps)
    
    # Clamp to reasonable range
    optimal_window = np.clip(median_gap, 1.0, 48.0)
    
    if verbose:
        print(f"[Bridge] Auto-detect: Median gap = {median_gap:.1f}h")
        print(f"[Bridge] Auto-detect: Optimal window = {optimal_window:.1f}h (clamped to [1h, 48h])")
    
    return optimal_window


def prepare_adversary_data_daily(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Legacy function: Daily prediction task "Will trade occur tomorrow?"
    
    Equivalent to prepare_adversary_data with prediction_window_hours=24
    """
    return prepare_adversary_data(
        trades, 
        prices, 
        prediction_window_hours=24.0,
        auto_detect_window=False,
        verbose=verbose
    )


# Backward compatibility
prepare_adversary_data_v1 = prepare_adversary_data_daily
