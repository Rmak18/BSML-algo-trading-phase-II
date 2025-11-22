"""
P7 Bridge Module V2 - Policy Distinguishability
Generates adversary training data to distinguish Baseline vs Uniform policy.

Task: "Can you tell which policy generated this trade?"
Label: 0 = Baseline, 1 = Uniform

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from typing import Tuple


def engineer_trade_features(trades: pd.DataFrame, prices: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Engineer features that capture policy differences.
    
    Focus on EXECUTION characteristics and timing patterns,
    not just price movements.
    
    Args:
        trades: Combined baseline + uniform trades with 'policy' label
        prices: Market price data
        verbose: Print diagnostics
    
    Returns:
        DataFrame with features and labels
    """
    
    all_obs = []
    
    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_trades) < 10:
            continue
        
        # Ensure datetime
        symbol_trades['date'] = pd.to_datetime(symbol_trades['date'])
        symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
        
        # Sort
        symbol_trades = symbol_trades.sort_values('date').reset_index(drop=True)
        symbol_prices = symbol_prices.sort_values('date').reset_index(drop=True)
        
        # Merge trades with market prices (on date only, ignoring time-of-day)
        symbol_trades['date_only'] = symbol_trades['date'].dt.date
        symbol_prices['date_only'] = symbol_prices['date'].dt.date
        
        merged = pd.merge(
            symbol_trades, 
            symbol_prices[['date_only', 'price']], 
            on='date_only', 
            how='left',
            suffixes=('_trade', '_market')
        )
        
        # Drop if missing market price
        merged = merged.dropna(subset=['price_market'])
        
        if len(merged) < 10:
            continue
        
        # =====================================================================
        # FEATURE GROUP 1: Price Execution Characteristics
        # (Uniform policy perturbs ref_price, baseline doesn't)
        # =====================================================================
        
        merged['price_deviation'] = (merged['ref_price'] - merged['price_market']) / (merged['price_market'] + 1e-8)
        merged['abs_price_deviation'] = merged['price_deviation'].abs()
        
        # Rolling statistics of price deviations
        merged['price_dev_ma5'] = merged['price_deviation'].rolling(5, min_periods=1).mean()
        merged['price_dev_std5'] = merged['price_deviation'].rolling(5, min_periods=1).std()
        merged['price_dev_ma20'] = merged['price_deviation'].rolling(20, min_periods=1).mean()
        merged['price_dev_std20'] = merged['price_deviation'].rolling(20, min_periods=1).std()
        
        # Max/min deviations in recent window
        merged['price_dev_max5'] = merged['abs_price_deviation'].rolling(5, min_periods=1).max()
        merged['price_dev_min5'] = merged['abs_price_deviation'].rolling(5, min_periods=1).min()
        
        # =====================================================================
        # FEATURE GROUP 2: Temporal Features
        # (Calendar patterns, not price-based)
        # =====================================================================
        
        merged['day_of_week'] = merged['date'].dt.dayofweek
        merged['day_of_month'] = merged['date'].dt.day
        merged['week_of_year'] = merged['date'].dt.isocalendar().week
        merged['month'] = merged['date'].dt.month
        
        # Time-of-day features (if timestamp has time component)
        merged['hour'] = merged['date'].dt.hour
        merged['minute'] = merged['date'].dt.minute
        
        # =====================================================================
        # FEATURE GROUP 3: Trade Timing Patterns
        # (Captures temporal clustering)
        # =====================================================================
        
        merged['days_since_last_trade'] = merged['date'].diff().dt.total_seconds() / 86400
        merged['days_since_last_trade'] = merged['days_since_last_trade'].fillna(0)
        
        # Trade velocity
        merged['trades_last_5d'] = merged['date'].rolling(5, min_periods=1).count()
        merged['trades_last_20d'] = merged['date'].rolling(20, min_periods=1).count()
        
        # Average inter-trade time
        merged['avg_days_between_trades_5'] = merged['days_since_last_trade'].rolling(5, min_periods=1).mean()
        merged['avg_days_between_trades_20'] = merged['days_since_last_trade'].rolling(20, min_periods=1).mean()
        
        # =====================================================================
        # FEATURE GROUP 4: Trade Direction Features
        # (Check for direction consistency patterns)
        # =====================================================================
        
        merged['side_numeric'] = (merged['side'] == 'BUY').astype(int)
        
        # Direction changes
        merged['direction_change'] = (merged['side_numeric'].diff() != 0).astype(int)
        merged['direction_changes_5d'] = merged['direction_change'].rolling(5, min_periods=1).sum()
        merged['direction_changes_20d'] = merged['direction_change'].rolling(20, min_periods=1).sum()
        
        # Consecutive same-direction trades
        merged['direction_streak'] = (
            merged['side_numeric']
            .groupby((merged['side_numeric'] != merged['side_numeric'].shift()).cumsum())
            .cumcount() + 1
        )
        
        # =====================================================================
        # FEATURE GROUP 5: Price Context (Market conditions)
        # (Still useful for context, but not the main signal)
        # =====================================================================
        
        merged['returns_1d'] = merged['price_market'].pct_change()
        merged['returns_5d'] = merged['price_market'].pct_change(5)
        merged['returns_20d'] = merged['price_market'].pct_change(20)
        
        merged['vol_5d'] = merged['returns_1d'].rolling(5, min_periods=1).std()
        merged['vol_20d'] = merged['returns_1d'].rolling(20, min_periods=1).std()
        
        merged['sma_5'] = merged['price_market'].rolling(5, min_periods=1).mean()
        merged['sma_20'] = merged['price_market'].rolling(20, min_periods=1).mean()
        merged['price_to_sma5'] = merged['price_market'] / (merged['sma_5'] + 1e-8)
        merged['price_to_sma20'] = merged['price_market'] / (merged['sma_20'] + 1e-8)
        
        # =====================================================================
        # FEATURE GROUP 6: Quantity Patterns
        # =====================================================================
        
        merged['qty_abs'] = merged['qty'].abs()
        merged['qty_ma5'] = merged['qty'].rolling(5, min_periods=1).mean()
        merged['qty_std5'] = merged['qty'].rolling(5, min_periods=1).std()
        
        # =====================================================================
        # LABEL
        # =====================================================================
        
        merged['label'] = merged['policy']
        
        all_obs.append(merged)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True).dropna()
    
    if verbose:
        n_baseline = (result['label'] == 0).sum()
        n_uniform = (result['label'] == 1).sum()
        n_total = len(result)
        
        feature_cols = [c for c in result.columns if c not in [
            'date', 'symbol', 'label', 'policy', 'policy_name',
            'price_trade', 'price_market', 'ref_price', 'side', 'qty',
            'date_only', 'side_numeric', 'direction_change'
        ]]
        
        print(f"[Bridge V2] Total observations: {n_total}")
        print(f"[Bridge V2] Baseline trades: {n_baseline} ({n_baseline/n_total*100:.1f}%)")
        print(f"[Bridge V2] Uniform trades: {n_uniform} ({n_uniform/n_total*100:.1f}%)")
        print(f"[Bridge V2] Features: {len(feature_cols)}")
        print(f"[Bridge V2] Symbols: {result['symbol'].nunique()}")
    
    return result


def prepare_adversary_data_v2(
    prices: pd.DataFrame,
    baseline_generate_fn,
    uniform_policy,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate adversary training data by creating trades from BOTH policies.
    
    Args:
        prices: Price data
        baseline_generate_fn: Function to generate baseline trades
        uniform_policy: UniformPolicy instance
        verbose: Print diagnostics
    
    Returns:
        DataFrame with features and binary labels (0=baseline, 1=uniform)
    """
    
    if verbose:
        print("[Bridge V2] Generating baseline trades...")
    
    # Generate baseline trades
    baseline_trades = baseline_generate_fn(prices)
    baseline_trades['policy'] = 0
    baseline_trades['policy_name'] = 'baseline'
    
    if verbose:
        print(f"[Bridge V2] → {len(baseline_trades)} baseline trades")
        print("[Bridge V2] Generating uniform trades...")
    
    # Generate uniform trades
    uniform_trades = uniform_policy.generate_trades(prices)
    uniform_trades['policy'] = 1
    uniform_trades['policy_name'] = 'uniform'
    
    if verbose:
        print(f"[Bridge V2] → {len(uniform_trades)} uniform trades")
        print("[Bridge V2] Engineering features...")
    
    # Combine
    all_trades = pd.concat([baseline_trades, uniform_trades], ignore_index=True)
    
    # Engineer features
    adversary_data = engineer_trade_features(all_trades, prices, verbose=verbose)
    
    return adversary_data


def time_split_data(data_df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple:
    """
    Chronological split preserving time-series order.
    
    Args:
        data_df: Data with 'date' column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        (train_df, val_df, test_df)
    """
    df = data_df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
