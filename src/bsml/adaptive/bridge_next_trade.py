"""
P7 Bridge Module - Next-Trade Prediction Task

Task: Given recent trade history, predict if there will be a trade tomorrow.
Goal: Compare predictability between Baseline vs Uniform policies.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from typing import Tuple


def create_next_trade_dataset(trades: pd.DataFrame, prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Create dataset for next-trade prediction task.
    
    For each date, look at past `lookback` trades and predict if there's a trade tomorrow.
    
    Args:
        trades: Trade history from a single policy
        prices: Market price data
        lookback: Number of past trades to use as features
        
    Returns:
        DataFrame with features and binary labels (1 = trade tomorrow, 0 = no trade)
    """
    
    all_obs = []
    
    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_trades) < lookback + 2:
            continue
        
        # Ensure datetime
        symbol_trades['date'] = pd.to_datetime(symbol_trades['date'])
        symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
        
        # Sort
        symbol_trades = symbol_trades.sort_values('date').reset_index(drop=True)
        symbol_prices = symbol_prices.sort_values('date').reset_index(drop=True)
        
        # Get date range
        all_dates = pd.date_range(
            start=symbol_prices['date'].min(),
            end=symbol_prices['date'].max(),
            freq='D'
        )
        
        # Create observations for each date (starting after lookback period)
        for i in range(lookback, len(all_dates) - 1):
            current_date = all_dates[i]
            next_date = all_dates[i + 1]
            
            # Get past trades (up to lookback)
            past_trades = symbol_trades[symbol_trades['date'] < current_date].tail(lookback)
            
            if len(past_trades) < lookback:
                continue
            
            # Check if there's a trade tomorrow
            has_trade_tomorrow = len(symbol_trades[symbol_trades['date'] == next_date]) > 0
            
            # Get current market context
            current_price_row = symbol_prices[symbol_prices['date'] <= current_date].iloc[-1] if len(symbol_prices[symbol_prices['date'] <= current_date]) > 0 else None
            
            if current_price_row is None:
                continue
            
            # =====================================================================
            # FEATURE ENGINEERING: Past Trade Patterns
            # =====================================================================
            
            features = {
                'symbol': symbol,
                'date': current_date,
                'label': int(has_trade_tomorrow)
            }
            
            # 1. Time since last trade
            last_trade_date = past_trades.iloc[-1]['date']
            features['days_since_last_trade'] = (current_date - last_trade_date).days
            
            # 2. Trade frequency
            features['trades_last_5'] = len(past_trades)
            
            # 3. Inter-trade intervals
            if len(past_trades) >= 2:
                intervals = past_trades['date'].diff().dt.days.dropna()
                features['avg_interval'] = intervals.mean()
                features['std_interval'] = intervals.std() if len(intervals) > 1 else 0
                features['min_interval'] = intervals.min()
                features['max_interval'] = intervals.max()
            else:
                features['avg_interval'] = 0
                features['std_interval'] = 0
                features['min_interval'] = 0
                features['max_interval'] = 0
            
            # 4. Trade direction patterns
            past_trades['side_numeric'] = (past_trades['side'] == 'BUY').astype(int)
            features['pct_buy'] = past_trades['side_numeric'].mean()
            features['last_side'] = past_trades.iloc[-1]['side_numeric']
            
            # Direction changes
            if len(past_trades) >= 2:
                direction_changes = (past_trades['side_numeric'].diff() != 0).sum()
                features['direction_changes'] = direction_changes
            else:
                features['direction_changes'] = 0
            
            # 5. Quantity patterns
            features['avg_qty'] = past_trades['qty'].abs().mean()
            features['std_qty'] = past_trades['qty'].abs().std()
            
            # 6. Current market context
            features['current_price'] = current_price_row['price']
            
            # Returns (if available)
            if len(symbol_prices[symbol_prices['date'] <= current_date]) >= 5:
                recent_prices = symbol_prices[symbol_prices['date'] <= current_date].tail(5)
                features['returns_1d'] = recent_prices['price'].pct_change().iloc[-1]
                features['returns_5d'] = (recent_prices['price'].iloc[-1] / recent_prices['price'].iloc[0]) - 1
                features['vol_5d'] = recent_prices['price'].pct_change().std()
            else:
                features['returns_1d'] = 0
                features['returns_5d'] = 0
                features['vol_5d'] = 0
            
            # 7. Day of week (cyclical patterns)
            features['day_of_week'] = current_date.dayofweek
            features['day_of_month'] = current_date.day
            
            # 8. Streak features
            # How many consecutive days WITH trades
            features['consecutive_trade_days'] = 0
            check_date = current_date - pd.Timedelta(days=1)
            while len(symbol_trades[symbol_trades['date'] == check_date]) > 0:
                features['consecutive_trade_days'] += 1
                check_date -= pd.Timedelta(days=1)
            
            all_obs.append(features)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_obs)
    
    # Fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def prepare_next_trade_data(
    prices: pd.DataFrame,
    baseline_generator,
    uniform_policy,
    lookback: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate datasets for both baseline and uniform policies.
    
    Returns:
        (baseline_data, uniform_data): Both as prediction datasets
    """
    
    # Generate baseline trades
    if verbose:
        print("[Bridge Next-Trade] Generating baseline trades...")
    baseline_trades = baseline_generator(prices)
    if verbose:
        print(f"[Bridge Next-Trade] → {len(baseline_trades)} baseline trades")
    
    # Create baseline dataset
    if verbose:
        print(f"[Bridge Next-Trade] Creating baseline prediction dataset (lookback={lookback})...")
    baseline_data = create_next_trade_dataset(baseline_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive_baseline = (baseline_data['label'] == 1).sum()
        n_negative_baseline = (baseline_data['label'] == 0).sum()
        print(f"[Bridge Next-Trade] Baseline: {len(baseline_data)} observations")
        print(f"[Bridge Next-Trade]   → {n_positive_baseline} with trade tomorrow ({n_positive_baseline/len(baseline_data)*100:.1f}%)")
        print(f"[Bridge Next-Trade]   → {n_negative_baseline} no trade tomorrow ({n_negative_baseline/len(baseline_data)*100:.1f}%)")
    
    # Generate uniform trades
    if verbose:
        print("[Bridge Next-Trade] Generating uniform trades...")
    uniform_trades = uniform_policy.generate_trades(prices)
    if verbose:
        print(f"[Bridge Next-Trade] → {len(uniform_trades)} uniform trades")
    
    # Create uniform dataset
    if verbose:
        print(f"[Bridge Next-Trade] Creating uniform prediction dataset (lookback={lookback})...")
    uniform_data = create_next_trade_dataset(uniform_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive_uniform = (uniform_data['label'] == 1).sum()
        n_negative_uniform = (uniform_data['label'] == 0).sum()
        print(f"[Bridge Next-Trade] Uniform: {len(uniform_data)} observations")
        print(f"[Bridge Next-Trade]   → {n_positive_uniform} with trade tomorrow ({n_positive_uniform/len(uniform_data)*100:.1f}%)")
        print(f"[Bridge Next-Trade]   → {n_negative_uniform} no trade tomorrow ({n_negative_uniform/len(uniform_data)*100:.1f}%)")
    
    return baseline_data, uniform_data


def time_split_data(data: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test.
    
    Args:
        data: Dataset with 'date' column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        (train, val, test) DataFrames
    """
    
    data = data.sort_values('date').reset_index(drop=True)
    
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train = data.iloc[:train_size]
    val = data.iloc[train_size:train_size + val_size]
    test = data.iloc[train_size + val_size:]
    
    return train, val, test
