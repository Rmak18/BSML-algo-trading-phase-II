"""
P7 Bridge Module - 3-Policy Comparison (Next-Trade Prediction)

Task: Compare predictability across Baseline, Uniform, Pink Noise, and OU policies.
Goal: Identify which randomization strategy best evades pattern detection.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


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


def prepare_three_policy_data(
    prices: pd.DataFrame,
    baseline_generator,
    uniform_policy,
    pink_policy,
    ou_policy,
    lookback: int = 5,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate datasets for baseline and all 3 randomization policies.
    
    Returns:
        {
            'baseline': DataFrame,
            'uniform': DataFrame,
            'pink_noise': DataFrame,
            'ou': DataFrame
        }
    """
    
    datasets = {}
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    
    if verbose:
        print("[Bridge 3-Policy] Generating baseline trades...")
    baseline_trades = baseline_generator(prices)
    if verbose:
        print(f"[Bridge 3-Policy] → {len(baseline_trades)} baseline trades")
    
    if verbose:
        print(f"[Bridge 3-Policy] Creating baseline prediction dataset (lookback={lookback})...")
    baseline_data = create_next_trade_dataset(baseline_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive = (baseline_data['label'] == 1).sum()
        n_negative = (baseline_data['label'] == 0).sum()
        print(f"[Bridge 3-Policy] Baseline: {len(baseline_data)} observations")
        print(f"[Bridge 3-Policy]   → {n_positive} with trade tomorrow ({n_positive/len(baseline_data)*100:.1f}%)")
        print(f"[Bridge 3-Policy]   → {n_negative} no trade tomorrow ({n_negative/len(baseline_data)*100:.1f}%)")
    
    datasets['baseline'] = baseline_data
    
    # =========================================================================
    # UNIFORM
    # =========================================================================
    
    if verbose:
        print("[Bridge 3-Policy] Generating uniform trades...")
    uniform_trades = uniform_policy.generate_trades(prices)
    if verbose:
        print(f"[Bridge 3-Policy] → {len(uniform_trades)} uniform trades")
    
    if verbose:
        print(f"[Bridge 3-Policy] Creating uniform prediction dataset...")
    uniform_data = create_next_trade_dataset(uniform_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive = (uniform_data['label'] == 1).sum()
        n_negative = (uniform_data['label'] == 0).sum()
        print(f"[Bridge 3-Policy] Uniform: {len(uniform_data)} observations")
        print(f"[Bridge 3-Policy]   → {n_positive} with trade tomorrow ({n_positive/len(uniform_data)*100:.1f}%)")
        print(f"[Bridge 3-Policy]   → {n_negative} no trade tomorrow ({n_negative/len(uniform_data)*100:.1f}%)")
    
    datasets['uniform'] = uniform_data
    
    # =========================================================================
    # PINK NOISE
    # =========================================================================
    
    if verbose:
        print("[Bridge 3-Policy] Generating pink noise trades...")
    pink_trades = pink_policy.generate_trades(prices)
    if verbose:
        print(f"[Bridge 3-Policy] → {len(pink_trades)} pink noise trades")
    
    if verbose:
        print(f"[Bridge 3-Policy] Creating pink noise prediction dataset...")
    pink_data = create_next_trade_dataset(pink_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive = (pink_data['label'] == 1).sum()
        n_negative = (pink_data['label'] == 0).sum()
        print(f"[Bridge 3-Policy] Pink Noise: {len(pink_data)} observations")
        print(f"[Bridge 3-Policy]   → {n_positive} with trade tomorrow ({n_positive/len(pink_data)*100:.1f}%)")
        print(f"[Bridge 3-Policy]   → {n_negative} no trade tomorrow ({n_negative/len(pink_data)*100:.1f}%)")
    
    datasets['pink_noise'] = pink_data
    
    # =========================================================================
    # ORNSTEIN-UHLENBECK
    # =========================================================================
    
    if verbose:
        print("[Bridge 3-Policy] Generating OU trades...")
    ou_trades = ou_policy.generate_trades(prices)
    if verbose:
        print(f"[Bridge 3-Policy] → {len(ou_trades)} OU trades")
    
    if verbose:
        print(f"[Bridge 3-Policy] Creating OU prediction dataset...")
    ou_data = create_next_trade_dataset(ou_trades, prices, lookback=lookback)
    
    if verbose:
        n_positive = (ou_data['label'] == 1).sum()
        n_negative = (ou_data['label'] == 0).sum()
        print(f"[Bridge 3-Policy] OU: {len(ou_data)} observations")
        print(f"[Bridge 3-Policy]   → {n_positive} with trade tomorrow ({n_positive/len(ou_data)*100:.1f}%)")
        print(f"[Bridge 3-Policy]   → {n_negative} no trade tomorrow ({n_negative/len(ou_data)*100:.1f}%)")
    
    datasets['ou'] = ou_data
    
    return datasets


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
