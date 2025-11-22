"""
P7 Bridge Module - CORRECT VERSION

Train adversary on BASELINE, test on RANDOMIZED policies

Owner: P7
Week: 4
"""

import numpy as np
import pandas as pd


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features"""
    prices = prices.copy()
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    result = []
    for symbol in prices['symbol'].unique():
        symbol_df = prices[prices['symbol'] == symbol].copy()
        
        # Returns
        symbol_df['returns_1d'] = symbol_df['price'].pct_change()
        symbol_df['returns_5d'] = symbol_df['price'].pct_change(5)
        symbol_df['returns_20d'] = symbol_df['price'].pct_change(20)
        
        # Volatility
        symbol_df['vol_5d'] = symbol_df['returns_1d'].rolling(5).std()
        symbol_df['vol_20d'] = symbol_df['returns_1d'].rolling(20).std()
        
        # Moving averages
        symbol_df['sma_5'] = symbol_df['price'].rolling(5).mean()
        symbol_df['sma_20'] = symbol_df['price'].rolling(20).mean()
        
        # Price ratios
        symbol_df['price_to_sma5'] = symbol_df['price'] / (symbol_df['sma_5'] + 1e-8)
        symbol_df['price_to_sma20'] = symbol_df['price'] / (symbol_df['sma_20'] + 1e-8)
        
        # Momentum
        symbol_df['momentum_5'] = symbol_df['returns_5d'] / (symbol_df['vol_5d'] + 1e-8)
        
        result.append(symbol_df)
    
    return pd.concat(result, ignore_index=True)


def prepare_adversary_data(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    prediction_window_hours: float = 4.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepare adversary dataset with features.
    
    Creates hourly observations and labels based on whether
    a trade occurs in the next X hours.
    """
    
    if len(trades) == 0 or len(prices) == 0:
        return pd.DataFrame()
    
    trades = trades.copy()
    prices = prices.copy()
    trades['date'] = pd.to_datetime(trades['date'], dayfirst=True)
    prices['date'] = pd.to_datetime(prices['date'], dayfirst=True)
    
    # Engineer features
    prices = engineer_features(prices)
    
    trades = trades.sort_values('date').reset_index(drop=True)
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    all_obs = []
    
    for symbol in prices['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_prices) < 20:
            continue
        
        # Create hourly observation grid
        min_date = symbol_prices['date'].min()
        max_date = symbol_prices['date'].max()
        obs_dates = pd.date_range(start=min_date, end=max_date, freq='1H')
        
        obs_data = []
        for obs_time in obs_dates:
            # Get most recent price data for features
            prior_prices = symbol_prices[symbol_prices['date'] <= obs_time]
            if len(prior_prices) == 0:
                continue
            
            nearest_price = prior_prices.iloc[-1].copy()
            
            # Label: will trade occur in next X hours?
            window_end = obs_time + pd.Timedelta(hours=prediction_window_hours)
            trades_in_window = symbol_trades[
                (symbol_trades['date'] > obs_time) & 
                (symbol_trades['date'] <= window_end)
            ]
            
            obs_row = nearest_price.to_dict()
            obs_row['date'] = obs_time
            obs_row['label'] = 1 if len(trades_in_window) > 0 else 0
            obs_row['signal'] = obs_row['label']
            
            obs_data.append(obs_row)
        
        if obs_data:
            symbol_df = pd.DataFrame(obs_data)
            all_obs.append(symbol_df)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True).dropna()
    
    if verbose:
        n_positive = result['label'].sum()
        n_total = len(result)
        feature_cols = [c for c in result.columns 
                       if c not in ['date', 'symbol', 'label', 'signal', 'price']]
        
        print(f"[Bridge] Total observations: {n_total}")
        print(f"[Bridge] Positive: {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"[Bridge] Negative: {n_total - n_positive}")
        print(f"[Bridge] Features: {len(feature_cols)} → {feature_cols}")
    
    return result
