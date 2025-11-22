"""
Bridge: P4 Policy Output -> P7 Adversary Input

Converts trade data to daily signal format for P6-style adversarial task.

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np


def trades_to_daily_signals(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert P4 trades to daily signal format (P6-style).
    
    For each date and symbol:
    - signal = 1 if trade occurred
    - signal = 0 if no trade occurred
    
    Args:
        trades_df: Trade records with columns [date/timestamp, symbol, side, qty, price]
        prices_df: Full price history (daily)
    
    Returns:
        DataFrame with columns [date, symbol, signal]
    """
    # Ensure datetime
    if 'date' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['date'])
    elif 'timestamp' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
        trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # Mark days with trades
    trades_df['signal'] = 1
    trade_signals = trades_df.groupby(['date', 'symbol'])['signal'].max().reset_index()
    
    # Create full date x symbol grid from prices
    price_dates = pd.to_datetime(prices_df['date']).unique()
    symbols = trades_df['symbol'].unique()
    
    full_grid = pd.MultiIndex.from_product(
        [price_dates, symbols],
        names=['date', 'symbol']
    ).to_frame(index=False)
    
    # Merge to get signals (0 where no trade)
    signals_df = full_grid.merge(trade_signals, on=['date', 'symbol'], how='left')
    signals_df['signal'] = signals_df['signal'].fillna(0).astype(int)
    
    return signals_df


def enrich_with_price_features(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based features for adversary (P6-style).
    
    Features:
    - Price momentum (5, 10, 20 day)
    - Volatility (10, 30, 60 day)
    - Price levels
    
    Args:
        signals_df: Daily signals [date, symbol, signal]
        prices_df: Price history
    
    Returns:
        Enriched DataFrame with features
    """
    enriched_list = []
    
    for symbol in signals_df['symbol'].unique():
        # Filter data for this symbol
        symbol_signals = signals_df[signals_df['symbol'] == symbol].copy()
        symbol_prices = prices_df[prices_df['symbol'] == symbol].copy()
        
        # Merge prices
        symbol_data = symbol_signals.merge(
            symbol_prices[['date', 'close']], 
            on='date', 
            how='left'
        )
        
        # Sort by date
        symbol_data = symbol_data.sort_values('date')
        
        # Price momentum features
        symbol_data['mom_5d'] = symbol_data['close'].pct_change(5)
        symbol_data['mom_10d'] = symbol_data['close'].pct_change(10)
        symbol_data['mom_20d'] = symbol_data['close'].pct_change(20)
        
        # Volatility features
        returns = symbol_data['close'].pct_change()
        symbol_data['vol_10d'] = returns.rolling(10).std() * np.sqrt(252)
        symbol_data['vol_30d'] = returns.rolling(30).std() * np.sqrt(252)
        symbol_data['vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Price level features
        symbol_data['price_ma_20'] = symbol_data['close'].rolling(20).mean()
        symbol_data['price_ma_50'] = symbol_data['close'].rolling(50).mean()
        symbol_data['price_vs_ma20'] = symbol_data['close'] / symbol_data['price_ma_20']
        
        # Signal history (lagged)
        symbol_data['signal_lag1'] = symbol_data['signal'].shift(1)
        symbol_data['signal_lag2'] = symbol_data['signal'].shift(2)
        symbol_data['trades_last_5d'] = symbol_data['signal'].rolling(5).sum()
        
        enriched_list.append(symbol_data)
    
    enriched_df = pd.concat(enriched_list, ignore_index=True)
    
    # Add time features
    enriched_df['day_of_week'] = pd.to_datetime(enriched_df['date']).dt.dayofweek
    enriched_df['month'] = pd.to_datetime(enriched_df['date']).dt.month
    enriched_df['is_monday'] = (enriched_df['day_of_week'] == 0).astype(int)
    enriched_df['is_friday'] = (enriched_df['day_of_week'] == 4).astype(int)
    
    return enriched_df


def prepare_adversary_data(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline: Convert trades to adversary-ready format.
    
    Steps:
    1. Convert trades to daily signals
    2. Add price features
    3. Create labels (next-day prediction)
    
    Args:
        trades_df: Trade records
        prices_df: Price history
    
    Returns:
        DataFrame ready for adversary training
    """
    # Step 1: Trades to signals
    signals_df = trades_to_daily_signals(trades_df, prices_df)
    
    # Step 2: Add features
    enriched_df = enrich_with_price_features(signals_df, prices_df)
    
    # Step 3: Create labels (predict NEXT day's signal)
    enriched_df = enriched_df.sort_values(['symbol', 'date'])
    enriched_df['label'] = enriched_df.groupby('symbol')['signal'].shift(-1)
    
    # Drop last row per symbol (no future label)
    enriched_df = enriched_df.dropna(subset=['label'])
    enriched_df['label'] = enriched_df['label'].astype(int)
    
    return enriched_df
