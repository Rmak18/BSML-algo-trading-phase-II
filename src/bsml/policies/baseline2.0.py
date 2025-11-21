# baseline.py - Deterministic baseline strategy
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, time


class DeterministicBaseline:
    """
    Baseline policy: execute at fixed times with fixed thresholds.
    
    No randomization - serves as ground truth for adversary comparison.
    """
    
    def __init__(
        self,
        execution_time: time = time(10, 30),  # 10:30 AM
        threshold_offset: float = 0.0,        # No offset from mid
        respect_market_hours: bool = True
    ):
        self.execution_time = execution_time
        self.threshold_offset = threshold_offset
        self.respect_market_hours = respect_market_hours
    
    def generate_trades(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate deterministic trades from signals.
        
        Args:
            signals: DataFrame with [date, symbol, side, target_qty]
            prices: DataFrame with [date, symbol, open, high, low, close]
            
        Returns:
            DataFrame with [date, symbol, side, qty, exec_time, price, threshold]
        """
        trades = []
        
        for _, signal in signals.iterrows():
            date = signal['date']
            symbol = signal['symbol']
            
            # Get price data for this date/symbol
            price_row = prices[
                (prices['date'] == date) & 
                (prices['symbol'] == symbol)
            ]
            
            if len(price_row) == 0:
                continue
                
            price_row = price_row.iloc[0]
            
            # Deterministic execution time
            exec_datetime = pd.Timestamp.combine(
                pd.Timestamp(date).date(),
                self.execution_time
            )
            
            # Deterministic threshold (mid price + offset)
            mid_price = (price_row['high'] + price_row['low']) / 2
            threshold = mid_price * (1 + self.threshold_offset)
            
            # Use close price as execution price (simplified)
            exec_price = price_row['close']
            
            trades.append({
                'date': date,
                'symbol': symbol,
                'side': signal['side'],
                'qty': signal['target_qty'],
                'exec_time': exec_datetime,
                'price': exec_price,
                'threshold': threshold,
                'ref_price': mid_price
            })
        
        return pd.DataFrame(trades)


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy entry point for baseline policy.
    
    Expected columns in prices: ['date', 'symbol', 'open', 'high', 'low', 'close']
    Returns: DataFrame with ['date','symbol','side','qty','ref_price']
    """
    baseline = DeterministicBaseline()
    
    # Generate simple buy signals (for testing)
    signals = pd.DataFrame({
        'date': prices['date'].unique(),
        'symbol': prices['symbol'].iloc[0] if len(prices) > 0 else 'SPY',
        'side': 'BUY',
        'target_qty': 1000
    })
    
    return baseline.generate_trades(signals, prices)
