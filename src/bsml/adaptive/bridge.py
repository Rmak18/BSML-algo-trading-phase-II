"""
Bridge: P4 Policy Output → P7 Adversary Input

Converts P4's trade format to enriched format for adversary training.

P4 Output: date, symbol, side, qty, ref_price, price
P7 Needs: timestamp, symbol, mid_price, exec_flag, + metadata

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np


def enrich_trades_for_adversary(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    policy_id: str = 'uniform'
) -> pd.DataFrame:
    """
    Enrich P4 trade records for adversary training.
    
    Args:
        trades_df: Raw trades from P4 policy
        prices_df: Original price data (for reference)
        policy_id: Policy identifier
    
    Returns:
        Enriched DataFrame with all required columns
    """
    enriched = trades_df.copy()
    
    # Rename core columns
    if 'date' in enriched.columns:
        enriched.rename(columns={'date': 'timestamp'}, inplace=True)
    
    if 'price' in enriched.columns:
        enriched['mid_price'] = enriched['price']
    
    # Add metadata columns
    enriched['policy_id'] = policy_id
    enriched['exec_flag'] = 1  # All rows in trades_df are executions
    
    # Add default volume (not critical for adversary)
    if 'volume' not in enriched.columns:
        enriched['volume'] = 1000.0
    
    # PnL approximation (for future use)
    if 'pnl' not in enriched.columns:
        enriched['pnl'] = np.where(
            enriched['side'].str.upper() == 'BUY',
            -(enriched['mid_price'] - enriched.get('ref_price', enriched['mid_price'])) * enriched['qty'],
            (enriched['mid_price'] - enriched.get('ref_price', enriched['mid_price'])) * enriched['qty']
        )
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(enriched['timestamp']):
        enriched['timestamp'] = pd.to_datetime(enriched['timestamp'])
    
    # Clean and sort
    enriched = enriched.dropna(subset=['timestamp', 'symbol', 'mid_price'])
    enriched = enriched.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    return enriched
