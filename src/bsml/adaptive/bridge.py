"""
Bridge between P4 policies and P6 adversary.

Converts P4 trade format to P6 feature format.

P4 produces: date, symbol, side, qty, ref_price, price
P6 expects: timestamp, symbol, mid_price, volume, realized_cost_lag,
            estimated_slippage_t_1, policy_id, exec_flag, pnl

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from typing import Optional


def enrich_trades_for_adversary(
    trades_df: pd.DataFrame, 
    prices_df: pd.DataFrame,
    policy_id: str = 'adaptive_uniform'
) -> pd.DataFrame:
    """
    Convert P4 policy output to P6 adversary input format.
    
    This is a CRITICAL bridge function that adapts between:
    - P4's trade schema (date, symbol, side, qty, ref_price, price)
    - P6's event schema (timestamp, symbol, mid_price, volume, etc.)
    
    Args:
        trades_df: Output from P4 policy.generate_trades()
        prices_df: Original price data (for context)
        policy_id: Identifier for this policy run
        
    Returns:
        DataFrame ready for P6's extract_features()
    """
    enriched = trades_df.copy()
    
    # ========================================================================
    # COLUMN RENAMING (P4 → P6)
    # ========================================================================
    
    # 1. date → timestamp (P6 expects 'timestamp')
    if 'date' in enriched.columns:
        enriched = enriched.rename(columns={'date': 'timestamp'})
    
    # 2. price → mid_price (P6 expects 'mid_price')
    if 'price' in enriched.columns and 'mid_price' not in enriched.columns:
        enriched['mid_price'] = enriched['price']
    
    # ========================================================================
    # ADD MISSING COLUMNS (with reasonable defaults/approximations)
    # ========================================================================
    
    # 3. Volume: merge from prices if available, else use default
    if 'volume' not in enriched.columns:
        if 'volume' in prices_df.columns:
            # Merge volume from original prices
            volume_map = prices_df.set_index(['date', 'symbol'])['volume'].to_dict()
            enriched['volume'] = enriched.apply(
                lambda row: volume_map.get((row['timestamp'], row['symbol']), 1000.0),
                axis=1
            )
        else:
            # Default volume
            enriched['volume'] = 1000.0
    
    # 4. Policy ID (identifies which policy generated these trades)
    enriched['policy_id'] = policy_id
    
    # 5. Execution flag (binary: 1 = trade executed, 0 = no trade)
    # For P7's purposes, all rows in trades_df represent executions
    enriched['exec_flag'] = 1
    
    # 6. Cost estimates (placeholders - P3 will provide real values in v1.0)
    if 'realized_cost_lag' not in enriched.columns:
        enriched['realized_cost_lag'] = 0.01  # 1 bps default
    
    if 'estimated_slippage_t_1' not in enriched.columns:
        enriched['estimated_slippage_t_1'] = 0.02  # 2 bps default
    
    # 7. PnL (placeholder - approximate from slippage)
    if 'pnl' not in enriched.columns:
        enriched['pnl'] = np.where(
            enriched['side'].str.upper() == 'BUY',
            -(enriched['mid_price'] - enriched['ref_price']) * enriched['qty'],
            (enriched['mid_price'] - enriched['ref_price']) * enriched['qty']
        )
    
    # 8. Action side (-1 for sell, +1 for buy)
    if 'action_side' not in enriched.columns:
        enriched['action_side'] = np.where(
            enriched['side'].str.upper() == 'BUY', 1, -1
        )
    
    # 9. Action size (absolute quantity)
    if 'action_size' not in enriched.columns:
        enriched['action_size'] = enriched['qty'].abs()
    
    # 10. Market order flag (assume market orders for simplicity)
    if 'is_market_order' not in enriched.columns:
        enriched['is_market_order'] = 1
    
    # ========================================================================
    # DATA CLEANING
    # ========================================================================
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(enriched['timestamp']):
        enriched['timestamp'] = pd.to_datetime(enriched['timestamp'])
    
    # Drop any rows with missing critical fields
    critical_cols = ['timestamp', 'symbol', 'mid_price']
    enriched = enriched.dropna(subset=critical_cols)
    
    # Sort by symbol and timestamp (required by P6's feature extraction)
    enriched = enriched.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    return enriched
