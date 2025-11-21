"""
Bridge: P4 Policy Output → P7 Adversary Input

Converts P4's trade format to enriched format for adversary training.

Key transformations:
- Standardize column names (date → timestamp)
- Add metadata (policy_id, exec_flag)
- Ensure datetime types
- Add derived features (PnL approximation)
- Clean and validate data

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings


def validate_trades(trades_df: pd.DataFrame) -> tuple:
    """
    Validate trade data before enrichment.
    
    Returns:
        (is_valid, error_message)
    """
    required_cols = ['symbol']
    
    # Check required columns
    missing = [c for c in required_cols if c not in trades_df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Check for empty
    if len(trades_df) == 0:
        return False, "Empty trades DataFrame"
    
    # Check for valid symbols
    if trades_df['symbol'].isna().any():
        return False, "NaN values in symbol column"
    
    return True, "OK"


def standardize_timestamp(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure timestamp column exists and is properly formatted.
    
    Handles multiple input formats:
    - 'date' column → rename to 'timestamp'
    - 'timestamp' column → ensure datetime type
    """
    df = trades_df.copy()
    
    # Rename date → timestamp
    if 'date' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'date': 'timestamp'}, inplace=True)
    
    # Ensure datetime type
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                warnings.warn(f"Could not convert timestamp to datetime: {e}")
    else:
        raise ValueError("No date/timestamp column found in trades!")
    
    return df


def add_metadata(trades_df: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    """
    Add metadata columns required by adversary.
    
    Metadata:
    - policy_id: Identifies which policy generated these trades
    - exec_flag: All rows are executions (=1)
    """
    df = trades_df.copy()
    
    df['policy_id'] = policy_id
    df['exec_flag'] = 1  # All rows in P4 output are executions
    
    return df


def add_price_fields(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure mid_price exists for adversary feature engineering.
    
    Priority:
    1. Use 'mid_price' if exists
    2. Use 'price' as fallback
    3. Use 'ref_price' as last resort
    """
    df = trades_df.copy()
    
    if 'mid_price' not in df.columns:
        if 'price' in df.columns:
            df['mid_price'] = df['price']
        elif 'ref_price' in df.columns:
            df['mid_price'] = df['ref_price']
        else:
            raise ValueError("No price column found (mid_price, price, or ref_price)")
    
    return df


def add_derived_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that might be useful for adversary.
    
    Features:
    - volume: Default if missing
    - pnl: Approximate PnL (if ref_price exists)
    """
    df = trades_df.copy()
    
    # Default volume
    if 'volume' not in df.columns:
        df['volume'] = 1000.0
    
    # Approximate PnL (useful for future extensions)
    if 'pnl' not in df.columns and 'ref_price' in df.columns:
        if 'side' in df.columns and 'qty' in df.columns:
            df['pnl'] = np.where(
                df['side'].str.upper() == 'BUY',
                -(df['mid_price'] - df['ref_price']) * df['qty'].abs(),
                (df['mid_price'] - df['ref_price']) * df['qty'].abs()
            )
    
    return df


def clean_and_sort(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning and sorting.
    
    Steps:
    1. Drop rows with missing critical fields
    2. Sort by symbol and timestamp
    3. Reset index
    """
    df = trades_df.copy()
    
    # Drop rows with missing critical fields
    critical_cols = ['timestamp', 'symbol', 'mid_price']
    df = df.dropna(subset=critical_cols)
    
    # Sort
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    return df


def enrich_trades_for_adversary(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    policy_id: str = 'uniform'
) -> pd.DataFrame:
    """
    Main enrichment pipeline: P4 trades → P7 adversary input.
    
    Pipeline:
    1. Validate input
    2. Standardize timestamp
    3. Add metadata
    4. Add price fields
    5. Add derived features
    6. Clean and sort
    
    Args:
        trades_df: Raw trades from P4 policy
        prices_df: Original price data (for reference, currently unused)
        policy_id: Policy identifier
    
    Returns:
        Enriched DataFrame ready for adversary training
    
    Raises:
        ValueError: If validation fails
    """
    # Validate
    is_valid, error_msg = validate_trades(trades_df)
    if not is_valid:
        raise ValueError(f"Trade validation failed: {error_msg}")
    
    # Pipeline
    enriched = trades_df.copy()
    enriched = standardize_timestamp(enriched)
    enriched = add_metadata(enriched, policy_id)
    enriched = add_price_fields(enriched)
    enriched = add_derived_features(enriched)
    enriched = clean_and_sort(enriched)
    
    # Final validation
    if len(enriched) == 0:
        raise ValueError("Enrichment resulted in empty DataFrame!")
    
    return enriched


def get_enrichment_stats(enriched_df: pd.DataFrame) -> dict:
    """
    Get statistics about enriched trades (for debugging).
    
    Returns:
        Dict with counts, date ranges, symbols
    """
    return {
        'n_rows': len(enriched_df),
        'n_symbols': enriched_df['symbol'].nunique(),
        'symbols': sorted(enriched_df['symbol'].unique().tolist()),
        'date_range': {
            'start': enriched_df['timestamp'].min(),
            'end': enriched_df['timestamp'].max()
        },
        'columns': enriched_df.columns.tolist()
    }
