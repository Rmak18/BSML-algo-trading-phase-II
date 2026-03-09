"""
Feature Extraction for Adaptive Adversary Classifiers

Extracts distinguishing features for three binary classification tasks:
1. Baseline vs Pink Noise - Detects 1/f spectral signature
2. Baseline vs OU - Detects mean-reverting noise patterns
3. Baseline vs Uniform - Detects time jitter + uniform price noise

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats, signal


def extract_pink_noise_features(baseline_df: pd.DataFrame, pink_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features that distinguish Pink Noise from Baseline.
    
    Pink Noise has 1/f spectral signature - low frequencies dominate.
    
    Features:
    - price_deviation_pct: Percentage deviation from baseline
    - abs_deviation: Absolute price difference
    - deviation_autocorr_lag1: Autocorrelation at lag 1
    - deviation_autocorr_lag2: Autocorrelation at lag 2
    - rolling_std_deviation: Rolling standard deviation (window=10)
    - spectral_power_low_freq: Power in low frequency band
    
    Args:
        baseline_df: Baseline trades (price == ref_price)
        pink_df: Pink noise trades (ref_price has 1/f noise)
    
    Returns:
        DataFrame with features for each trade
    """
    
    # Ensure alignment
    assert len(baseline_df) == len(pink_df), "Trade counts must match"
    assert (baseline_df['date'] == pink_df['date']).all(), "Dates must align"
    assert (baseline_df['symbol'] == pink_df['symbol']).all(), "Symbols must align"
    
    features = pd.DataFrame()
    
    # Basic deviation features
    price_deviation = pink_df['ref_price'].values - baseline_df['price'].values
    features['price_deviation_pct'] = (price_deviation / baseline_df['price'].values) * 100
    features['abs_deviation'] = np.abs(price_deviation)
    
    # Autocorrelation features (per symbol to maintain temporal structure)
    autocorr_lag1 = []
    autocorr_lag2 = []
    rolling_std = []
    spectral_power = []
    
    for symbol in baseline_df['symbol'].unique():
        symbol_mask = baseline_df['symbol'] == symbol
        symbol_deviations = price_deviation[symbol_mask]
        
        # Autocorrelation
        if len(symbol_deviations) > 2:
            # Lag 1
            lag1 = np.corrcoef(symbol_deviations[:-1], symbol_deviations[1:])[0, 1]
            lag1 = lag1 if not np.isnan(lag1) else 0.0
            
            # Lag 2
            if len(symbol_deviations) > 3:
                lag2 = np.corrcoef(symbol_deviations[:-2], symbol_deviations[2:])[0, 1]
                lag2 = lag2 if not np.isnan(lag2) else 0.0
            else:
                lag2 = 0.0
        else:
            lag1 = lag2 = 0.0
        
        autocorr_lag1.extend([lag1] * symbol_mask.sum())
        autocorr_lag2.extend([lag2] * symbol_mask.sum())
        
        # Rolling standard deviation (window=10)
        symbol_series = pd.Series(symbol_deviations)
        rolling_std_vals = symbol_series.rolling(window=10, min_periods=1).std().fillna(0).values
        rolling_std.extend(rolling_std_vals)
        
        # Spectral power in low frequency band
        # Use FFT to detect 1/f signature
        if len(symbol_deviations) >= 16:
            # Compute power spectral density
            freqs, psd = signal.periodogram(symbol_deviations, fs=1.0)
            
            # Low frequency band: first 20% of frequencies
            low_freq_cutoff = int(len(freqs) * 0.2)
            low_freq_power = np.sum(psd[:low_freq_cutoff]) / np.sum(psd) if np.sum(psd) > 0 else 0.5
        else:
            low_freq_power = 0.5  # Neutral value
        
        spectral_power.extend([low_freq_power] * symbol_mask.sum())
    
    features['deviation_autocorr_lag1'] = autocorr_lag1
    features['deviation_autocorr_lag2'] = autocorr_lag2
    features['rolling_std_deviation'] = rolling_std
    features['spectral_power_low_freq'] = spectral_power
    
    # Add metadata for tracking
    features['symbol'] = baseline_df['symbol'].values
    features['date'] = baseline_df['date'].values
    
    return features


def extract_ou_features(baseline_df: pd.DataFrame, ou_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features that distinguish OU (mean-reverting) from Baseline.
    
    OU process exhibits mean reversion - deviations tend to return to zero.
    
    Features:
    - price_deviation_pct: Percentage deviation from baseline
    - abs_deviation: Absolute price difference
    - mean_reversion_signal: Whether consecutive deviations have opposite signs
    - deviation_change: Change in deviation from previous timestep
    - rolling_std_deviation: Rolling standard deviation
    - zero_crossing_frequency: How often deviations cross zero
    
    Args:
        baseline_df: Baseline trades
        ou_df: OU trades
    
    Returns:
        DataFrame with features
    """
    
    # Ensure alignment
    assert len(baseline_df) == len(ou_df), "Trade counts must match"
    assert (baseline_df['date'] == ou_df['date']).all(), "Dates must align"
    assert (baseline_df['symbol'] == ou_df['symbol']).all(), "Symbols must align"
    
    features = pd.DataFrame()
    
    # Basic deviation features
    price_deviation = ou_df['ref_price'].values - baseline_df['price'].values
    features['price_deviation_pct'] = (price_deviation / baseline_df['price'].values) * 100
    features['abs_deviation'] = np.abs(price_deviation)
    
    # Mean reversion features (per symbol)
    mean_reversion_signals = []
    deviation_changes = []
    rolling_std = []
    zero_crossing_freq = []
    
    for symbol in baseline_df['symbol'].unique():
        symbol_mask = baseline_df['symbol'] == symbol
        symbol_deviations = price_deviation[symbol_mask]
        
        # Mean reversion signal: consecutive deviations have opposite signs
        if len(symbol_deviations) > 1:
            signs = np.sign(symbol_deviations)
            sign_changes = signs[:-1] * signs[1:]
            # -1 means opposite signs (mean reversion), +1 means same sign
            reversion_signal = np.concatenate([[0], sign_changes])  # Pad first value
            mean_reversion_signals.extend(reversion_signal.tolist())
            
            # Deviation change
            dev_change = np.diff(symbol_deviations, prepend=symbol_deviations[0])
            deviation_changes.extend(dev_change.tolist())
        else:
            mean_reversion_signals.extend([0] * symbol_mask.sum())
            deviation_changes.extend([0] * symbol_mask.sum())
        
        # Rolling std
        symbol_series = pd.Series(symbol_deviations)
        rolling_std_vals = symbol_series.rolling(window=10, min_periods=1).std().fillna(0).values
        rolling_std.extend(rolling_std_vals)
        
        # Zero crossing frequency (per 20 trades)
        if len(symbol_deviations) >= 20:
            zero_crossings = np.sum(np.diff(np.sign(symbol_deviations)) != 0)
            crossing_freq = zero_crossings / len(symbol_deviations)
        else:
            crossing_freq = 0.0
        
        zero_crossing_freq.extend([crossing_freq] * symbol_mask.sum())
    
    features['mean_reversion_signal'] = mean_reversion_signals
    features['deviation_change'] = deviation_changes
    features['rolling_std_deviation'] = rolling_std
    features['zero_crossing_frequency'] = zero_crossing_freq
    
    # Add metadata
    features['symbol'] = baseline_df['symbol'].values
    features['date'] = baseline_df['date'].values
    
    return features


def extract_uniform_features(baseline_df: pd.DataFrame, uniform_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features that distinguish Uniform (time jitter + uniform noise) from Baseline.
    
    Uniform policy has:
    1. Intraday timestamps (not just dates)
    2. Uniform distribution of price noise
    
    Features:
    - price_deviation_pct: Percentage deviation from baseline
    - abs_deviation: Absolute price difference
    - has_intraday_timestamp: Boolean flag for sub-daily timing
    - hour_of_day: Hour component (0-23, or 0 if no time)
    - minute_of_day: Minute component (0-59, or 0 if no time)
    - time_since_last_trade_hours: Hours since previous trade
    - deviation_uniformity_score: KS test p-value for uniformity
    
    Args:
        baseline_df: Baseline trades
        uniform_df: Uniform trades
    
    Returns:
        DataFrame with features
    """
    
    # Ensure alignment
    assert len(baseline_df) == len(uniform_df), "Trade counts must match"
    
    features = pd.DataFrame()
    
    # Basic deviation features
    price_deviation = uniform_df['ref_price'].values - baseline_df['price'].values
    features['price_deviation_pct'] = (price_deviation / baseline_df['price'].values) * 100
    features['abs_deviation'] = np.abs(price_deviation)
    
    # Time features
    has_intraday = []
    hours = []
    minutes = []
    time_since_last = []
    
    for idx, date_str in enumerate(uniform_df['date']):
        # Check if timestamp has time component
        if isinstance(date_str, str) and ' ' in date_str:
            has_time = True
            try:
                dt = pd.to_datetime(date_str)
                hour = dt.hour
                minute = dt.minute
            except:
                hour = minute = 0
        else:
            has_time = False
            hour = minute = 0
        
        has_intraday.append(1 if has_time else 0)
        hours.append(hour)
        minutes.append(minute)
    
    features['has_intraday_timestamp'] = has_intraday
    features['hour_of_day'] = hours
    features['minute_of_day'] = minutes
    
    # Time since last trade (per symbol)
    for symbol in baseline_df['symbol'].unique():
        symbol_mask = uniform_df['symbol'] == symbol
        symbol_dates = pd.to_datetime(uniform_df.loc[symbol_mask, 'date'])
        
        # Calculate time differences
        time_diffs = symbol_dates.diff().dt.total_seconds() / 3600.0  # Convert to hours
        time_diffs = time_diffs.fillna(0).values
        
        # Map back to full array
        time_since_last.extend(time_diffs.tolist())
    
    features['time_since_last_trade_hours'] = time_since_last
    
    # Deviation uniformity score (per symbol)
    uniformity_scores = []
    
    for symbol in baseline_df['symbol'].unique():
        symbol_mask = baseline_df['symbol'] == symbol
        symbol_deviations = price_deviation[symbol_mask]
        
        if len(symbol_deviations) >= 10:
            # Normalize deviations to [0, 1]
            dev_min = symbol_deviations.min()
            dev_max = symbol_deviations.max()
            if dev_max > dev_min:
                normalized = (symbol_deviations - dev_min) / (dev_max - dev_min)
                
                # KS test against uniform distribution
                ks_stat, p_value = stats.kstest(normalized, 'uniform')
                uniformity_score = p_value  # High p-value = uniform
            else:
                uniformity_score = 0.0
        else:
            uniformity_score = 0.5  # Neutral
        
        uniformity_scores.extend([uniformity_score] * symbol_mask.sum())
    
    features['deviation_uniformity_score'] = uniformity_scores
    
    # Add metadata
    features['symbol'] = baseline_df['symbol'].values
    features['date'] = baseline_df['date'].values
    
    return features


def create_binary_labels(n_baseline: int, n_policy: int) -> np.ndarray:
    """
    Create binary labels for classification.
    
    Args:
        n_baseline: Number of baseline samples
        n_policy: Number of policy samples
    
    Returns:
        Array of labels: 0 for baseline, 1 for policy
    """
    return np.concatenate([
        np.zeros(n_baseline, dtype=int),
        np.ones(n_policy, dtype=int)
    ])


def combine_features_for_binary_classification(
    baseline_features: pd.DataFrame,
    policy_features: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Combine baseline and policy features for binary classification.
    
    Args:
        baseline_features: Features extracted from baseline trades
        policy_features: Features extracted from policy trades
    
    Returns:
        Tuple of (combined_features, labels)
    """
    
    # Drop metadata columns for training
    feature_cols = [col for col in baseline_features.columns 
                    if col not in ['symbol', 'date']]
    
    X_baseline = baseline_features[feature_cols]
    X_policy = policy_features[feature_cols]
    
    # Combine
    X = pd.concat([X_baseline, X_policy], ignore_index=True)
    y = create_binary_labels(len(X_baseline), len(X_policy))
    
    return X, y


if __name__ == "__main__":
    """Test feature extraction"""
    
    print("="*80)
    print("FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Create synthetic test data
    np.random.seed(42)
    n = 100
    
    baseline_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'symbol': ['TEST'] * n,
        'price': 100 + np.random.randn(n) * 2,
        'ref_price': 100 + np.random.randn(n) * 2
    })
    baseline_df['ref_price'] = baseline_df['price']  # Baseline: price == ref_price
    
    # Pink noise: add 1/f noise
    pink_noise = np.random.randn(n).cumsum() * 0.5  # Brownian motion approximation
    pink_df = baseline_df.copy()
    pink_df['ref_price'] = baseline_df['price'] + pink_noise
    
    # OU: add mean-reverting noise
    ou_noise = np.zeros(n)
    for i in range(1, n):
        ou_noise[i] = 0.9 * ou_noise[i-1] + np.random.randn() * 0.5
    ou_df = baseline_df.copy()
    ou_df['ref_price'] = baseline_df['price'] + ou_noise
    
    # Uniform: add uniform noise + time jitter
    uniform_df = baseline_df.copy()
    uniform_df['ref_price'] = baseline_df['price'] + np.random.uniform(-2, 2, n)
    uniform_df['date'] = uniform_df['date'].apply(
        lambda x: x + pd.Timedelta(minutes=np.random.randint(-30, 30))
    )
    
    # Extract features
    print("\n1. Pink Noise Features:")
    pink_features = extract_pink_noise_features(baseline_df, pink_df)
    print(pink_features.head())
    print(f"Shape: {pink_features.shape}")
    
    print("\n2. OU Features:")
    ou_features = extract_ou_features(baseline_df, ou_df)
    print(ou_features.head())
    print(f"Shape: {ou_features.shape}")
    
    print("\n3. Uniform Features:")
    uniform_features = extract_uniform_features(baseline_df, uniform_df)
    print(uniform_features.head())
    print(f"Shape: {uniform_features.shape}")
    
    print("\n✓ Feature extraction successful!")
