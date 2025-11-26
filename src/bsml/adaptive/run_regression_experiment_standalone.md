#!/usr/bin/env python3
"""
Standalone Regression Experiment Runner
Can be executed directly on GitHub Codespaces or any Python environment

Usage:
    python run_regression_experiment_standalone.py

Owner: P7 (Matteo)
Week: 3
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STANDALONE REGRESSION-BASED ADVERSARY EXPERIMENT")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Generate synthetic data
print("[Step 1] Generating synthetic trade data...")

np.random.seed(42)
n_days = 250
symbols = ['SPY', 'QQQ', 'IVV', 'VOO', 'VTI']
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

all_baseline_trades = []
all_pink_trades = []
all_ou_trades = []
all_uniform_trades = []

for symbol in symbols:
    base_price = 100 + np.random.randn(n_days).cumsum() * 0.5
    base_price = np.maximum(base_price, 50)
    
    # Baseline
    for i, (date, price) in enumerate(zip(dates, base_price)):
        all_baseline_trades.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'qty': 100,
            'price': price,
            'ref_price': price
        })
    
    # Pink Noise
    pink_noise = np.random.randn(n_days).cumsum() * 0.3
    pink_prices = base_price + pink_noise
    
    for i, (date, orig_price, pink_price) in enumerate(zip(dates, base_price, pink_prices)):
        all_pink_trades.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'qty': 100,
            'price': orig_price,
            'ref_price': pink_price
        })
    
    # OU Process
    ou_noise = np.zeros(n_days)
    ou_noise[0] = np.random.randn() * 0.5
    theta, sigma = 0.15, 0.02
    for i in range(1, n_days):
        ou_noise[i] = ou_noise[i-1] - theta * ou_noise[i-1] + sigma * np.random.randn()
    ou_prices = base_price + ou_noise * base_price * 0.03
    
    for i, (date, orig_price, ou_price) in enumerate(zip(dates, base_price, ou_prices)):
        all_ou_trades.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'qty': 100,
            'price': orig_price,
            'ref_price': ou_price
        })
    
    # Uniform
    uniform_noise = (np.random.rand(n_days) - 0.5) * 2 * 0.03
    uniform_prices = base_price * (1 + uniform_noise)
    
    for i, (date, orig_price, uniform_price) in enumerate(zip(dates, base_price, uniform_prices)):
        all_uniform_trades.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'qty': 100,
            'price': orig_price,
            'ref_price': uniform_price
        })

baseline_df = pd.DataFrame(all_baseline_trades)
pink_df = pd.DataFrame(all_pink_trades)
ou_df = pd.DataFrame(all_ou_trades)
uniform_df = pd.DataFrame(all_uniform_trades)

print(f"  ✓ Generated {len(baseline_df)} trades for {len(symbols)} symbols")

# Feature extraction
print("\n[Step 2] Extracting features...")

def extract_features(baseline_df, policy_df):
    features = pd.DataFrame()
    features['baseline_price'] = baseline_df['price'].values
    features['symbol'] = baseline_df['symbol'].values
    features['side_binary'] = (baseline_df['side'] == 'BUY').astype(int).values
    
    dates = pd.to_datetime(baseline_df['date'])
    features['day_of_week'] = dates.dt.dayofweek.values
    features['month'] = dates.dt.month.values
    
    features = pd.get_dummies(features, columns=['symbol'], prefix='symbol')
    target = policy_df['ref_price'].values
    
    return features, target

X_pink, y_pink = extract_features(baseline_df, pink_df)
X_ou, y_ou = extract_features(baseline_df, ou_df)
X_uniform, y_uniform = extract_features(baseline_df, uniform_df)

print(f"  ✓ Features extracted: {X_pink.shape[1]} features, {X_pink.shape[0]} samples")

# Train adversaries
print("\n[Step 3] Training price prediction adversaries...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

results = {}

for policy_name, X, y, baseline_prices in [
    ('Pink Noise', X_pink, y_pink, baseline_df['price'].values),
    ('OU Process', X_ou, y_ou, baseline_df['price'].values),
    ('Uniform', X_uniform, y_uniform, baseline_df['price'].values)
]:
    print(f"\n{'='*80}")
    print(f"ADVERSARY: Predicting {policy_name} Prices")
    print(f"{'='*80}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    baseline_prices_test = X_test['baseline_price'].values
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    absolute_errors = np.abs(y_test - y_pred)
    pct_errors = (absolute_errors / baseline_prices_test) * 100
    mae_pct = pct_errors.mean()
    median_pct = np.median(pct_errors)
    
    exploitable_threshold = 0.5
    exploitable_fraction = (pct_errors < exploitable_threshold).mean()
    
    results[policy_name] = {
        'mae': mae,
        'r2': r2,
        'mae_pct': mae_pct,
        'median_pct': median_pct,
        'exploitable_fraction': exploitable_fraction
    }
    
    print(f"\nAbsolute Errors:")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\nPercentage Errors (KEY METRIC):")
    print(f"  Mean:   {mae_pct:.4f}%")
    print(f"  Median: {median_pct:.4f}%")
    
    print(f"\nExploitability Analysis:")
    print(f"  Trades predictable within 0.5%: {exploitable_fraction*100:.1f}%")
    
    if mae_pct < 0.5:
        print(f"\n  ⚠️  HIGHLY EXPLOITABLE - MAE < 0.5%")
    elif mae_pct < 1.0:
        print(f"\n  ⚠️  MODERATELY EXPLOITABLE - MAE < 1.0%")
    else:
        print(f"\n  ✓ SAFE - MAE > 1.0%")
    
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Predictive Features:")
    print(feature_importance.head().to_string(index=False))

# Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary_df = pd.DataFrame.from_dict(results, orient='index')
summary_df = summary_df.round(4)
summary_df['exploitable_fraction'] = (summary_df['exploitable_fraction'] * 100).round(1)
summary_df.columns = ['MAE ($)', 'R²', 'MAE%', 'Median%', 'Exploitable (%)']

print("\n" + summary_df.to_string())

print("\n" + "="*80)
print("ADAPTATION DECISIONS")
print("="*80)
print(f"Threshold: 0.5% MAE\n")

for policy_name, metrics in results.items():
    status = "⚠️ ADAPT" if metrics['mae_pct'] < 0.5 else "✓ SAFE"
    print(f"{policy_name:15s} MAE%: {metrics['mae_pct']:6.4f}%  →  {status}")

summary_df.to_csv('regression_results_summary.csv')
print(f"\n✓ Results saved to: regression_results_summary.csv")

print("\n" + "="*80)
print("✅ EXPERIMENT COMPLETE")
print("="*80)
