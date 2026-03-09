#!/usr/bin/env python3
"""
Quick Test - Price Prediction Adversary
Run from: src/bsml/adaptive/

Usage:
    python test_regression.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK REGRESSION TEST - Price Prediction Adversary")
print("="*80)
print(f"Location: src/bsml/adaptive/")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Generate synthetic data
print("[1/3] Generating synthetic data...")
np.random.seed(42)
n = 500
symbols = ['SPY', 'QQQ', 'IVV']
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

baseline_trades = []
pink_trades = []

for i, date in enumerate(dates):
    symbol = symbols[i % len(symbols)]
    base_price = 100 + np.random.randn() * 10
    pink_price = base_price + np.random.randn() * 0.4  # 0.4% noise
    
    baseline_trades.append({
        'date': date.strftime('%Y-%m-%d'),
        'symbol': symbol,
        'side': 'BUY' if i % 2 == 0 else 'SELL',
        'price': base_price,
        'ref_price': base_price
    })
    
    pink_trades.append({
        'date': date.strftime('%Y-%m-%d'),
        'symbol': symbol,
        'side': 'BUY' if i % 2 == 0 else 'SELL',
        'price': base_price,
        'ref_price': pink_price
    })

baseline_df = pd.DataFrame(baseline_trades)
pink_df = pd.DataFrame(pink_trades)
print(f"  ✓ {len(baseline_df)} trades generated")

# Extract features
print("\n[2/3] Training adversary...")
features = pd.DataFrame()
features['baseline_price'] = baseline_df['price'].values
features['symbol'] = baseline_df['symbol'].values
features['side_binary'] = (baseline_df['side'] == 'BUY').astype(int).values

dates_obj = pd.to_datetime(baseline_df['date'])
features['day_of_week'] = dates_obj.dt.dayofweek.values
features['month'] = dates_obj.dt.month.values

features = pd.get_dummies(features, columns=['symbol'], prefix='symbol')
target = pink_df['ref_price'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

baseline_prices_test = X_test['baseline_price'].values

# Train strong adversary
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

absolute_errors = np.abs(y_test - y_pred)
pct_errors = (absolute_errors / baseline_prices_test) * 100
mae_pct = pct_errors.mean()
median_pct = np.median(pct_errors)

exploitable_threshold = 0.5
exploitable_fraction = (pct_errors < exploitable_threshold).mean()

print(f"  ✓ Model trained: 200 trees, depth 20")

# Results
print("\n[3/3] Results:")
print("="*80)
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
    print(f"  → Adversary can profit after transaction costs")
elif mae_pct < 1.0:
    print(f"\n  ⚠️  MODERATELY EXPLOITABLE - MAE < 1.0%")
    print(f"  → Adversary might profit in low-cost environments")
else:
    print(f"\n  ✓ SAFE - MAE > 1.0%")
    print(f"  → Randomization is strong enough")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
print("\nThis demonstrates:")
print("- Strong adversary design (200 trees, depth 20)")
print("- Economic interpretation (MAE% vs transaction costs)")
print("- Clear exploitability threshold (0.5%)")
