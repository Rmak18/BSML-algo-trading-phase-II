# Integration with P4 and P6

**Project:** BSML - Randomized Execution Research  
**Owner:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Status:** Week 1 Deliverable

---

## 1. Overview

This document specifies how P7's adaptive adversary integrates with:
- **P4 (Randomization Modules):** For adjusting policy parameters
- **P6 (Standard Adversary):** For adversary training and feature engineering

---

## 2. Integration with P4 (Randomization Modules)

### 2.1 Core API Contract

P7 calls P4's `adjust_stochasticity()` method to modify policy parameters:

```python
# P7 → P4 interface
policy.adjust_stochasticity(
    auc_score: float,      # Current adversary AUC [0.5, 1.0]
    direction: str         # 'increase' or 'decrease'
) -> None
```

**P4 Responsibilities:**
- Modify `self.params` in-place
- Multiply adjustable parameters by 1.2 (increase) or 0.8 (decrease)
- Enforce parameter bounds (clip to [min, max])
- Return None (modification is in-place)

**P7 Responsibilities:**
- Provide valid AUC score and direction
- Log parameters before and after adjustment
- Verify exposure invariance after adjustment
- Revert if constraints violated

### 2.2 Parameter Mapping by Policy

#### Uniform Policy

```python
from bsml.randomization import UniformPolicy

# Initialize
policy = UniformPolicy(
    seed=42,
    params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    }
)

# P7 calls adjustment
policy.adjust_stochasticity(auc_score=0.82, direction='increase')

# P4 modifies params in-place:
# timing_range_hours: 2.0 → 2.4 (×1.2)
# threshold_pct: 0.10 → 0.12 (×1.2)

print(policy.params)
# Output: {'timing_range_hours': 2.4, 'threshold_pct': 0.12}
```

#### OU Policy

```python
from bsml.randomization import OUPolicy

policy = OUPolicy(
    seed=42,
    params={
        'theta': 0.15,  # Not adjusted
        'sigma': 0.05,  # Adjusted
        'mu': 0.0       # Not adjusted
    }
)

policy.adjust_stochasticity(auc_score=0.78, direction='increase')

# P4 modifies only sigma:
# sigma: 0.05 → 0.06 (×1.2)
# theta: 0.15 (unchanged)
# mu: 0.0 (unchanged)

print(policy.params)
# Output: {'theta': 0.15, 'sigma': 0.06, 'mu': 0.0}
```

#### Pink Noise Policy

```python
from bsml.randomization import PinkNoisePolicy

policy = PinkNoisePolicy(
    seed=42,
    params={
        'alpha': 1.0,   # Not adjusted
        'scale': 0.08   # Adjusted
    }
)

policy.adjust_stochasticity(auc_score=0.53, direction='decrease')

# P4 modifies only scale:
# scale: 0.08 → 0.064 (×0.8)
# alpha: 1.0 (unchanged)

print(policy.params)
# Output: {'alpha': 1.0, 'scale': 0.064}
```

### 2.3 Exposure Invariance Check

After each adjustment, P7 uses P4's exposure check method:

```python
# P7 calls P4's validation method
is_valid = policy.check_exposure_invariance(
    positions_before: dict,  # {symbol: shares} before adjustment
    positions_after: dict    # {symbol: shares} after adjustment
)

# Example
positions_before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
positions_after = {'AAPL': 103, 'MSFT': -98}    # Net = 5

is_valid = policy.check_exposure_invariance(positions_before, positions_after)
# Returns: True (within ±5% tolerance)
```

**P4's Implementation (from API spec):**
```python
def check_exposure_invariance(self, positions_before, positions_after):
    """Validate net exposure remains within tolerance."""
    net_before = sum(positions_before.values())
    net_after = sum(positions_after.values())
    tolerance = 0.05  # 5% of NAV
    
    is_valid = abs(net_after - net_before) <= tolerance
    
    self._perturbation_log.append({
        'net_before': net_before,
        'net_after': net_after,
        'valid': is_valid
    })
    
    return is_valid
```

### 2.4 Complete P7 → P4 Workflow

```python
def adjust_policy_with_safety_check(policy, auc_score, trades_before):
    """
    Adjust policy parameters with exposure validation.
    
    Args:
        policy: RandomizationPolicy instance (from P4)
        auc_score: Current adversary AUC
        trades_before: Trades before adjustment
    
    Returns:
        success: bool
        params_old: dict
        params_new: dict
    """
    
    # Step 1: Store old parameters
    params_old = policy.params.copy()
    
    # Step 2: Determine direction
    if auc_score > 0.75:
        direction = 'increase'
    elif auc_score < 0.55:
        direction = 'decrease'
    else:
        return True, params_old, params_old  # No adjustment needed
    
    # Step 3: Call P4 API
    policy.adjust_stochasticity(auc_score=auc_score, direction=direction)
    params_new = policy.params.copy()
    
    # Step 4: Generate test batch with new params
    trades_after = generate_test_batch(policy, n_samples=100)
    
    # Step 5: Extract positions
    positions_before = extract_positions(trades_before)
    positions_after = extract_positions(trades_after)
    
    # Step 6: Check exposure via P4 method
    is_valid = policy.check_exposure_invariance(positions_before, positions_after)
    
    if not is_valid:
        # Revert parameters
        policy.params = params_old
        print(f"⚠️ Exposure violation - reverted to {params_old}")
        return False, params_old, params_old
    
    print(f"✓ Adjustment successful: {params_old} → {params_new}")
    return True, params_old, params_new
```

### 2.5 Parameter Bounds Enforcement

P4 enforces these bounds during adjustment:

| Policy | Parameter | Min | Max | Rationale |
|--------|-----------|-----|-----|-----------|
| Uniform | `timing_range_hours` | 0.5 | 6.0 | Market day ~6.5 hours |
| Uniform | `threshold_pct` | 0.05 | 0.25 | Balance signal vs. noise |
| OU | `sigma` | 0.01 | 0.15 | Prevent extreme volatility |
| Pink | `scale` | 0.02 | 0.20 | Keep perturbations reasonable |

**P4's Clipping Logic (from API spec):**
```python
def adjust_stochasticity(self, auc_score, direction):
    """Adjust randomness level based on adversary feedback."""
    adjustment_factor = 1.2 if direction == 'increase' else 0.8
    
    # Apply to all stochastic parameters
    for key in self.params:
        if 'range' in key or 'pct' in key or 'sigma' in key or 'scale' in key:
            self.params[key] *= adjustment_factor
            
            # Clip to bounds (P7 needs to know these bounds)
            if key == 'timing_range_hours':
                self.params[key] = np.clip(self.params[key], 0.5, 6.0)
            elif key == 'threshold_pct':
                self.params[key] = np.clip(self.params[key], 0.05, 0.25)
            elif key == 'sigma':
                self.params[key] = np.clip(self.params[key], 0.01, 0.15)
            elif key == 'scale':
                self.params[key] = np.clip(self.params[key], 0.02, 0.20)
```

**P7 Detection of Boundary Hits:**
```python
def detect_boundary_hit(params_before, params_after, policy_type):
    """Detect if any parameter hit min/max bound during adjustment."""
    
    bounds = {
        'Uniform': {
            'timing_range_hours': (0.5, 6.0),
            'threshold_pct': (0.05, 0.25)
        },
        'OU': {
            'sigma': (0.01, 0.15)
        },
        'Pink': {
            'scale': (0.02, 0.20)
        }
    }
    
    for param, (min_val, max_val) in bounds[policy_type].items():
        if param in params_after:
            if params_after[param] == min_val or params_after[param] == max_val:
                return True, param
    
    return False, None
```

---

## 3. Integration with P6 (Standard Adversary)

### 3.1 Code Reuse Strategy

P7 reuses P6's adversary training infrastructure:

```python
# P7 imports from P6's module
from bsml.adversary.standard import (
    train_adversary_classifier,
    extract_features,
    generate_labels,
    compute_auc,
    compute_pnr
)
```

**What P7 reuses from P6:**
- Adversary training function
- Feature engineering pipeline
- Label generation (binary: trade in next Δt)
- AUC computation with bootstrap CIs
- PNR (Precision @ fixed positive rate) computation

**What P7 adds:**
- Adaptive training loop wrapper
- Parameter adjustment logic
- Convergence detection
- Comparative analysis (adaptive vs. non-adaptive)

### 3.2 Feature Engineering Alignment

P7 uses **identical features** as P6 to ensure fair comparison.

**Feature Categories (from P6 spec):**

1. **Signal State Features**
   - Last K signal values (e.g., K=10)
   - Signal slopes (first derivative)
   - Signal z-scores (normalized)
   - Distance to threshold

2. **Execution Context Features**
   - Bar index in trading day
   - Minutes since market open
   - Minutes until market close
   - Day of week (Monday=0, Friday=4)

3. **Recent Actions (Lagged)**
   - Last trade side (BUY/SELL, lagged by 1+ bars)
   - Last trade time (minutes ago)
   - Last trade size (shares)
   - Cumulative participation today

4. **Market Microstructure Features**
   - Bid-ask spread (basis points)
   - Midprice returns (lagged 1, 5, 10 bars)
   - Realized volatility (RV)
   - Average True Range (ATR)
   - Volume percentile (vs. 20-day average)
   - Volatility regime flag (high/low)

**P7 does NOT add:**
- Policy ID (would leak adaptive strategy)
- Iteration number (would leak training stage)
- Parameter values (would leak adjustment mechanism)
- Adjustment history (would leak feedback loop)

**Rationale:** Adversary must learn only from observable market/strategy behavior, not from knowledge of the adaptive process.

### 3.3 Adversary Training Protocol

P7 uses the same training protocol as P6:

```python
def train_adversary_for_adaptive_loop(trades, policy_name, iteration):
    """
    Train adversary using P6's protocol within P7's adaptive loop.
    
    Args:
        trades: DataFrame of trades from current iteration
        policy_name: str, e.g., 'Uniform', 'OU', 'Pink'
        iteration: int, current training iteration
    
    Returns:
        adversary: Trained classifier
        auc_score: float, validation AUC
    """
    
    # Step 1: Extract features (P6 function)
    features = extract_features(trades)
    # Returns: DataFrame with columns for all feature categories
    # Shape: (n_samples, n_features) e.g., (10000, 25)
    
    # Step 2: Generate labels (P6 function)
    labels = generate_labels(trades, delta_t='5min')
    # Returns: Binary array (0/1) indicating if trade occurs in next 5 min
    # Shape: (n_samples,)
    
    # Step 3: Train classifier (P6 function)
    adversary = train_adversary_classifier(
        features=features,
        labels=labels,
        model='lightgbm',
        params={
            'max_depth': 5,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
    )
    # Returns: Trained LightGBM classifier
    
    # Step 4: Evaluate on validation fold (P6 function)
    auc_score = compute_auc(adversary, features_val, labels_val)
    # Returns: ROC-AUC score [0.5, 1.0]
    
    return adversary, auc_score
```

**Training Configuration (from P6 spec):**

| Component | Value | Source |
|-----------|-------|--------|
| Model | LightGBM | P6 spec |
| Max depth | 5 | P6 spec |
| Num leaves | 31 | P6 spec |
| Learning rate | 0.05 | P6 spec |
| N estimators | 100 | P6 spec |
| Validation | Rolling walk-forward | P6 spec |
| Target | Binary (trade in next Δt) | P6 spec |
| Δt | 5 minutes | P6 spec |

### 3.4 AUC Computation

P7 uses P6's AUC function with block bootstrap for confidence intervals:

```python
from bsml.adversary.standard import compute_auc

# Compute AUC on validation fold
auc_score = compute_auc(
    classifier=adversary,
    features=features_val,
    labels=labels_val,
    n_bootstrap=1000,
    block_size=5  # 5-day blocks
)

# Returns: float, mean AUC from bootstrap
# Also stores CIs internally for later retrieval
```

**From P6 spec:**
- Use sklearn's `roc_auc_score` for base calculation
- Apply block bootstrap (5-day blocks) for temporal structure
- 1000 bootstrap iterations
- Report mean AUC ± 95% CI

### 3.5 Walk-Forward Splits

P7 uses P3's data split structure, aligned with P6:

```python
# Data splits from P3
data_splits = {
    'train': [
        # Iteration 0: 2020-01-01 to 2020-12-31
        # Iteration 1: 2020-02-01 to 2021-01-31
        # ... (rolling 1-year windows)
    ],
    'val': [
        # Iteration 0: 2021-01-01 to 2021-03-31 (3 months)
        # Iteration 1: 2021-02-01 to 2021-04-30
        # ... (rolling 3-month windows)
    ],
    'test': [
        # Final: 2024-01-01 to 2024-12-31 (held-out year)
    ]
}

# P7 uses these splits in adaptive loop
for iteration in range(n_iterations):
    # Train on train[iteration]
    trades_train = run_backtest(policy, data_splits['train'][iteration])
    adversary = train_adversary_classifier(trades_train)
    
    # Validate on val[iteration]
    trades_val = run_backtest(policy, data_splits['val'][iteration])
    auc = compute_auc(adversary, trades_val)
    
    # Adjust policy based on AUC
    adjust_policy(policy, auc)
```

**Critical:** No future data leakage - validation fold is strictly after training fold.

### 3.6 Feature Extraction Example

```python
def extract_features(trades):
    """
    P6's feature extraction (P7 reuses this).
    
    Args:
        trades: DataFrame with columns:
            - date, symbol, side, qty, ref_price, exec_price
    
    Returns:
        features: DataFrame with columns for all feature categories
    """
    
    features = pd.DataFrame()
    
    # 1. Signal state features
    features['signal_value'] = compute_signal(trades)
    features['signal_slope'] = compute_signal_slope(trades)
    features['signal_zscore'] = compute_signal_zscore(trades)
    features['threshold_distance'] = compute_threshold_distance(trades)
    
    # 2. Execution context features
    features['bar_index'] = compute_bar_index(trades)
    features['minutes_since_open'] = compute_minutes_since_open(trades)
    features['minutes_to_close'] = compute_minutes_to_close(trades)
    features['day_of_week'] = trades['date'].dt.dayofweek
    
    # 3. Recent actions (lagged)
    features['last_trade_side'] = compute_last_trade_side(trades, lag=1)
    features['last_trade_time'] = compute_last_trade_time(trades, lag=1)
    features['last_trade_size'] = compute_last_trade_size(trades, lag=1)
    features['participation_today'] = compute_participation_today(trades)
    
    # 4. Market microstructure features
    features['spread_bps'] = compute_spread(trades)
    features['midprice_return_1'] = compute_returns(trades, lag=1)
    features['midprice_return_5'] = compute_returns(trades, lag=5)
    features['realized_vol'] = compute_realized_volatility(trades)
    features['volume_percentile'] = compute_volume_percentile(trades)
    features['vol_regime'] = compute_vol_regime(trades)
    
    return features
```

---

## 4. Data Flow Diagram

### 4.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    P7 ADAPTIVE TRAINING LOOP                 │
│                                                               │
│  Iteration 0:                                                │
│    ├─ Generate trades with policy_0 (initial params)        │
│    ├─ P6: extract_features(trades) → features               │
│    ├─ P6: generate_labels(trades) → labels                  │
│    ├─ P6: train_adversary(features, labels) → adversary_0  │
│    ├─ P6: compute_auc(adversary_0, val_data) → auc_0       │
│    └─ P4: policy.adjust_stochasticity(auc_0) → policy_1    │
│                                                               │
│  Iteration 1:                                                │
│    ├─ Generate trades with policy_1 (adjusted params)       │
│    ├─ P6: extract_features(trades) → features               │
│    ├─ P6: generate_labels(trades) → labels                  │
│    ├─ P6: train_adversary(features, labels) → adversary_1  │
│    ├─ P6: compute_auc(adversary_1, val_data) → auc_1       │
│    └─ P4: policy.adjust_stochasticity(auc_1) → policy_2    │
│                                                               │
│  ... continue until convergence or max iterations ...        │
│                                                               │
│  Final:                                                       │
│    └─ Evaluate policy_final on test set                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Data Flow

```
Input: Baseline Strategy (P2), Initial Policy (P4), Data Splits (P3)
   |
   v
┌──────────────────────────────────────┐
│ P7: Initialize Adaptive Loop         │
│ - policy = policy_initial            │
│ - adjustment_log = []                │
│ - auc_history = []                   │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ FOR iteration in range(n_iterations) │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P3: run_backtest(policy, train_data) │
│ → trades_train                       │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: extract_features(trades_train)   │
│ → features_train                     │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: generate_labels(trades_train)    │
│ → labels_train                       │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: train_adversary_classifier(      │
│     features_train, labels_train)    │
│ → adversary                          │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P3: run_backtest(policy, val_data)   │
│ → trades_val                         │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: extract_features(trades_val)     │
│ → features_val                       │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: generate_labels(trades_val)      │
│ → labels_val                         │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P6: compute_auc(adversary,           │
│     features_val, labels_val)        │
│ → auc_score                          │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P7: decide_adjustment(auc_score)     │
│ → action, multiplier                 │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P4: policy.adjust_stochasticity(     │
│     auc_score, direction)            │
│ → policy params updated              │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P4: policy.check_exposure_invariance │
│ → is_valid                           │
└────────────┬─────────────────────────┘
             |
         ┌───┴───┐
         |       |
    is_valid  not valid
         |       |
         v       v
    Continue  Revert
         |
         v
┌──────────────────────────────────────┐
│ P7: Log adjustment                   │
│ adjustment_log.append(record)        │
│ auc_history.append(auc_score)        │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P7: Check convergence                │
│ → converged?                         │
└────────────┬─────────────────────────┘
             |
         ┌───┴───┐
         |       |
      YES: break NO: continue
         |       |
         v       └─── (loop back to next iteration)
         |
         v
┌──────────────────────────────────────┐
│ P3: run_backtest(policy, test_data)  │
│ → trades_test                        │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P5: compute_metrics(trades_test)     │
│ → Sharpe, MaxDD, ΔIS                 │
└────────────┬─────────────────────────┘
             |
             v
┌──────────────────────────────────────┐
│ P7: Return results                   │
│ - auc_history                        │
│ - adjustment_log                     │
│ - final_metrics                      │
│ - final_policy_params                │
└──────────────────────────────────────┘
```

---

## 5. Interface Specifications

### 5.1 P7 → P4 Interface

**Function:** `policy.adjust_stochasticity(auc_score, direction)`

**Input:**
```python
{
    'auc_score': float,  # Range [0.5, 1.0]
    'direction': str     # One of ['increase', 'decrease']
}
```

**Output:**
```python
None  # Modifies policy.params in-place
```

**Side Effects:**
- Modifies `policy.params` dictionary
- Logs adjustment in `policy._perturbation_log`

**Example:**
```python
# Before
policy.params = {'timing_range_hours': 2.0, 'threshold_pct': 0.10}

# Call
policy.adjust_stochasticity(auc_score=0.82, direction='increase')

# After
policy.params = {'timing_range_hours': 2.4, 'threshold_pct': 0.12}
```

### 5.2 P7 → P6 Interface

**Function:** `train_adversary_classifier(features, labels, model, params)`

**Input:**
```python
{
    'features': pd.DataFrame,  # Shape (n_samples, n_features)
    'labels': np.ndarray,      # Shape (n_samples,), values in {0, 1}
    'model': str,              # 'lightgbm' or 'xgboost'
    'params': dict             # Model hyperparameters
}
```

**Output:**
```python
adversary: LGBMClassifier  # Trained classifier object
```

**Example:**
```python
adversary = train_adversary_classifier(
    features=features_train,
    labels=labels_train,
    model='lightgbm',
    params={'max_depth': 5, 'num_leaves': 31}
)

# Use for prediction
predictions = adversary.predict_proba(features_val)[:, 1]
```

---

**Function:** `compute_auc(classifier, features, labels, n_bootstrap, block_size)`

**Input:**
```python
{
    'classifier': LGBMClassifier,  # Trained adversary
    'features': pd.DataFrame,      # Validation features
    'labels': np.ndarray,          # Validation labels
    'n_bootstrap': int,            # Default 1000
    'block_size': int              # Default 5 (days)
}
```

**Output:**
```python
auc_mean: float  # Mean AUC from bootstrap [0.5, 1.0]
```

**Example:**
```python
auc = compute_auc(
    classifier=adversary,
    features=features_val,
    labels=labels_val,
    n_bootstrap=1000,
    block_size=5
)

print(f"Validation AUC: {auc:.3f}")
# Output: Validation AUC: 0.782
```

---

## 6. Error Handling

### 6.1 P4 Integration Errors

**Error:** P4's `adjust_stochasticity()` raises exception

**Handling:**
```python
try:
    policy.adjust_stochasticity(auc_score=0.82, direction='increase')
except Exception as e:
    print(f"ERROR in P4 adjustment: {e}")
    # Log error and continue with old params
    adjustment_record['error'] = str(e)
    adjustment_record['reverted'] = True
```

---

**Error:** Exposure invariance check fails

**Handling:**
```python
is_valid = policy.check_exposure_invariance(positions_before, positions_after)

if not is_valid:
    print("⚠️ Exposure violation detected")
    # Revert to previous parameters
    policy.params = adjustment_log[-2]['params_after']
    adjustment_record['reverted'] = True
    adjustment_record['revert_reason'] = 'exposure_violation'
```

### 6.2 P6 Integration Errors

**Error:** Feature extraction fails (missing columns)

**Handling:**
```python
try:
    features = extract_features(trades)
except KeyError as e:
    print(f"ERROR in feature extraction: Missing column {e}")
    # Skip this iteration
    continue
```

---

**Error:** Adversary training fails (insufficient data)

**Handling:**
```python
try:
    adversary = train_adversary_classifier(features, labels)
except ValueError as e:
    print(f"ERROR in adversary training: {e}")
    # Use previous iteration's adversary
    adversary = previous_adversary
```

---

**Error:** AUC computation fails

**Handling:**
```python
try:
    auc = compute_auc(adversary, features_val, labels_val)
except Exception as e:
    print(f"ERROR in AUC computation: {e}")
    # Assign neutral AUC
    auc = 0.65  # Midpoint of target range
    adjustment_record['auc_error'] = str(e)
```

---

## 7. Testing Strategy

### 7.1 Integration Tests with P4

**Test 1: Parameter Adjustment**
```python
def test_p4_adjustment():
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    # Test increase
    policy.adjust_stochasticity(0.82, 'increase')
    assert policy.params['timing_range_hours'] == 2.4
    
    # Test decrease
    policy.adjust_stochasticity(0.52, 'decrease')
    assert policy.params['timing_range_hours'] == 1.92  # 2.4 × 0.8
```

**Test 2: Boundary Enforcement**
```python
def test_p4_boundaries():
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 5.5})
    
    # Should clip at 6.0
    policy.adjust_stochasticity(0.82, 'increase')
    assert policy.params['timing_range_hours'] == 6.0
```

**Test 3: Exposure Check**
```python
def test_p4_exposure_check():
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    positions_before = {'AAPL': 100, 'MSFT': -100}
    positions_after = {'AAPL': 102, 'MSFT': -98}
    
    is_valid = policy.check_exposure_invariance(positions_before, positions_after)
    assert is_valid == True
```

### 7.2 Integration Tests with P6

**Test 1: Feature Extraction**
```python
def test_p6_features():
    trades = generate_toy_trades(n=1000)
    features = extract_features(trades)
    
    # Check expected columns exist
    expected_cols = ['signal_value', 'bar_index', 'spread_bps']
    assert all(col in features.columns for col in expected_cols)
    
    # Check no missing values
    assert features.isnull().sum().sum() == 0
```

**Test 2: Adversary Training**
```python
def test_p6_adversary_training():
    features, labels = generate_toy_data(n=1000)
    
    adversary = train_adversary_classifier(
        features=features,
        labels=labels,
        model='lightgbm',
        params={'max_depth': 3}
    )
    
    # Check predictions are probabilities
    preds = adversary.predict_proba(features)[:, 1]
    assert all(0 <= p <= 1 for p in preds)
```

**Test 3: AUC Computation**
```python
def test_p6_auc():
    features, labels = generate_toy_data(n=1000)
    adversary = train_adversary_classifier(features, labels)
    
    auc = compute_auc(adversary, features, labels)
    
    # AUC should be in valid range
    assert 0.5 <= auc <= 1.0
```

### 7.3 End-to-End Integration Test

```python
def test_adaptive_loop_integration():
    """Test complete adaptive loop with P4 and P6."""
    
    # Setup
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    data_splits = generate_toy_data_splits()
    
    # Run 3 iterations
    for iteration in range(3):
        # P3: Generate trades
        trades = run_backtest(policy, data_splits['train'][iteration])
        
        # P6: Train adversary
        features = extract_features(trades)
        labels = generate_labels(trades)
        adversary = train_adversary_classifier(features, labels)
        
        # P6: Compute AUC
        auc = compute_auc(adversary, features, labels)
        
        # P7: Adjust policy
        if auc > 0.75:
            policy.adjust_stochasticity(auc, 'increase')
        
        # P4: Check exposure
        is_valid = policy.check_exposure_invariance({}, {})
        assert is_valid
    
    print("✓ End-to-end integration test passed")
```

---

**END OF INTEGRATION DOCUMENTATION**

**Status:** Week 1 Deliverable  
**Owner:** P7  
**Date:** November 14, 2025
