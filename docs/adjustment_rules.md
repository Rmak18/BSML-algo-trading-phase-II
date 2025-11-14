# Adjustment Rules Documentation

**Project:** BSML - Randomized Execution Research  
**Owner:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Status:** Week 1 Deliverable

---

## 1. Overview

This document specifies the exact logic for adjusting randomization policy parameters based on adversary AUC scores. The adaptive adversary uses threshold-based rules to increase or decrease stochasticity dynamically during training.

---

## 2. AUC Thresholds

### 2.1 Proposed Threshold Values

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `AUC_HIGH_THRESHOLD` | **0.75** | Strategy is too predictable |
| `AUC_LOW_THRESHOLD` | **0.55** | Strategy is too random |
| `AUC_TARGET_MIN` | **0.60** | Lower bound of optimal range |
| `AUC_TARGET_MAX` | **0.70** | Upper bound of optimal range |

### 2.2 Rationale

**Why 0.75 for high threshold?**
- AUC > 0.75 indicates adversary has strong predictive signal
- In market microstructure, AUC 0.70-0.75 suggests exploitable patterns
- Provides margin before reaching highly exploitable territory (AUC > 0.80)

**Why 0.55 for low threshold?**
- Random classifier has AUC = 0.50
- AUC < 0.55 means barely better than random guessing
- Excessive randomness likely degrades returns without predictability benefit

**Why (0.60, 0.70) as target range?**
- Middle ground: unpredictable enough to resist adversaries
- Not so random as to destroy signal quality
- Wide enough range (10 percentage points) to avoid constant oscillations

### 2.3 Visual Representation

```
        Too Random          Optimal Range        Too Predictable
            ↓                    ↓                      ↓
    |-------|========|====================|========|-------|
   0.50    0.55     0.60                 0.70     0.75    1.00
            ↑                                       ↑
      DECREASE trigger                        INCREASE trigger
```

---

## 3. Adjustment Factor

### 3.1 Proposed Values

| Action | Multiplier | Effect |
|--------|-----------|--------|
| **Increase stochasticity** | **1.20** | Multiply adjustable params by 1.2 (20% increase) |
| **Decrease stochasticity** | **0.80** | Multiply adjustable params by 0.8 (20% decrease) |

### 3.2 Examples

**Uniform Policy - Increase:**
```python
# Before adjustment
timing_range_hours = 2.0
threshold_pct = 0.10

# After INCREASE (×1.2)
timing_range_hours = 2.4  # 2.0 × 1.2
threshold_pct = 0.12       # 0.10 × 1.2
```

**OU Policy - Decrease:**
```python
# Before adjustment
sigma = 0.06

# After DECREASE (×0.8)
sigma = 0.048  # 0.06 × 0.8
```

### 3.3 Rationale

**Why 20% adjustment?**
- Large enough to have measurable effect on AUC
- Small enough to avoid instability and constraint violations
- P4 API spec uses 1.2/0.8 as example values
- Can be tuned down to 1.15/0.85 (10%) if oscillations occur

---

## 4. Decision Logic

### 4.1 Flowchart

```
                        Start: Receive AUC Score
                                   |
                                   v
                        ┌──────────────────────┐
                        │ AUC > 0.75?          │
                        │ (Too Predictable)    │
                        └──────────┬───────────┘
                                   |
                           YES ────┤──── NO
                                   |           |
                                   v           v
                        ┌──────────────┐   ┌──────────────────────┐
                        │  INCREASE    │   │ AUC < 0.55?          │
                        │ stochasticity│   │ (Too Random)         │
                        │  (×1.2)      │   └──────────┬───────────┘
                        └──────────────┘              |
                                              YES ────┤──── NO
                                                      |           |
                                                      v           v
                                           ┌──────────────┐   ┌──────────────────────┐
                                           │  DECREASE    │   │ 0.60 ≤ AUC ≤ 0.70?   │
                                           │ stochasticity│   │ (In Target Range)    │
                                           │  (×0.8)      │   └──────────┬───────────┘
                                           └──────────────┘              |
                                                                  YES ────┤──── NO
                                                                          |           |
                                                                          v           v
                                                                   ┌──────────┐   ┌──────────────┐
                                                                   │   HOLD   │   │ AUC > 0.65?  │
                                                                   │  params  │   │ (Midpoint)   │
                                                                   └──────────┘   └──────┬───────┘
                                                                                          |
                                                                                  YES ────┤──── NO
                                                                                          |           |
                                                                                          v           v
                                                                                   ┌──────────┐  ┌──────────┐
                                                                                   │ NUDGE_UP │  │NUDGE_DOWN│
                                                                                   │  (×1.2)  │  │  (×0.8)  │
                                                                                   └──────────┘  └──────────┘
```

### 4.2 Pseudocode

```python
def decide_adjustment(auc_score):
    """
    Decide what adjustment action to take based on AUC score.
    
    Args:
        auc_score: Adversary AUC on validation fold [0.5, 1.0]
    
    Returns:
        action: str, one of ['INCREASE', 'DECREASE', 'HOLD', 'NUDGE_UP', 'NUDGE_DOWN']
        multiplier: float or None
    """
    
    # Thresholds
    AUC_HIGH = 0.75
    AUC_LOW = 0.55
    AUC_TARGET_MIN = 0.60
    AUC_TARGET_MAX = 0.70
    TARGET_MIDPOINT = 0.65
    
    # Primary rules
    if auc_score > AUC_HIGH:
        return 'INCREASE', 1.20
    
    elif auc_score < AUC_LOW:
        return 'DECREASE', 0.80
    
    elif AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
        return 'HOLD', None
    
    # Secondary rules (between thresholds)
    else:
        if auc_score > TARGET_MIDPOINT:
            return 'NUDGE_UP', 1.20
        else:
            return 'NUDGE_DOWN', 0.80
```

### 4.3 Action Descriptions

| Action | Condition | Multiplier | Description |
|--------|-----------|-----------|-------------|
| **INCREASE** | AUC > 0.75 | 1.20 | Strong predictability signal → add more randomness |
| **DECREASE** | AUC < 0.55 | 0.80 | Too random, barely predictive → reduce randomness |
| **HOLD** | 0.60 ≤ AUC ≤ 0.70 | None | In optimal range → maintain current params |
| **NUDGE_UP** | 0.65 < AUC < 0.75 | 1.20 | Slightly high → gentle increase toward target |
| **NUDGE_DOWN** | 0.55 < AUC < 0.65 | 0.80 | Slightly low → gentle decrease toward target |

---

## 5. Parameter Application

### 5.1 Which Parameters Are Adjusted

Not all policy parameters are adjustable. Only stochastic parameters that control randomness magnitude:

#### Uniform Policy

| Parameter | Adjustable? | Rationale |
|-----------|-------------|-----------|
| `timing_range_hours` | ✅ YES | Controls timing perturbation magnitude |
| `threshold_pct` | ✅ YES | Controls threshold perturbation magnitude |

#### OU Policy

| Parameter | Adjustable? | Rationale |
|-----------|-------------|-----------|
| `sigma` (volatility) | ✅ YES | Controls noise magnitude |
| `theta` (mean reversion) | ❌ NO | Changing alters process character |
| `mu` (long-term mean) | ❌ NO | Keep centered at zero |

#### Pink Noise Policy

| Parameter | Adjustable? | Rationale |
|-----------|-------------|-----------|
| `scale` (amplitude) | ✅ YES | Controls noise magnitude |
| `alpha` (spectral exponent) | ❌ NO | Changing alters noise color (pink → white/brown) |

### 5.2 Application Formula

For each adjustable parameter `p`:

```python
if action in ['INCREASE', 'NUDGE_UP']:
    p_new = p_old × 1.20
    
elif action in ['DECREASE', 'NUDGE_DOWN']:
    p_new = p_old × 0.80
    
elif action == 'HOLD':
    p_new = p_old  # No change
```

### 5.3 Boundary Enforcement

After adjustment, parameters are clipped to valid ranges:

| Policy | Parameter | Min | Max |
|--------|-----------|-----|-----|
| Uniform | `timing_range_hours` | 0.5 | 6.0 |
| Uniform | `threshold_pct` | 0.05 | 0.25 |
| OU | `sigma` | 0.01 | 0.15 |
| Pink | `scale` | 0.02 | 0.20 |

```python
def apply_adjustment(param_value, action, min_bound, max_bound):
    """Apply adjustment with boundary clipping."""
    
    if action in ['INCREASE', 'NUDGE_UP']:
        new_value = param_value * 1.20
    elif action in ['DECREASE', 'NUDGE_DOWN']:
        new_value = param_value * 0.80
    else:  # HOLD
        new_value = param_value
    
    # Clip to bounds
    new_value = max(min_bound, min(max_bound, new_value))
    
    return new_value
```

---

## 6. Adjustment History Tracking

### 6.1 What to Log

For each adjustment decision, record:

```python
adjustment_record = {
    'iteration': int,              # Training iteration number
    'auc': float,                  # Validation AUC score
    'action': str,                 # 'INCREASE', 'DECREASE', 'HOLD', etc.
    'multiplier': float or None,   # 1.20, 0.80, or None
    'rationale': str,              # Human-readable explanation
    'params_before': dict,         # Parameter values before adjustment
    'params_after': dict,          # Parameter values after adjustment
    'bounded': bool,               # True if any param hit min/max bound
    'reverted': bool               # True if adjustment was reverted (exposure violation)
}
```

### 6.2 Example Log Entry

```python
{
    'iteration': 5,
    'auc': 0.782,
    'action': 'INCREASE',
    'multiplier': 1.20,
    'rationale': 'AUC 0.782 > 0.75 (too predictable)',
    'params_before': {'timing_range_hours': 2.0, 'threshold_pct': 0.10},
    'params_after': {'timing_range_hours': 2.4, 'threshold_pct': 0.12},
    'bounded': False,
    'reverted': False
}
```

---

## 7. Convergence Detection

### 7.1 Convergence Criterion

Training is considered converged when:

**AUC stays within target range [0.60, 0.70] for 5 consecutive iterations**

### 7.2 Implementation

```python
def check_convergence(auc_history, patience=5):
    """
    Check if AUC has converged (stable in target range).
    
    Args:
        auc_history: List of AUC scores from all iterations
        patience: Number of consecutive iterations in target range required
    
    Returns:
        converged: bool
        convergence_iteration: int or None
    """
    
    AUC_TARGET_MIN = 0.60
    AUC_TARGET_MAX = 0.70
    
    if len(auc_history) < patience:
        return False, None
    
    # Check last N iterations
    last_n = auc_history[-patience:]
    all_in_range = all(AUC_TARGET_MIN <= auc <= AUC_TARGET_MAX for auc in last_n)
    
    if all_in_range:
        convergence_iteration = len(auc_history) - patience + 1
        return True, convergence_iteration
    
    return False, None
```

---

## 8. Oscillation Detection

### 8.1 Oscillation Definition

Oscillation occurs when adjustments alternate direction without making progress:

**Pattern:** INCREASE → DECREASE → INCREASE (or vice versa)

### 8.2 Detection Logic

```python
def detect_oscillation(adjustment_history, window=3):
    """
    Detect if adjustments are oscillating.
    
    Args:
        adjustment_history: List of adjustment records
        window: Number of recent adjustments to check
    
    Returns:
        is_oscillating: bool
    """
    
    if len(adjustment_history) < window:
        return False
    
    # Get last N actions
    last_actions = [rec['action'] for rec in adjustment_history[-window:]]
    
    # Check for alternating patterns
    oscillation_patterns = [
        ['INCREASE', 'DECREASE', 'INCREASE'],
        ['DECREASE', 'INCREASE', 'DECREASE'],
        ['NUDGE_UP', 'NUDGE_DOWN', 'NUDGE_UP'],
        ['NUDGE_DOWN', 'NUDGE_UP', 'NUDGE_DOWN']
    ]
    
    for pattern in oscillation_patterns:
        if last_actions == pattern:
            return True
    
    return False
```

### 8.3 Oscillation Mitigation

If oscillation detected:

1. **Warning:** Print warning message to console and log
2. **Smoothing:** Switch to exponential moving average (EMA) of AUC
3. **Reduce Factor:** Change adjustment from 1.2/0.8 to 1.1/0.9 (gentler)
4. **Patience:** Require 2 consecutive iterations outside target before adjusting

```python
def compute_ema_auc(auc_history, alpha=0.3):
    """Exponential moving average of AUC scores."""
    if len(auc_history) == 0:
        return None
    
    ema = auc_history[0]
    for auc in auc_history[1:]:
        ema = alpha * auc + (1 - alpha) * ema
    
    return ema
```

---

## 9. Exposure Invariance Check

### 9.1 Constraint

After each parameter adjustment, verify:

**|net_exposure_after - net_exposure_before| ≤ 5% of NAV**

### 9.2 Implementation

```python
def verify_exposure_invariance(trades_before, trades_after, tolerance=0.05):
    """
    Check that adjusted policy maintains exposure constraint.
    
    Args:
        trades_before: Trades generated before adjustment
        trades_after: Trades generated after adjustment
        tolerance: Maximum allowed drift (default 5%)
    
    Returns:
        is_valid: bool
        net_before: float
        net_after: float
    """
    
    # Compute net positions
    net_before = compute_net_exposure(trades_before)
    net_after = compute_net_exposure(trades_after)
    
    # Check drift
    drift = abs(net_after - net_before)
    is_valid = drift <= tolerance
    
    return is_valid, net_before, net_after


def compute_net_exposure(trades):
    """Compute net exposure from trades."""
    net = 0
    for trade in trades:
        if trade['side'] == 'BUY':
            net += trade['qty']
        elif trade['side'] == 'SELL':
            net -= trade['qty']
    return net
```

### 9.3 Violation Handling

If exposure constraint violated:

1. **Revert:** Restore previous parameter values
2. **Log:** Mark adjustment record with `reverted=True`
3. **Continue:** Move to next iteration without adjusting
4. **Halt:** If 3 consecutive violations, stop training for manual review

```python
def handle_exposure_violation(policy, adjustment_log):
    """Revert parameter adjustment due to exposure violation."""
    
    if len(adjustment_log) < 2:
        print("ERROR: Cannot revert, no previous parameters")
        return False
    
    # Restore previous parameters
    previous_params = adjustment_log[-2]['params_after']
    policy.params = previous_params.copy()
    
    # Mark current adjustment as reverted
    adjustment_log[-1]['reverted'] = True
    
    print(f"Reverted to params: {policy.params}")
    return True
```

---

## 10. Complete Adjustment Algorithm

### 10.1 Main Function

```python
def adaptive_adjustment_step(
    auc_score,
    policy,
    iteration,
    adjustment_history,
    trades_current
):
    """
    Execute one complete adaptive adjustment step.
    
    Args:
        auc_score: Current validation AUC
        policy: RandomizationPolicy instance
        iteration: Current iteration number
        adjustment_history: List of prior adjustments
        trades_current: Current trades for exposure check
    
    Returns:
        adjustment_record: Dict with adjustment details
        success: bool (False if reverted)
    """
    
    # Step 1: Decide action based on AUC
    action, multiplier = decide_adjustment(auc_score)
    
    # Step 2: Store old parameters
    params_before = policy.params.copy()
    
    # Step 3: Apply adjustment (if not HOLD)
    if action != 'HOLD':
        policy.adjust_stochasticity(auc_score, direction='increase' if 'UP' in action else 'decrease')
    
    params_after = policy.params.copy()
    
    # Step 4: Check for oscillation
    oscillating = detect_oscillation(adjustment_history)
    
    # Step 5: Create record
    record = {
        'iteration': iteration,
        'auc': auc_score,
        'action': action,
        'multiplier': multiplier,
        'rationale': f"AUC {auc_score:.3f} → {action}",
        'params_before': params_before,
        'params_after': params_after,
        'oscillation_warning': oscillating,
        'reverted': False
    }
    
    # Step 6: Exposure check (if params changed)
    if action != 'HOLD':
        # Generate small test batch with new params
        trades_test = generate_test_batch(policy)
        
        is_valid, net_before, net_after = verify_exposure_invariance(
            trades_current, trades_test
        )
        
        if not is_valid:
            print(f"⚠️ EXPOSURE VIOLATION: {net_before:.2f} → {net_after:.2f}")
            handle_exposure_violation(policy, adjustment_history + [record])
            record['reverted'] = True
            return record, False
    
    return record, True
```

---

## 11. Usage Example

```python
# Initialize
policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0, 'threshold_pct': 0.10})
adjustment_history = []

# Training loop
for iteration in range(20):
    # Train adversary and get AUC
    auc = train_and_evaluate_adversary(policy, data)
    
    # Execute adaptive adjustment
    record, success = adaptive_adjustment_step(
        auc_score=auc,
        policy=policy,
        iteration=iteration,
        adjustment_history=adjustment_history,
        trades_current=current_trades
    )
    
    adjustment_history.append(record)
    
    print(f"Iteration {iteration}: AUC={auc:.3f}, Action={record['action']}")
    
    # Check convergence
    converged, conv_iter = check_convergence([r['auc'] for r in adjustment_history])
    if converged:
        print(f"✓ Converged at iteration {conv_iter}")
        break
```

---

**END OF ADJUSTMENT RULES DOCUMENTATION**

**Status:** Week 1 Deliverable  
**Next:** Integration with P4/P6 in Week 2  
**Owner:** P7
