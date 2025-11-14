# Adjustment Logic Documentation

**Project:** BSML - Randomized Execution Research  
**Owner:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Status:** Week 2 Deliverable

---

## 1. Overview

This document provides detailed explanation of the adaptive adjustment logic implemented in `adaptive_adversary_v0.1.py`. It explains how AUC scores are translated into parameter adjustments to balance predictability against expected returns.

---

## 2. Core Decision Logic

### 2.1 Decision Tree

The adaptive adversary uses a threshold-based decision tree to determine actions:

```
                    Input: AUC Score [0.5, 1.0]
                              |
                              v
                    ┌─────────────────────┐
                    │   AUC > 0.75?       │
                    │ (Too Predictable)   │
                    └──────┬──────────────┘
                           |
                    YES ───┤─── NO
                           |          |
                           v          v
                    ┌──────────┐  ┌─────────────────────┐
                    │ INCREASE │  │   AUC < 0.55?       │
                    │   ×1.2   │  │ (Too Random)        │
                    └──────────┘  └──────┬──────────────┘
                                         |
                                  YES ───┤─── NO
                                         |          |
                                         v          v
                                  ┌──────────┐  ┌──────────────────────┐
                                  │ DECREASE │  │ 0.60 ≤ AUC ≤ 0.70?   │
                                  │   ×0.8   │  │ (In Target Range)    │
                                  └──────────┘  └──────┬───────────────┘
                                                       |
                                                YES ───┤─── NO
                                                       |          |
                                                       v          v
                                                ┌──────────┐  ┌─────────────┐
                                                │   HOLD   │  │ AUC > 0.65? │
                                                │  (none)  │  │ (Midpoint)  │
                                                └──────────┘  └──────┬──────┘
                                                                     |
                                                              YES ───┤─── NO
                                                                     |          |
                                                                     v          v
                                                              ┌──────────┐  ┌──────────┐
                                                              │ NUDGE_UP │  │NUDGE_DOWN│
                                                              │   ×1.2   │  │   ×0.8   │
                                                              └──────────┘  └──────────┘
```

### 2.2 Threshold Values

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `AUC_HIGH_THRESHOLD` | **0.75** | Adversary predicts too well → strategy too predictable |
| `AUC_LOW_THRESHOLD` | **0.55** | Adversary barely predicts → strategy too random |
| `AUC_TARGET_MIN` | **0.60** | Lower bound of optimal range |
| `AUC_TARGET_MAX` | **0.70** | Upper bound of optimal range |
| `AUC_TARGET_MIDPOINT` | **0.65** | Center of target range (for nudging) |

### 2.3 Visual Representation

```
    Random Guessing          Optimal Zone         Exploitable
           ↓                      ↓                    ↓
    |------|======|===================|======|---------|
   0.50   0.55   0.60               0.70    0.75     1.00
           ↑                                  ↑
      DECREASE                           INCREASE
      trigger                            trigger
      
      |<--- Too Random --->|<-- Good -->|<-- Too Predictable -->|
```

---

## 3. Action Definitions

### 3.1 INCREASE

**Trigger:** `AUC > 0.75`

**Action:** Multiply all adjustable parameters by 1.2 (20% increase)

**Rationale:** 
- Adversary has strong predictive signal (AUC > 0.75)
- Strategy is too predictable and exploitable
- Need more randomness to reduce predictability

**Example:**
```python
# Before
timing_range_hours = 2.0
threshold_pct = 0.10

# After INCREASE
timing_range_hours = 2.4  # 2.0 × 1.2
threshold_pct = 0.12       # 0.10 × 1.2
```

**Financial Implication:**
- Trades become less predictable
- May increase implementation shortfall slightly
- Reduces risk of front-running or adverse selection

---

### 3.2 DECREASE

**Trigger:** `AUC < 0.55`

**Action:** Multiply all adjustable parameters by 0.8 (20% decrease)

**Rationale:**
- Adversary cannot predict (AUC barely above random 0.50)
- Strategy is too random, likely destroying signal quality
- Need less randomness to preserve returns

**Example:**
```python
# Before
timing_range_hours = 2.0
threshold_pct = 0.10

# After DECREASE
timing_range_hours = 1.6  # 2.0 × 0.8
threshold_pct = 0.08       # 0.10 × 0.8
```

**Financial Implication:**
- Trades become more deterministic
- May improve Sharpe ratio by preserving signal
- Acceptable since adversary already cannot predict

---

### 3.3 HOLD

**Trigger:** `0.60 ≤ AUC ≤ 0.70`

**Action:** No change to parameters

**Rationale:**
- AUC in optimal "sweet spot"
- Good balance between unpredictability and performance
- Maintain current configuration

**Example:**
```python
# Before
timing_range_hours = 2.4
threshold_pct = 0.12

# After HOLD (no change)
timing_range_hours = 2.4
threshold_pct = 0.12
```

**Financial Implication:**
- Strategy is working well
- No trade-off adjustment needed
- Monitor for convergence

---

### 3.4 NUDGE_UP

**Trigger:** `0.65 < AUC < 0.75` (between target and high threshold)

**Action:** Multiply all adjustable parameters by 1.2 (same as INCREASE)

**Rationale:**
- AUC slightly above target midpoint (0.65)
- Not critically high, but trending toward predictable
- Gentle push toward target range

**Example:**
```python
# AUC = 0.72 (slightly high)
# Before
timing_range_hours = 2.2

# After NUDGE_UP
timing_range_hours = 2.64  # 2.2 × 1.2
```

**Difference from INCREASE:**
- Same multiplier, but triggered at lower urgency
- More of a fine-tuning adjustment
- Aims to prevent reaching high threshold

---

### 3.5 NUDGE_DOWN

**Trigger:** `0.55 < AUC < 0.65` (between low threshold and target)

**Action:** Multiply all adjustable parameters by 0.8 (same as DECREASE)

**Rationale:**
- AUC slightly below target midpoint (0.65)
- Not critically low, but trending toward too random
- Gentle push toward target range

**Example:**
```python
# AUC = 0.58 (slightly low)
# Before
timing_range_hours = 2.2

# After NUDGE_DOWN
timing_range_hours = 1.76  # 2.2 × 0.8
```

**Difference from DECREASE:**
- Same multiplier, but triggered at lower urgency
- More of a fine-tuning adjustment
- Aims to prevent reaching low threshold

---

## 4. Adjustment Factors

### 4.1 Multiplier Values

| Direction | Multiplier | Effect | Example |
|-----------|-----------|--------|---------|
| Increase stochasticity | **1.20** | +20% | 2.0 → 2.4 |
| Decrease stochasticity | **0.80** | -20% | 2.0 → 1.6 |
| Hold | **None** | 0% | 2.0 → 2.0 |

### 4.2 Compounding Over Iterations

Adjustments are **multiplicative** and **compound** over multiple iterations:

```python
# Starting value
timing_range_hours = 2.0

# Iteration 1: INCREASE (AUC = 0.82)
timing_range_hours = 2.0 × 1.2 = 2.4

# Iteration 2: INCREASE (AUC = 0.78)
timing_range_hours = 2.4 × 1.2 = 2.88

# Iteration 3: HOLD (AUC = 0.65)
timing_range_hours = 2.88  # No change

# Iteration 4: DECREASE (AUC = 0.52)
timing_range_hours = 2.88 × 0.8 = 2.304
```

**Key Property:** Multiple increases followed by decreases do **not** return to original value due to compounding:

```python
# Starting: 2.0
# After INCREASE: 2.0 × 1.2 = 2.4
# After DECREASE: 2.4 × 0.8 = 1.92 ≠ 2.0
```

### 4.3 Geometric Progression

After `n` consecutive INCREASE actions:
```
final_value = initial_value × (1.2)^n
```

After `n` consecutive DECREASE actions:
```
final_value = initial_value × (0.8)^n
```

**Example - 5 consecutive increases:**
```
timing_range: 2.0 → 2.4 → 2.88 → 3.456 → 4.147 → 4.977
```

---

## 5. Parameter-Specific Application

### 5.1 Which Parameters Get Adjusted

Not all policy parameters are adjustable. The logic only modifies **stochasticity control parameters**:

#### Uniform Policy

| Parameter | Adjusted? | Multiplier Applied |
|-----------|-----------|-------------------|
| `timing_range_hours` | ✅ YES | ×1.2 or ×0.8 |
| `threshold_pct` | ✅ YES | ×1.2 or ×0.8 |

**Both parameters adjusted simultaneously**

#### OU Policy

| Parameter | Adjusted? | Reason |
|-----------|-----------|--------|
| `theta` (mean reversion) | ❌ NO | Changes process character |
| `sigma` (volatility) | ✅ YES | Controls noise magnitude |
| `mu` (long-term mean) | ❌ NO | Keep centered at zero |

**Only `sigma` adjusted**

#### Pink Noise Policy

| Parameter | Adjusted? | Reason |
|-----------|-----------|--------|
| `alpha` (spectral exponent) | ❌ NO | Defines 1/f noise color |
| `scale` (amplitude) | ✅ YES | Controls noise magnitude |

**Only `scale` adjusted**

### 5.2 Application Examples by Policy

**Uniform Policy:**
```python
# AUC = 0.82 → INCREASE
# Before
params = {'timing_range_hours': 2.0, 'threshold_pct': 0.10}

# After
params = {'timing_range_hours': 2.4, 'threshold_pct': 0.12}
# Both multiplied by 1.2
```

**OU Policy:**
```python
# AUC = 0.82 → INCREASE
# Before
params = {'theta': 0.15, 'sigma': 0.05, 'mu': 0.0}

# After
params = {'theta': 0.15, 'sigma': 0.06, 'mu': 0.0}
# Only sigma multiplied by 1.2, others unchanged
```

**Pink Noise Policy:**
```python
# AUC = 0.52 → DECREASE
# Before
params = {'alpha': 1.0, 'scale': 0.08}

# After
params = {'alpha': 1.0, 'scale': 0.064}
# Only scale multiplied by 0.8, alpha unchanged
```

---

## 6. Boundary Enforcement

### 6.1 Post-Adjustment Clipping

After applying multiplier, parameters are **clipped to valid bounds**:

```python
def apply_adjustment(value, multiplier, min_bound, max_bound):
    """Apply adjustment with boundary clipping."""
    new_value = value * multiplier
    clipped_value = max(min_bound, min(max_bound, new_value))
    return clipped_value
```

### 6.2 Boundary Values

| Policy | Parameter | Min | Max |
|--------|-----------|-----|-----|
| Uniform | `timing_range_hours` | 0.5 | 6.0 |
| Uniform | `threshold_pct` | 0.05 | 0.25 |
| OU | `sigma` | 0.01 | 0.15 |
| Pink | `scale` | 0.02 | 0.20 |

### 6.3 Boundary Hit Examples

**Example 1: Hit max boundary**
```python
# timing_range_hours = 5.5
# AUC = 0.82 → INCREASE
# Raw: 5.5 × 1.2 = 6.6
# Clipped: 6.0 (max boundary)
```

**Example 2: Hit min boundary**
```python
# timing_range_hours = 0.6
# AUC = 0.52 → DECREASE
# Raw: 0.6 × 0.8 = 0.48
# Clipped: 0.5 (min boundary)
```

**Example 3: Within bounds (no clipping)**
```python
# timing_range_hours = 3.0
# AUC = 0.82 → INCREASE
# Raw: 3.0 × 1.2 = 3.6
# Final: 3.6 (no clipping needed)
```

---

## 7. Rationale Generation

Each adjustment includes a human-readable rationale explaining the decision:

### 7.1 Rationale Templates

```python
# INCREASE
"AUC 0.820 > 0.75 (too predictable)"

# DECREASE
"AUC 0.520 < 0.55 (too random)"

# HOLD
"AUC 0.650 in target range [0.60, 0.70]"

# NUDGE_UP
"AUC 0.720 above target midpoint 0.65"

# NUDGE_DOWN
"AUC 0.580 below target midpoint 0.65"
```

### 7.2 Rationale Purpose

- **Debugging:** Understand why adjustment was made
- **Logging:** Track decision history
- **Reporting:** Explain adaptive behavior in paper
- **Transparency:** Make black-box adaptive process interpretable

---

## 8. Complete Adjustment Step Workflow

### 8.1 Step-by-Step Process

```
Input: auc_score, policy, iteration, adjustment_history
    |
    v
┌─────────────────────────────────────┐
│ 1. Decide Action                    │
│    action, multiplier =             │
│    decide_adjustment(auc_score)     │
└────────────┬────────────────────────┘
             |
             v
┌─────────────────────────────────────┐
│ 2. Store Parameters Before          │
│    params_before = policy.params    │
└────────────┬────────────────────────┘
             |
             v
┌─────────────────────────────────────┐
│ 3. Apply Adjustment (if not HOLD)   │
│    policy.adjust_stochasticity()    │
└────────────┬────────────────────────┘
             |
             v
┌─────────────────────────────────────┐
│ 4. Store Parameters After           │
│    params_after = policy.params     │
└────────────┬────────────────────────┘
             |
             v
┌─────────────────────────────────────┐
│ 5. Check Oscillation                │
│    oscillating = detect_oscillation │
└────────────┬────────────────────────┘
             |
             v
┌─────────────────────────────────────┐
│ 6. Create Adjustment Record         │
│    return {iteration, auc, action,  │
│            params_before, params_   │
│            after, oscillation, ...} │
└─────────────────────────────────────┘
    |
    v
Output: adjustment_record
```

### 8.2 Example Execution Trace

```python
# Initial state
policy.params = {'timing_range_hours': 2.0, 'threshold_pct': 0.10}
auc_score = 0.82

# Step 1: Decide
action = 'INCREASE'
multiplier = 1.20

# Step 2: Store before
params_before = {'timing_range_hours': 2.0, 'threshold_pct': 0.10}

# Step 3: Apply adjustment
policy.adjust_stochasticity(0.82, 'increase')
# Internally: params['timing_range_hours'] *= 1.2
#            params['threshold_pct'] *= 1.2

# Step 4: Store after
params_after = {'timing_range_hours': 2.4, 'threshold_pct': 0.12}

# Step 5: Check oscillation
oscillating = False  # (no prior adjustments)

# Step 6: Create record
record = {
    'iteration': 0,
    'auc': 0.82,
    'action': 'INCREASE',
    'multiplier': 1.20,
    'rationale': 'AUC 0.820 > 0.75 (too predictable)',
    'params_before': {'timing_range_hours': 2.0, 'threshold_pct': 0.10},
    'params_after': {'timing_range_hours': 2.4, 'threshold_pct': 0.12},
    'oscillation_warning': False
}
```

---

## 9. Edge Cases and Special Behaviors

### 9.1 Exact Threshold Values

**Question:** What happens at exact threshold values?

```python
# AUC = 0.75 (exactly at high threshold)
decide_adjustment(0.75)  # Returns: ('INCREASE', 1.20)

# AUC = 0.55 (exactly at low threshold)
decide_adjustment(0.55)  # Returns: ('DECREASE', 0.80)

# AUC = 0.60 (exactly at target lower bound)
decide_adjustment(0.60)  # Returns: ('HOLD', None)

# AUC = 0.70 (exactly at target upper bound)
decide_adjustment(0.70)  # Returns: ('HOLD', None)
```

**Rule:** Thresholds are **inclusive** on the action side:
- `AUC >= 0.75` → INCREASE
- `AUC <= 0.55` → DECREASE
- `0.60 <= AUC <= 0.70` → HOLD

### 9.2 Parameter Already at Boundary

**Question:** What happens if parameter is already at max and we INCREASE?

```python
# Parameter at max boundary
policy.params = {'timing_range_hours': 6.0}

# AUC = 0.85 → INCREASE
adaptive_step(0.85, policy, 0, [])

# After adjustment
# Raw: 6.0 × 1.2 = 7.2
# Clipped: 6.0 (stays at max)
```

**Behavior:**
- Adjustment still recorded as 'INCREASE'
- Parameter stays at boundary (clipped)
- Future adjustments will keep trying to increase (until AUC changes)

### 9.3 Very First Iteration

**Question:** How does oscillation detection work on first iteration?

```python
# First iteration: adjustment_history is empty
record = adaptive_step(0.82, policy, 0, adjustment_history=[])

# Oscillation check
oscillating = detect_oscillation([], window=3)
# Returns: False (insufficient history)

# Record includes
record['oscillation_warning'] = False
```

**Behavior:**
- Oscillation detection requires at least 3 adjustments
- First 2 iterations always return False for oscillation
- Starts detecting from iteration 3 onward

### 9.4 HOLD Action Details

**Question:** Does HOLD action create an adjustment record?

**Answer:** Yes, HOLD creates a record but with `multiplier=None` and `params_before == params_after`:

```python
record = adaptive_step(0.65, policy, 0, [])

# Record contents
{
    'action': 'HOLD',
    'multiplier': None,  # No multiplier applied
    'params_before': {'timing_range_hours': 2.0},
    'params_after': {'timing_range_hours': 2.0},  # Unchanged
    ...
}
```

---

## 10. Design Rationale

### 10.1 Why These Threshold Values?

**High Threshold = 0.75:**
- Market microstructure literature: AUC > 0.75 indicates exploitable predictability
- Conservative: triggers before reaching highly exploitable territory (>0.80)
- Gives margin for adjustment before serious exploitation risk

**Low Threshold = 0.55:**
- Just above random guessing (0.50)
- Adversary provides minimal signal below 0.55
- Avoids excessive randomness that destroys returns

**Target Range = [0.60, 0.70]:**
- Wide enough to avoid constant adjustments (10 percentage points)
- Narrow enough to maintain meaningful unpredictability
- Centered around 0.65 (good balance point)

### 10.2 Why 20% Adjustment Factor?

**Multiplier = 1.2 (increase) and 0.8 (decrease):**
- **Large enough:** 20% change has measurable effect on AUC
- **Small enough:** Avoids instability and constraint violations
- **Symmetric:** 1.2 × 0.8 ≈ 0.96 (slight decay, not exact reversal)
- **Tested by P4:** Validated to maintain exposure invariance

**Alternatives considered:**
- 10% (1.1/0.9): Too conservative, slow convergence
- 30% (1.3/0.7): Too aggressive, instability risk
- 20% (1.2/0.8): **Goldilocks** - just right

### 10.3 Why Separate NUDGE Actions?

**Purpose of NUDGE_UP and NUDGE_DOWN:**
- Provides **graduated response** to AUC deviations
- Distinguishes urgency: INCREASE (critical) vs NUDGE_UP (preventative)
- Improves interpretability in logs and paper

**Same Multiplier as INCREASE/DECREASE:**
- Simplicity: Only 2 multipliers to tune (1.2 and 0.8)
- v0.1 implementation uses same multiplier for NUDGE and full adjustment
- v1.0 may differentiate: NUDGE uses 1.1/0.9, full uses 1.2/0.8

---

## 11. Monitoring and Diagnostics

### 11.1 Key Metrics to Track

During adaptive training, monitor:

1. **AUC Trajectory:** Plot AUC over iterations
2. **Parameter Trajectory:** Plot each parameter over iterations
3. **Action Distribution:** Count of each action type
4. **Boundary Hits:** How often parameters hit min/max
5. **Oscillation Frequency:** Number of oscillation warnings
6. **Convergence Time:** Iterations until AUC stable in target

### 11.2 Healthy vs. Unhealthy Patterns

**Healthy Pattern:**
```
AUC: 0.85 → 0.78 → 0.72 → 0.68 → 0.65 → 0.67 → 0.63 (converged)
Actions: INC → INC → NUDGE → NUDGE → HOLD → HOLD → HOLD
```
- Steady decrease from high to target
- Transitions from aggressive (INC) to gentle (NUDGE) to stable (HOLD)
- Converges within ~7 iterations

**Unhealthy Pattern (Oscillation):**
```
AUC: 0.82 → 0.52 → 0.85 → 0.50 → 0.88 (no convergence)
Actions: INC → DEC → INC → DEC → INC
```
- Oscillates between high and low
- Alternating INC/DEC actions
- No progress toward target

**Unhealthy Pattern (Boundary Saturation):**
```
timing_range: 5.5 → 6.0 → 6.0 → 6.0 → 6.0 (stuck at max)
Actions: INC → INC → INC → INC → INC
AUC: 0.82 → 0.80 → 0.78 → 0.77 → 0.76 (still too high)
```
- Parameter hits boundary and cannot adjust further
- AUC not entering target despite max randomness
- May need higher max bound or different strategy

---

## 12. Implementation Notes

### 12.1 Code Location

- **Main function:** `decide_adjustment()` in `adaptive_adversary_v0.1.py`
- **Integration:** Called by `adaptive_step()`
- **Constants:** Defined at module level

### 12.2 Function Signature

```python
def decide_adjustment(auc_score: float) -> Tuple[str, Optional[float]]:
    """
    Returns:
        action: str in ['INCREASE', 'DECREASE', 'HOLD', 'NUDGE_UP', 'NUDGE_DOWN']
        multiplier: float (1.20 or 0.80) or None (for HOLD)
    """
```

### 12.3 Testing

All logic thoroughly tested in `tests/test_adaptive_loop.py`:
- ✅ All threshold triggers
- ✅ Edge cases (exact thresholds)
- ✅ Boundary enforcement
- ✅ Compounding over iterations
- ✅ Parameter-specific application

---

**END OF ADJUSTMENT LOGIC DOCUMENTATION**

**Status:** Week 2 Deliverable  
**Owner:** P7  
**Date:** November 14, 2025  
**Next:** Use this logic in Week 3 v1.0 with real P6 adversary
