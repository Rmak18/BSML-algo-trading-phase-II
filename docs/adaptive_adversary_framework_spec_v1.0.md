# Adaptive Adversary Framework Specification v1.0

**Project:** BSML - Randomized Execution Research  
**Owner:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Status:** Week 1 Deliverable - FINALIZED  
**Version:** 1.0 (Locked for Week 2 Implementation)


---

## 1. Executive Summary

The **Adaptive Adversary Framework** is a closed-loop system that dynamically adjusts randomization policy parameters based on adversary predictability feedback during training. Unlike P6's static adversary approach, P7 continuously monitors how well an adversary classifier can predict trade timing and adjusts policy stochasticity to maintain an optimal balance between unpredictability and expected returns.

**Key Innovation:** Real-time AUC-based feedback loop that increases randomness when trades are too predictable (high AUC) and decreases randomness when overly random (low AUC, hurting returns).

**Research Question:** Does adaptive parameter tuning via adversary feedback reduce implementation shortfall more effectively than fixed randomization parameters, while maintaining comparable Sharpe ratios?

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   ADAPTIVE TRAINING LOOP (P7)                    │
│                                                                   │
│   ┌──────────┐       ┌─────────────┐       ┌──────────────┐    │
│   │ Baseline │  -->  │ Randomize   │  -->  │   Execute    │    │
│   │ Strategy │       │   Policy    │       │    Trades    │    │
│   │   (P2)   │       │    (P4)     │       │    (P3)      │    │
│   └──────────┘       └──────┬──────┘       └──────┬───────┘    │
│                             │                      │             │
│                             │ adjust               │ trade       │
│                             │ params               │ data        │
│                             │                      │             │
│                      ┌──────┴──────────────────────┴────────┐   │
│                      │   ADAPTIVE ADVERSARY CORE (P7)       │   │
│                      │                                       │   │
│                      │  1. Train Classifier (P6 protocol)   │   │
│                      │  2. Compute AUC on Validation        │   │
│                      │  3. Compare vs Thresholds            │   │
│                      │  4. Adjust Policy Params (P4 API)    │   │
│                      │  5. Check Exposure Invariance        │   │
│                      │  6. Log Adjustment History           │   │
│                      │  7. Repeat for N Iterations          │   │
│                      └──────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

**P7 (This Module) Owns:**
- Adaptive training loop orchestration
- AUC threshold evaluation logic
- Parameter adjustment decisions
- Convergence monitoring
- Comparative analysis (adaptive vs. non-adaptive)

**P7 Depends On:**
- P4: `adjust_stochasticity()` API for modifying policy parameters
- P6: Adversary training protocol and feature engineering
- P5: Metrics computation (Sharpe, MaxDD, ΔIS) for final evaluation
- P3: Backtesting harness and logging infrastructure

---

## 3. Adaptive Feedback Mechanism

### 3.1 Core Algorithm

The adaptive adversary operates on a simple feedback principle:

**IF** adversary predicts too well (AUC > threshold) **THEN** strategy is too predictable → increase randomness  
**ELSE IF** adversary cannot predict (AUC ≈ random) **THEN** strategy is too random → decrease randomness  
**ELSE** maintain current parameters (in optimal range)

### 3.2 AUC Thresholds (FINALIZED)

After team review, the following thresholds are **locked for Week 2 implementation**:

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `AUC_HIGH_THRESHOLD` | **0.75** | Above this = too predictable, adversary has strong signal | 
| `AUC_LOW_THRESHOLD` | **0.55** | Below this = too random, barely better than coin flip (0.50) | 
| `AUC_TARGET_RANGE` | **(0.60, 0.70)** | Optimal "sweet spot" balancing unpredictability and returns | 

**Rationale for 0.75 high threshold:**
- AUC > 0.75 indicates adversary has exploitable signal
- Market microstructure literature suggests AUC 0.70-0.75 is threshold for profitable front-running
- Conservative choice: trigger adjustment before reaching exploitable levels

**Rationale for 0.55 low threshold:**
- AUC < 0.55 indicates strategy is nearly random (0.50 = pure chance)
- Excessive randomness likely degrades returns without predictability benefit
- Allow some buffer above 0.50 to avoid overreaction

**Rationale for (0.60, 0.70) target range:**
- Middle ground: unpredictable enough to resist adversaries
- Not so random as to destroy signal quality
- Wide enough range to avoid constant oscillations

### 3.3 Adjustment Factor (FINALIZED)

After P4 coordination, the adjustment factor is **locked**:

| Action | Multiplier | Example |
|--------|-----------|---------|
| **Increase stochasticity** | **1.20** | `timing_range: 2.0 → 2.4 hours` |
| **Decrease stochasticity** | **0.80** | `timing_range: 2.0 → 1.6 hours` | 

**Rationale:**
- 20% adjustment is aggressive enough to have measurable effect
- Not so aggressive as to cause instability or violate constraints
- P4 tested on toy data: 1.2/0.8 provides good convergence properties
- Can tune down to 1.15/0.85 (10%) in Week 3 if oscillations observed

**Alternative considered:** 1.1/0.9 (10% change) - deemed too conservative after simulation

### 3.4 Decision Logic (Pseudocode)

```python
def adaptive_step(auc_score, policy, iteration, adjustment_history):
    """
    Execute one adaptive adjustment step.
    
    Args:
        auc_score: Current adversary AUC on validation fold [0.5, 1.0]
        policy: RandomizationPolicy instance from P4
        iteration: Current training iteration number
        adjustment_history: List of prior adjustments for oscillation detection
    
    Returns:
        adjustment_record: Dict with action, rationale, updated params
    """
    
    # Constants (locked from spec)
    AUC_HIGH = 0.75
    AUC_LOW = 0.55
    AUC_TARGET_MIN = 0.60
    AUC_TARGET_MAX = 0.70
    
    # Decision tree
    if auc_score > AUC_HIGH:
        # TOO PREDICTABLE: increase randomness
        policy.adjust_stochasticity(auc_score, direction='increase')
        action = 'INCREASE'
        rationale = f"AUC {auc_score:.3f} > {AUC_HIGH} (too predictable)"
        
    elif auc_score < AUC_LOW:
        # TOO RANDOM: decrease randomness
        policy.adjust_stochasticity(auc_score, direction='decrease')
        action = 'DECREASE'
        rationale = f"AUC {auc_score:.3f} < {AUC_LOW} (too random, hurting returns)"
        
    elif AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
        # OPTIMAL RANGE: hold current parameters
        action = 'HOLD'
        rationale = f"AUC {auc_score:.3f} in target range [{AUC_TARGET_MIN}, {AUC_TARGET_MAX}]"
        
    else:
        # BETWEEN THRESHOLDS: nudge toward target midpoint
        target_mid = (AUC_TARGET_MIN + AUC_TARGET_MAX) / 2  # 0.65
        if auc_score > target_mid:
            policy.adjust_stochasticity(auc_score, direction='increase')
            action = 'NUDGE_UP'
            rationale = f"AUC {auc_score:.3f} above target midpoint {target_mid:.2f}"
        else:
            policy.adjust_stochasticity(auc_score, direction='decrease')
            action = 'NUDGE_DOWN'
            rationale = f"AUC {auc_score:.3f} below target midpoint {target_mid:.2f}"
    
    # Record adjustment
    record = {
        'iteration': iteration,
        'auc': auc_score,
        'action': action,
        'rationale': rationale,
        'params_before': policy.params.copy() if action != 'HOLD' else None,
        'params_after': policy.params.copy()
    }
    
    # Oscillation detection (Week 2 enhancement)
    if len(adjustment_history) >= 3:
        last_3_actions = [h['action'] for h in adjustment_history[-3:]]
        if last_3_actions == ['INCREASE', 'DECREASE', 'INCREASE'] or \
           last_3_actions == ['DECREASE', 'INCREASE', 'DECREASE']:
            print(f"WARNING: Oscillation detected at iteration {iteration}")
            record['oscillation_warning'] = True
    
    return record
```

---

## 4. Integration with P4 Randomization Modules

### 4.1 Policy Parameter Mapping (FINALIZED)

Each policy has specific parameters that P7 adjusts via P4's API:

#### Uniform Policy

| Parameter | Type | Initial | Min | Max | Adjustable |
|-----------|------|---------|-----|-----|------------|
| `timing_range_hours` | float | 2.0 | 0.5 | 6.0 | ✅ YES |
| `threshold_pct` | float | 0.10 | 0.05 | 0.25 | ✅ YES |

**Adjustment Example:**
```python
# Initial state
policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0, 'threshold_pct': 0.10})

# After INCREASE (×1.2)
policy.adjust_stochasticity(auc_score=0.82, direction='increase')
# Result: timing_range_hours = 2.4, threshold_pct = 0.12
```

#### OU Policy

| Parameter | Type | Initial | Min | Max | Adjustable |
|-----------|------|---------|-----|-----|------------|
| `theta` (mean reversion) | float | 0.15 | 0.05 | 0.50 | ❌ NO (changes process character) |
| `sigma` (volatility) | float | 0.05 | 0.01 | 0.15 | ✅ YES |
| `mu` (long-term mean) | float | 0.0 | -0.1 | 0.1 | ❌ NO (keep centered) |

**Adjustment Example:**
```python
policy = OUPolicy(seed=42, params={'theta': 0.15, 'sigma': 0.05, 'mu': 0.0})

# After INCREASE (×1.2)
policy.adjust_stochasticity(auc_score=0.78, direction='increase')
# Result: sigma = 0.06 (only sigma adjusts, theta and mu fixed)
```

#### Pink Noise Policy

| Parameter | Type | Initial | Min | Max | Adjustable |
|-----------|------|---------|-----|-----|------------|
| `alpha` (spectral exponent) | float | 1.0 | 0.5 | 1.5 | ❌ NO (defines noise color) |
| `scale` (amplitude) | float | 0.08 | 0.02 | 0.20 | ✅ YES |

**Adjustment Example:**
```python
policy = PinkNoisePolicy(seed=42, params={'alpha': 1.0, 'scale': 0.08})

# After DECREASE (×0.8)
policy.adjust_stochasticity(auc_score=0.53, direction='decrease')
# Result: scale = 0.064 (only scale adjusts, alpha fixed at 1.0)
```

### 4.2 API Contract with P4 (LOCKED)

**P7 calls P4 method:**
```python
policy.adjust_stochasticity(auc_score: float, direction: str) -> None
```

**P4 responsibilities:**
- Modify `self.params` in-place by multiplying adjustable params by 1.2 (increase) or 0.8 (decrease)
- Enforce parameter bounds (clip to [min, max])
- Return None (modification happens in-place)

**P7 responsibilities after calling adjust_stochasticity():**
- Log old and new parameter values
- Verify exposure invariance via `policy.check_exposure_invariance()`
- Revert adjustment if constraint violated

### 4.3 Exposure Invariance Verification

After each adjustment, P7 must verify exposure constraints are maintained:

```python
def verify_adjustment_safety(policy, trades_before, trades_after):
    """
    Ensure adjusted policy maintains exposure invariance.
    
    Args:
        policy: Adjusted RandomizationPolicy instance
        trades_before: Trades generated before adjustment
        trades_after: Trades generated after adjustment
    
    Returns:
        bool: True if safe, False if constraint violated
    """
    
    # Extract positions from trades
    positions_before = compute_net_positions(trades_before)
    positions_after = compute_net_positions(trades_after)
    
    # Check via P4's built-in method
    is_valid = policy.check_exposure_invariance(positions_before, positions_after)
    
    if not is_valid:
        print(f"CONSTRAINT VIOLATION: Net exposure drift detected")
        print(f"  Before: {sum(positions_before.values())}")
        print(f"  After: {sum(positions_after.values())}")
        print(f"  Reverting adjustment...")
        return False
    
    return True
```

**Mitigation if violation detected:**
1. Revert `policy.params` to previous values (stored in adjustment log)
2. Mark iteration with `constraint_violation=True` flag
3. Skip to next iteration without adjusting
4. If 3 consecutive violations, halt adaptive training and flag for manual review

---

## 5. Integration with P6 Standard Adversary

### 5.1 Adversary Training Protocol Alignment

P7 uses **identical training protocol** as P6 to ensure fair comparison:

| Component | P6 (Standard) | P7 (Adaptive) | Alignment |
|-----------|---------------|---------------|-----------|
| **Model** | LightGBM/XGBoost | LightGBM/XGBoost | ✅ Same |
| **Target** | Binary (trade in next Δt) | Binary (trade in next Δt) | ✅ Same |
| **Features** | Signal state, execution context, market microstructure | Identical feature set | ✅ Reuse P6 code |
| **Training** | Rolling walk-forward splits | Rolling walk-forward splits | ✅ Same |
| **Validation** | AUC on held-out fold | AUC on held-out fold | ✅ Same |
| **Policy params** | Fixed throughout | Adaptive (changes per iteration) | ⚠️ Key difference |

### 5.2 Code Reuse from P6

**P6 provides:**
- `train_adversary_classifier()` - Train LightGBM on trade features
- `extract_features()` - Generate feature matrix from trades
- `generate_labels()` - Create binary labels (trade in next Δt)
- `compute_auc()` - Calculate ROC-AUC with bootstrap CIs

**P7 imports and extends:**
```python
# P7 imports P6 components
from bsml.adversary.standard import (
    train_adversary_classifier,
    extract_features,
    generate_labels,
    compute_auc
)

# P7 adds adaptive loop wrapper
def adaptive_adversary_training_loop(baseline_strategy, policy_initial, data_splits):
    # Uses P6's functions inside adaptive iteration loop
    for iteration in range(n_iterations):
        trades = run_backtest(baseline_strategy, policy, data_splits['train'][iteration])
        
        # P6 code: train adversary
        adversary = train_adversary_classifier(
            features=extract_features(trades),
            labels=generate_labels(trades, delta_t='5min')
        )
        
        # P6 code: compute AUC
        auc_score = compute_auc(adversary, val_features, val_labels)
        
        # P7 code: adaptive adjustment
        adjustment = adaptive_step(auc_score, policy, iteration)
```

**P6 review status:** ✅ Approved code reuse approach (Nov 7)

### 5.3 Feature Engineering Alignment

P7 uses **exact same features** as P6 to ensure adversary has same information:

**Feature Categories (from P6 spec):**
1. **Signal state:** Last K signal values, slopes, z-scores, threshold distance
2. **Execution context:** Bar index, minutes since open/close, day-of-week
3. **Recent actions (lagged):** Last trade side/time/size
4. **Market microstructure:** Spread, returns, RV/ATR, volume percentile

**P7 does NOT add:**
- Policy ID as feature (would leak adaptive strategy)
- Iteration number (would leak training stage)
- Adjustment history (would leak feedback loop)

**Rationale:** Adversary must learn from observable market data only, not from knowledge of adaptive mechanism

---

## 6. Training Loop Specification

### 6.1 Main Loop Structure

```python
def adaptive_adversary_training_loop(
    baseline_strategy,
    policy_initial,
    data_splits,
    n_iterations=20,
    early_stop_patience=5
):
    """
    Main adaptive training loop with early stopping.
    
    Args:
        baseline_strategy: P2's deterministic strategy function
        policy_initial: Initial RandomizationPolicy from P4
        data_splits: Dict with 'train', 'val', 'test' walk-forward splits
        n_iterations: Maximum training iterations (default 20)
        early_stop_patience: Stop if in target range for N consecutive iters
    
    Returns:
        results: Dict containing:
            - auc_history: List of AUC scores per iteration
            - adjustment_log: List of adjustment records
            - final_policy_params: Final parameter values
            - final_metrics: Sharpe, MaxDD, ΔIS on test set
            - convergence_iteration: Iteration where converged (or None)
    """
    
    import copy
    from bsml.adversary.standard import (
        train_adversary_classifier,
        extract_features,
        generate_labels,
        compute_auc
    )
    
    # Initialize
    policy = copy.deepcopy(policy_initial)
    adjustment_log = []
    auc_history = []
    convergence_counter = 0
    converged_iteration = None
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE ADVERSARY TRAINING LOOP")
    print(f"Policy: {policy.__class__.__name__}")
    print(f"Initial params: {policy.params}")
    print(f"Max iterations: {n_iterations}")
    print(f"{'='*60}\n")
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
        
        # STEP 1: Generate trades with current policy
        print("  Generating trades...")
        trades = run_backtest(
            baseline_strategy,
            policy,
            data_splits['train'][iteration]
        )
        print(f"  Generated {len(trades)} trade events")
        
        # STEP 2: Train adversary classifier
        print("  Training adversary...")
        features = extract_features(trades)
        labels = generate_labels(trades, delta_t='5min')
        
        adversary = train_adversary_classifier(
            features=features,
            labels=labels,
            model='lightgbm',
            params={'max_depth': 5, 'num_leaves': 31}
        )
        
        # STEP 3: Evaluate on validation fold
        print("  Evaluating on validation...")
        val_trades = run_backtest(
            baseline_strategy,
            policy,
            data_splits['val'][iteration]
        )
        val_features = extract_features(val_trades)
        val_labels = generate_labels(val_trades, delta_t='5min')
        
        auc_score = compute_auc(adversary, val_features, val_labels)
        auc_history.append(auc_score)
        print(f"  Validation AUC: {auc_score:.3f}")
        
        # STEP 4: Adaptive adjustment decision
        print("  Evaluating adjustment...")
        adjustment = adaptive_step(auc_score, policy, iteration, adjustment_log)
        adjustment_log.append(adjustment)
        
        print(f"  Action: {adjustment['action']}")
        print(f"  Rationale: {adjustment['rationale']}")
        if adjustment['action'] != 'HOLD':
            print(f"  Updated params: {adjustment['params_after']}")
        
        # STEP 5: Exposure invariance check (if params changed)
        if adjustment['action'] in ['INCREASE', 'DECREASE', 'NUDGE_UP', 'NUDGE_DOWN']:
            print("  Checking exposure invariance...")
            # Generate small test batch with new params
            test_batch = run_backtest(
                baseline_strategy,
                policy,
                data_splits['train'][iteration][:100]  # Small sample
            )
            
            is_safe = verify_adjustment_safety(policy, trades, test_batch)
            
            if not is_safe:
                print("  ⚠️  EXPOSURE VIOLATION - Reverting adjustment")
                if len(adjustment_log) >= 2:
                    policy.params = adjustment_log[-2]['params_after']
                adjustment['reverted'] = True
                convergence_counter = 0  # Reset convergence
                continue
        
        # STEP 6: Check convergence (in target range?)
        AUC_TARGET_MIN, AUC_TARGET_MAX = 0.60, 0.70
        if AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
            convergence_counter += 1
            print(f"  ✓ In target range ({convergence_counter}/{early_stop_patience})")
            
            if convergence_counter >= early_stop_patience:
                converged_iteration = iteration + 1
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {converged_iteration}")
                print(f"AUC stable in target range for {early_stop_patience} iterations")
                print(f"{'='*60}\n")
                break
        else:
            convergence_counter = 0  # Reset if leaves target range
        
        # STEP 7: Oscillation detection
        if 'oscillation_warning' in adjustment and adjustment['oscillation_warning']:
            print("  ⚠️  OSCILLATION DETECTED - Consider reducing adjustment factor")
    
    # STEP 8: Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    test_trades = run_backtest(
        baseline_strategy,
        policy,
        data_splits['test']
    )
    
    # Compute final metrics (P5 functions)
    from bsml.metrics import compute_sharpe, compute_max_drawdown, compute_delta_is
    
    final_metrics = {
        'sharpe': compute_sharpe(test_trades),
        'max_drawdown': compute_max_drawdown(test_trades),
        'delta_is': compute_delta_is(test_trades, baseline_trades)
    }
    
    print(f"Final Sharpe: {final_metrics['sharpe']:.3f}")
    print(f"Final MaxDD: {final_metrics['max_drawdown']:.3f}")
    print(f"Final ΔIS: {final_metrics['delta_is']:.2f} bps")
    
    # Return results
    return {
        'policy_name': policy.__class__.__name__,
        'auc_history': auc_history,
        'adjustment_log': adjustment_log,
        'final_policy_params': policy.params,
        'final_metrics': final_metrics,
        'test_trades': test_trades,
        'converged_iteration': converged_iteration,
        'n_iterations_run': iteration + 1
    }
```

### 6.2 Early Stopping Criteria (FINALIZED)

Training stops if **any** of the following conditions met:

| Condition | Criterion | Rationale |
|-----------|-----------|-----------|
| **Convergence** | AUC in target range for 5 consecutive iterations | Stable optimal state reached |
| **Parameter bounds** | Any param hits min/max bound | Cannot adjust further |
| **Exposure violations** | 3 consecutive constraint violations | Adjustment strategy failing |
| **Iteration limit** | Reached `n_iterations=20` | Computational budget exhausted |

**Convergence patience = 5** chosen because:
- Need multiple iterations to confirm stability (not just luck)
- 5 iterations = ~25% of training budget (reasonable)
- Balances thoroughness vs. computational cost

### 6.3 Logging Schema

Each iteration logs the following to CSV:

**File:** `results/adaptive_log_<policy>_seed<seed>.csv`

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int | Iteration number (0-indexed) |
| `auc` | float | Validation AUC score |
| `action` | str | 'INCREASE', 'DECREASE', 'HOLD', 'NUDGE_UP', 'NUDGE_DOWN' |
| `rationale` | str | Human-readable reason for action |
| `timing_range_hours` | float | (Uniform) timing parameter after adjustment |
| `threshold_pct` | float | (Uniform) threshold parameter after adjustment |
| `sigma` | float | (OU) volatility parameter after adjustment |
| `scale` | float | (Pink) amplitude parameter after adjustment |
| `exposure_violation` | bool | True if adjustment violated constraint |
| `reverted` | bool | True if adjustment was reverted |
| `oscillation_warning` | bool | True if oscillation detected |
| `in_target_range` | bool | True if AUC in [0.60, 0.70] |

---

## 7. Comparative Analysis: Adaptive vs. Non-Adaptive

### 7.1 Experimental Design

For each policy (Baseline, Uniform, OU, Pink), P7 runs **two parallel experiments**:

| Variant | Description | Policy Params | Iterations |
|---------|-------------|---------------|------------|
| **Non-Adaptive (P6)** | Standard adversary, fixed params | Initial values (unchanged) | 1 (single run) |
| **Adaptive (P7)** | Adaptive adversary, dynamic params | Adjusted via feedback | 20 (or until convergence) |

**Controlled variables (same for both):**
- Seed: Same master seed for reproducibility
- Data splits: Identical train/val/test folds
- Adversary model: Same LightGBM architecture and hyperparams
- Features: Same feature engineering
- Cost model: Same transaction cost parameters

**Measured variables (different):**
- Final policy parameters
- AUC trajectory over iterations (adaptive only)
- Sharpe ratio on test set
- MaxDD on test set
- ΔIS vs. baseline

### 7.2 Comparison Metrics

**Primary Metrics (from P5 spec):**

1. **Sharpe Ratio (annualized)**
   - Formula: `(mean_return / std_return) × √252`
   - Higher is better
   - Compare: `Sharpe_adaptive` vs. `Sharpe_nonadaptive`

2. **Max Drawdown**
   - Formula: `min_t((E_t - P_t) / P_t)` where `P_t = max(E_u)` for `u ≤ t`
   - More negative = worse
   - Compare: `MaxDD_adaptive` vs. `MaxDD_nonadaptive`

3. **ΔIS (Implementation Shortfall)**
   - Formula: `IS_policy - IS_baseline`
   - Lower (more negative) = better
   - Compare: `ΔIS_adaptive` vs. `ΔIS_nonadaptive`

**Secondary Metrics (P7-specific):**

4. **Final AUC**
   - Adversary AUC on test set with final policy params
   - Lower = less predictable (better)
   - Compare: `AUC_adaptive` vs. `AUC_nonadaptive`

5. **Iterations to Convergence**
   - Number of iterations until AUC stable in target range
   - Only applicable to adaptive variant
   - Lower = faster convergence (better computational efficiency)

6. **Parameter Stability**
   - Coefficient of variation of parameters over last 5 iterations
   - Lower = more stable (less oscillation)

### 7.3 Statistical Testing

Use **block bootstrap** (as per P5 protocol) to compute confidence intervals:

```python
def compare_adaptive_vs_nonadaptive(results_adaptive, results_nonadaptive, n_bootstrap=1000):
    """
    Compare adaptive vs. non-adaptive using block bootstrap CIs.
    
    Returns:
        comparison_table: DataFrame with metrics and CIs
    """
    
    metrics = ['sharpe', 'max_drawdown', 'delta_is', 'final_auc']
    comparison = []
    
    for metric in metrics:
        # Bootstrap CI for adaptive
        adaptive_ci = block_bootstrap_ci(
            results_adaptive['test_trades'],
            metric=metric,
            n_bootstrap=n_bootstrap,
            block_size=5  # 5-day blocks
        )
        
        # Bootstrap CI for non-adaptive
        nonadaptive_ci = block_bootstrap_ci(
            results_nonadaptive['test_trades'],
            metric=metric,
            n_bootstrap=n_bootstrap,
            block_size=5
        )
        
        # Check if CIs overlap
        overlap = (adaptive_ci[0] <= nonadaptive_ci[1]) and (nonadaptive_ci[0] <= adaptive_ci[1])
        
        comparison.append({
            'metric': metric,
            'adaptive_mean': results_adaptive['final_metrics'][metric],
            'adaptive_ci_low': adaptive_ci[0],
            'adaptive_ci_high': adaptive_ci[1],
            'nonadaptive_mean': results_nonadaptive['final_metrics'][metric],
            'nonadaptive_ci_low': nonadaptive_ci[0],
            'nonadaptive_ci_high': nonadaptive_ci[1],
            'ci_overlap': overlap,
            'significant_difference': not overlap
        })
    
    return pd.DataFrame(comparison)
```

**Interpretation:**
- If CIs do not overlap → statistically significant difference
- If CIs overlap → cannot reject null hypothesis (no difference)

---

## 8. Deliverables & Outputs

### 8.1 Code Deliverables

| Week | File | Description |
|------|------|-------------|
| **Week 2** | `adaptive_adversary_v0.1.py` | Skeleton implementation with toy data tests |
| **Week 2** | `tests/test_adaptive_loop.py` | Unit tests for adjustment logic |
| **Week 3** | `adaptive_adversary_v1.py` | Full implementation with all policies |
| **Week 3** | `scripts/run_adaptive_pilot.py` | Pilot run on Uniform policy |
| **Week 4** | `scripts/run_adaptive_all_policies.py` | Full sweep across all policies |
| **Week 4** | `scripts/compare_adaptive_vs_nonadaptive.py` | Comparative analysis script |
| **Finish** | `scripts/reproduce_adaptive.sh` | One-command reproduction |

### 8.2 Results & Figures

| Week | Output | Description |
|------|--------|-------------|
| **Week 3** | `results/uniform_pilot_adaptive.csv` | Pilot results on Uniform policy |
| **Week 3** | `figures/auc_trajectory_uniform.png` | AUC over iterations (pilot) |
| **Week 4** | `results/adaptive_vs_nonadaptive_all_policies.csv` | Full comparison table |
| **Week 4** | `figures/adaptive_comparison_sharpe.png` | Side-by-side Sharpe comparison |
| **Week 4** | `figures/adaptive_comparison_maxdd.png` | Side-by-side MaxDD comparison |
| **Week 4** | `figures/adaptive_comparison_delta_is.png` | Side-by-side ΔIS comparison |
| **Week 4** | `figures/param_evolution_uniform.png` | Parameter trajectory (Uniform) |
| **Week 4** | `figures/param_evolution_ou.png` | Parameter trajectory (OU) |
| **Week 4** | `figures/param_evolution_pink.png` | Parameter trajectory (Pink) |
| **Week 4** | `results/convergence_summary.csv` | Iterations to convergence by policy |

### 8.3 Documentation Deliverables

| Week | Document | Description |
|------|----------|-------------|
| **Week 1** | `docs/adaptive_adversary_framework_spec_v1.0.md` | **This document** |
| **Week 2** | `docs/adjustment_logic.md` | Detailed adjustment rules |
| **Week 2** | `docs/integration_with_P4_P6.md` | Integration details |
| **Week 3** | `docs/tuning_notes_week3.md` | Observations on threshold tuning |
| **Week 4** | `docs/final_hyperparameters.md` | Final tuned values and rationale |
| **Finish** | `paper_sections/adaptive_adversary_methods.tex` | Methods section for paper |
| **Finish** | `paper_sections/adaptive_adversary_results.tex` | Results section for paper |

### 8.4 Key Figures for Paper

**Figure 1: AUC Trajectory**
- X-axis: Iteration number
- Y-axis: Validation AUC
- Lines: One per policy (Uniform, OU, Pink)
- Shaded region: Target range [0.60, 0.70]
- Horizontal lines: AUC_HIGH (0.75), AUC_LOW (0.55)

**Figure 2: Parameter Evolution (Uniform)**
- X-axis: Iteration number
- Y-axis (left): `timing_range_hours`
- Y-axis (right): `threshold_pct`
- Shows how parameters adjust over training

**Figure 3: Adaptive vs. Non-Adaptive Comparison**
- 3-panel bar chart
- Panel 1: Sharpe ratio (adaptive vs. non-adaptive for each policy)
- Panel 2: MaxDD (adaptive vs. non-adaptive for each policy)
- Panel 3: ΔIS (adaptive vs. non-adaptive for each policy)
- Error bars: 95% CIs from bootstrap

**Figure 4: Convergence Analysis**
- Bar chart: Iterations to convergence by policy
- Shows computational cost vs. benefit

---

## 9. Risks & Mitigations

### 9.1 Risk: Adaptive Loop Instability

**Description:** Parameters oscillate (increase → decrease → increase) without converging.

**Probability:** Medium (20-30% chance based on RL literature)

**Impact:** High (wastes computational budget, no useful results)

**Mitigation Strategy:**
1. **Momentum/Smoothing:** Only adjust if AUC outside target for 2+ consecutive iterations
2. **Exponential Moving Average:** Smooth AUC with EMA before decision (α=0.3)
3. **Reduce Adjustment Factor:** If oscillations detected, switch from 1.2/0.8 to 1.1/0.9
4. **Early Detection:** Flag oscillations after 3 alternating actions, pause for manual review

**Implementation:**
```python
# Add to adaptive_step()
def compute_smoothed_auc(auc_history, alpha=0.3):
    """Exponential moving average of AUC scores."""
    if len(auc_history) == 0:
        return None
    ema = auc_history[0]
    for auc in auc_history[1:]:
        ema = alpha * auc + (1 - alpha) * ema
    return ema

# Use smoothed AUC for decisions if oscillations detected
```

**P1 Sign-Off:** ✅ Approved mitigation plan (Nov 7)

### 9.2 Risk: Exposure Constraint Violations

**Description:** Increased randomness causes net exposure to drift beyond ±5% tolerance.

**Probability:** Low (10-15% based on P4 testing)

**Impact:** High (violates finance guardrails, results invalid)

**Mitigation Strategy:**
1. **Pre-Check:** Test adjustment on small batch before full backtest
2. **Automatic Revert:** If violation detected, immediately revert params
3. **Hard Bounds:** Enforce parameter ranges that P4 validated as safe
4. **Manual Review:** If 3 consecutive violations, halt and flag for P1/P2 review

**Parameter Bounds (P4-validated):**
- `timing_range_hours`: [0.5, 6.0] - tested safe by P4
- `threshold_pct`: [0.05, 0.25] - tested safe by P4
- `sigma` (OU): [0.01, 0.15] - tested safe by P4
- `scale` (Pink): [0.02, 0.20] - tested safe by P4

**P2 Sign-Off:** ✅ Approved constraint checks (Nov 6)

### 9.3 Risk: No Improvement vs. Non-Adaptive

**Description:** Adaptive adversary shows no statistically significant improvement over fixed parameters.

**Probability:** Medium (40-50% - this is novel research)

**Impact:** Low (null result is still publishable)

**Mitigation Strategy:**
1. **Document Honestly:** Null results are valuable scientific findings
2. **Investigate Regimes:** May help in specific market conditions (high vol, low liquidity)
3. **Analyze Why:** Understand failure modes - too aggressive? Too conservative?
4. **Discussion Section:** Include thoughtful analysis of when/why adaptation doesn't help

**Acceptable Outcomes:**
- Adaptive improves ΔIS by >5% → Strong positive result
- Adaptive shows no difference (CIs overlap) → Informative null result
- Adaptive degrades performance → Important negative result (document failure modes)

**P1 Sign-Off:** ✅ Agreed null results acceptable (Nov 7)

### 9.4 Risk: Computational Budget Exceeded

**Description:** Adaptive training requires 20× more compute than non-adaptive (20 iterations vs. 1).

**Probability:** Medium (30-40% if full symbol universe used)

**Impact:** Medium (delays timeline, incomplete results)

**Mitigation Strategy:**
1. **Prioritize:** Focus on most informative policies first (Uniform, then OU, then Pink)
2. **Smaller Universe:** Use subset of symbols for Week 3 pilot (coordinate with P1)
3. **Parallel Runs:** P3 manages parallel execution across policies (coordinate with P3)
4. **Caching:** Cache intermediate results to avoid recomputation (P3 responsibility)
5. **Early Stopping:** Use convergence criteria to avoid unnecessary iterations

**Compute Budget Estimate:**
- Non-adaptive: 4 policies × 1 iteration × 1 hour = 4 hours
- Adaptive: 3 policies × 20 iterations × 1 hour = 60 hours
- With early stopping (avg 12 iters): 3 policies × 12 iters × 1 hour = 36 hours
- **Total: ~40 hours compute (acceptable within timeline)**

**P3 Sign-Off:** ✅ Approved parallel execution plan (Nov 8)

---

## 10. Coordination Plan

### 10.1 Dependencies Timeline

| Week | Dependency | Required From | Status |
|------|------------|---------------|--------|
| **Week 1** | Parameter bounds validation | P4 | ✅ Complete |
| **Week 1** | Adversary training code | P6 | ✅ Ready to reuse |
| **Week 2** | Backtest runner API | P3 | ⏳ In progress |
| **Week 2** | adjust_stochasticity() tested | P4 | ⏳ Testing Week 2 |
| **Week 3** | Metrics computation functions | P5 | 📅 Due Week 3 |
| **Week 4** | Parallel run infrastructure | P3 | 📅 Due Week 4 |

### 10.2 Weekly Sync Agenda

**Week 1 Sync (Nov 8):**
- ✅ Review this finalized spec document
- ✅ Confirm sign-offs from all teams
- ✅ Discuss Week 2 implementation plan
- ✅ Resolve open questions

**Week 2 Sync (Nov 15):**
- Demo adaptive_adversary_v0.1 skeleton
- Review P4 integration test results
- Discuss any API changes needed
- Plan Week 3 pilot experiment

**Week 3 Sync (Nov 22):**
- Present Uniform pilot results
- Review AUC trajectories and convergence
- Decide if threshold tuning needed (0.75/0.55 vs. 0.70/0.60)
- Coordinate Week 4 full sweeps with P3/P5

**Week 4 Sync (Nov 29):**
- Review full comparative results (all policies)
- Finalize paper figures and tables
- Discuss adaptive vs. non-adaptive interpretation
- Plan Finish window tasks

### 10.3 Communication Channels

**For quick questions:**
- Slack channel: #bsml-adaptive-adversary
- Tag: @P7, @P4 (Neel), @P6, @P5, @P3

**For code reviews:**
- GitHub PRs with reviewers: P4 (randomization changes), P6 (adversary code), P3 (infra)

**For design discussions:**
- Weekly sync meetings (Fridays 2pm)
- Ad-hoc pairing sessions as needed

**For blocking issues:**
- Escalate to P1 (Ray) immediately
- Document in project log for timeline tracking

---

## 11. Success Criteria

### 11.1 Technical Success (Required)

✅ **Convergence:** Adaptive training converges (AUC stable in target range) for at least 2 out of 3 policies

✅ **Reproducibility:** All results regenerate from single command with fixed seeds

✅ **Comparative Analysis:** Clean head-to-head comparison (adaptive vs. non-adaptive) for all policies

✅ **Constraint Compliance:** No exposure invariance violations in final test results

✅ **Code Quality:** Passes all unit tests, documented, reviewed by P4/P6

### 11.2 Scientific Success (Aspirational)

⭐ **Strong Positive:** Adaptive improves ΔIS by >5% vs. non-adaptive (p < 0.05)

⭐ **Moderate Positive:** Adaptive improves ΔIS by 2-5% vs. non-adaptive (95% CI excludes zero)

⭐ **Informative Null:** No statistically significant difference, but clear understanding of why

⭐ **Regime-Specific:** Adaptive helps in specific regimes (e.g., high volatility periods)

### 11.3 Deliverable Success (Required)

✅ **Code:** All files in Section 8.1 delivered on schedule

✅ **Results:** All CSVs and figures in Section 8.2 delivered on schedule

✅ **Documentation:** All docs in Section 8.3 delivered on schedule

✅ **Paper Sections:** Methods and Results sections ready for P1 to integrate

✅ **Reproducibility:** P3 confirms one-command reproduction works

---

## 12. Open Questions RESOLVED

| Question | Answer | Resolved By | Date |
|----------|--------|-------------|------|
| AUC thresholds: 0.75/0.55 or adjust? | **0.75/0.55 locked** | Team consensus | Nov 7 |
| Adjustment factor: 1.2/0.8 or 1.1/0.9? | **1.2/0.8 locked** (can tune Week 3) | P4 + Team | Nov 7 |
| Convergence: 5 iters or continue to 20? | **5 consecutive iters = early stop** | Team consensus | Nov 8 |
| Regime-specific thresholds? | **No, use global thresholds** (simplicity) | P1 + P5 | Nov 8 |
| P6 code reuse approach? | **Direct import + extend** | P6 + P7 | Nov 7 |
| Market hours constraint? | **Yes, respect RTH** (via P4's perturbation bounds) | P1 + P2 | Nov 6 |

**All open questions from Day 0 are now RESOLVED. Spec is locked for Week 2 implementation.**

---

## 13. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 (Draft) | Nov 2, 2025 | P7 | Day 0 initial design |
| 0.9 (Review) | Nov 5, 2025 | P7 | Incorporated team feedback |
| **1.0 (Final)** | **Nov 9, 2025** | **P7** | **All sign-offs complete, spec locked** |

---

## 14. Appendices

### Appendix A: Glossary

- **AUC (Area Under Curve):** ROC-AUC metric for binary classifier performance [0.5, 1.0]
- **Adaptive Adversary:** P7's dynamic parameter adjustment framework
- **Standard Adversary:** P6's fixed-parameter adversary baseline
- **Stochasticity:** Amount of randomness in policy (adjustable via P4 API)
- **Exposure Invariance:** Constraint that net exposure ≈ 0 ± 5%
- **Walk-Forward:** Time-series cross-validation with no future data leakage
- **ΔIS (Delta Implementation Shortfall):** Cost difference vs. ideal execution

### Appendix B: Mathematical Notation

- `s_t`: Signal value at time t
- `θ`: Threshold parameter
- `σ`: Volatility parameter (OU policy)
- `α`: Spectral exponent (Pink policy)
- `E_t`: Equity curve at time t
- `P_t`: Running peak of equity
- `AUC_H`: AUC_HIGH_THRESHOLD = 0.75
- `AUC_L`: AUC_LOW_THRESHOLD = 0.55
- `AUC_T`: AUC_TARGET_RANGE = (0.60, 0.70)

### Appendix C: Code Style Guidelines

**Python Style:**
- Follow PEP 8
- Type hints for function signatures
- Docstrings in NumPy format
- Maximum line length: 100 characters

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `AdaptiveAdversary`)
- Functions: `snake_case` (e.g., `adaptive_step`)
- Constants: `UPPER_CASE` (e.g., `AUC_HIGH_THRESHOLD`)
- Private methods: `_leading_underscore`

**Testing:**
- Pytest framework
- Test coverage >80%
- Fixtures for repeated setup
- Mock external dependencies

---

**END OF SPECIFICATION**

**Status:** ✅ FINALIZED - Ready for Week 2 Implementation  
**Next Milestone:** Deliver `adaptive_adversary_v0.1.py` by Nov 16  
**Owner:** P7  
