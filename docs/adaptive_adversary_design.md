# P7 Day 0 Deliverables - Adaptive Adversary Framework Design

**Date:** November 2, 2025  
**Owner:** P7 (Adaptive Adversary Framework)  
**Status:** Day 0 Complete

---

## Executive Summary

This document defines the **Adaptive Adversary Framework** that dynamically adjusts randomization policy parameters based on predictability feedback during training. The framework integrates P6's standard adversary classifier with P4's randomization modules to create a closed-loop system that balances unpredictability against expected returns.

**Core Innovation:** Unlike the static standard adversary (P6), the adaptive adversary monitors AUC scores in real-time and adjusts policy stochasticity parameters to maintain an optimal predictability-performance trade-off.

---

## 1. Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTIVE TRAINING LOOP                    │
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │   Baseline   │──→   │ Randomization│──→   │  Execute   │ │
│  │  Strategy    │      │   Policy     │      │   Trades   │ │
│  │    (P2)      │      │    (P4)      │      │   (P3)     │ │
│  └──────────────┘      └──────┬───────┘      └─────┬──────┘ │
│                               │                     │        │
│                               │ adjust              │        │
│                               │ stochasticity       │ trade  │
│                               │                     │ data   │
│                               │                     ↓        │
│                        ┌──────┴─────────────────────────┐   │
│                        │   ADAPTIVE ADVERSARY (P7)      │   │
│                        │                                 │   │
│                        │  1. Train classifier (P6 spec) │   │
│                        │  2. Compute AUC score          │   │
│                        │  3. Evaluate vs thresholds     │   │
│                        │  4. Adjust policy params       │   │
│                        │  5. Log adjustment history     │   │
│                        └────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Adaptive Feedback Mechanism

### 2.1 AUC-Based Adjustment Logic

The core decision rule uses AUC thresholds to trigger parameter adjustments:

```python
# Pseudocode for adaptive adjustment
def adaptive_step(auc_score, policy, iteration):
    """
    Core adaptive logic executed after each adversary training epoch.
    
    Args:
        auc_score: Current adversary AUC [0.5, 1.0]
        policy: RandomizationPolicy instance from P4
        iteration: Current training iteration
    
    Returns:
        adjustment_decision: dict with action and rationale
    """
    
    # Define thresholds
    AUC_HIGH_THRESHOLD = 0.75   # Too predictable
    AUC_LOW_THRESHOLD = 0.55    # Too random
    AUC_TARGET_RANGE = (0.60, 0.70)  # Optimal range
    
    # Decision logic
    if auc_score > AUC_HIGH_THRESHOLD:
        # Strategy is too predictable → increase randomness
        policy.adjust_stochasticity(auc_score, direction='increase')
        action = 'INCREASE'
        rationale = f"AUC {auc_score:.3f} > {AUC_HIGH_THRESHOLD} (too predictable)"
        
    elif auc_score < AUC_LOW_THRESHOLD:
        # Strategy is too random → decrease randomness
        policy.adjust_stochasticity(auc_score, direction='decrease')
        action = 'DECREASE'
        rationale = f"AUC {auc_score:.3f} < {AUC_LOW_THRESHOLD} (too random)"
        
    elif AUC_TARGET_RANGE[0] <= auc_score <= AUC_TARGET_RANGE[1]:
        # In optimal range → no adjustment
        action = 'HOLD'
        rationale = f"AUC {auc_score:.3f} in target range {AUC_TARGET_RANGE}"
        
    else:
        # Between thresholds → minor adjustment toward target
        target_midpoint = (AUC_TARGET_RANGE[0] + AUC_TARGET_RANGE[1]) / 2
        if auc_score > target_midpoint:
            policy.adjust_stochasticity(auc_score, direction='increase')
            action = 'NUDGE_UP'
        else:
            policy.adjust_stochasticity(auc_score, direction='decrease')
            action = 'NUDGE_DOWN'
        rationale = f"AUC {auc_score:.3f} nudging toward target {target_midpoint:.2f}"
    
    return {
        'iteration': iteration,
        'auc': auc_score,
        'action': action,
        'rationale': rationale,
        'params_after': policy.params.copy()
    }
```

### 2.2 Adjustment Parameters

**Initial Thresholds (Week 1):**
- `AUC_HIGH_THRESHOLD = 0.75` - Trigger increase if adversary predicts too well
- `AUC_LOW_THRESHOLD = 0.55` - Trigger decrease if too random (barely better than coin flip)
- `AUC_TARGET_RANGE = (0.60, 0.70)` - Optimal "sweet spot"

**Adjustment Factor (from P4 API):**
- Increase: multiply stochastic params by 1.2 (20% more randomness)
- Decrease: multiply by 0.8 (20% less randomness)

**Rationale:** 
- Start conservative with 20% adjustments
- Monitor convergence in Week 2-3
- Tune based on empirical results (may change to 1.15/0.85 if 1.2/0.8 is too aggressive)

---

## 3. Integration with P4 Randomization Modules

### 3.1 Policy Parameter Mapping

Each policy has stochastic parameters that the adaptive adversary can adjust:

| Policy | Adjustable Parameters | Initial Values | Adjustment Impact |
|--------|----------------------|----------------|-------------------|
| **Uniform** | `timing_range_hours` | 2.0 | ±2 hours → ±2.4 hours (increase) |
|  | `threshold_pct` | 0.10 | ±10% → ±12% (increase) |
| **OU** | `sigma` (volatility) | 0.05 | 0.05 → 0.06 (increase) |
|  | `theta` (mean reversion) | 0.15 | Indirect effect via sigma |
| **Pink** | `scale` (amplitude) | 0.08 | 0.08 → 0.096 (increase) |
|  | `alpha` (spectral exp) | 1.0 | Fixed (changing alpha changes noise character) |

### 3.2 API Contract with P4

P7 calls P4's `adjust_stochasticity()` method as defined in P4's API spec:

```python
# P7 → P4 interface
policy.adjust_stochasticity(
    auc_score=0.82,      # Current adversary AUC
    direction='increase'  # 'increase' or 'decrease'
)

# P4 internally modifies self.params in-place
# P7 logs the updated params for analysis
```

**Constraint Check:** After each adjustment, P7 must verify:
- Exposure invariance still holds (via P4's `check_exposure_invariance()`)
- Parameters stay within reasonable bounds (e.g., timing_range < 6 hours)

---

## 4. Integration with P6 Standard Adversary

### 4.1 Training Protocol Alignment

The adaptive adversary uses **the same training protocol** as P6's standard adversary:

**From P6 spec:**
- Model: LightGBM/XGBoost gradient boosted trees
- Target: Binary (will policy place order in next Δt?)
- Features: Signal state, execution context, recent actions, market microstructure
- Training: Rolling walk-forward splits
- Metrics: AUC (ROC-AUC) with block bootstrap CIs

**P7 Addition:**
- After each training epoch, compute AUC on validation fold
- Feed AUC back to adjustment logic
- Retrain adversary on next iteration with adjusted policy

### 4.2 Comparative Analysis

P7 produces **two parallel runs** for each policy:

1. **Non-adaptive (baseline):** Standard adversary with fixed policy params (P6's approach)
2. **Adaptive:** Adversary with dynamic policy param adjustments (P7's approach)

**Deliverable:** Side-by-side comparison showing:
- AUC trajectories over training iterations
- Final policy parameters
- Performance metrics (Sharpe, MaxDD, ΔIS) from P5

---

## 5. Training Loop Specification

### 5.1 Pseudocode

```python
def adaptive_adversary_training_loop(
    baseline_strategy,
    policy_initial,
    data_splits,
    n_iterations=20
):
    """
    Main adaptive training loop.
    
    Args:
        baseline_strategy: P2's deterministic strategy
        policy_initial: Initial RandomizationPolicy from P4
        data_splits: Walk-forward train/val/test splits from P3
        n_iterations: Number of adaptive training iterations
    
    Returns:
        results: dict with AUC history, param history, trades, metrics
    """
    
    # Initialize
    policy = copy.deepcopy(policy_initial)
    adjustment_log = []
    auc_history = []
    
    for iteration in range(n_iterations):
        print(f"\n=== Adaptive Iteration {iteration+1}/{n_iterations} ===")
        
        # 1. Generate trades with current policy
        trades = run_backtest(
            baseline_strategy,
            policy,
            data_splits['train'][iteration]
        )
        
        # 2. Train adversary classifier (P6 protocol)
        adversary = train_adversary_classifier(
            trades=trades,
            features=extract_features(trades),
            labels=generate_labels(trades, delta_t='5min'),
            model='lightgbm'
        )
        
        # 3. Evaluate on validation fold
        val_trades = run_backtest(
            baseline_strategy,
            policy,
            data_splits['val'][iteration]
        )
        val_features = extract_features(val_trades)
        val_labels = generate_labels(val_trades, delta_t='5min')
        
        auc_score = compute_auc(adversary, val_features, val_labels)
        auc_history.append(auc_score)
        
        print(f"Validation AUC: {auc_score:.3f}")
        
        # 4. Adaptive adjustment
        adjustment = adaptive_step(auc_score, policy, iteration)
        adjustment_log.append(adjustment)
        
        print(f"Action: {adjustment['action']}")
        print(f"Rationale: {adjustment['rationale']}")
        print(f"Updated params: {adjustment['params_after']}")
        
        # 5. Exposure invariance check
        if not verify_exposure_invariance(trades, policy):
            print("WARNING: Exposure constraint violated - reverting adjustment")
            policy.params = adjustment_log[-2]['params_after']  # Revert
    
    # Final evaluation on test set
    test_trades = run_backtest(
        baseline_strategy,
        policy,
        data_splits['test']
    )
    
    final_metrics = compute_metrics(test_trades)  # Sharpe, MaxDD, ΔIS from P5
    
    return {
        'auc_history': auc_history,
        'adjustment_log': adjustment_log,
        'final_policy_params': policy.params,
        'final_metrics': final_metrics,
        'test_trades': test_trades
    }
```

### 5.2 Early Stopping Criteria

Stop adaptive training if:
1. AUC stays in target range for 5 consecutive iterations (converged)
2. Parameters hit boundary constraints (e.g., timing_range > 6 hours)
3. Exposure invariance violations exceed 3 in a row
4. Iteration limit reached (n_iterations = 20)

---

## 6. Deliverables & Outputs

### 6.1 Week-by-Week Outputs

**Week 2 (Nov 10-16):**
- `adaptive_adversary_v0.1.py` - Skeleton implementation
- `tests/test_adaptive_loop.py` - Unit tests on toy data
- `docs/adjustment_logic.md` - Documentation of decision rules

**Week 3 (Nov 17-23):**
- `adaptive_adversary_v1.py` - Full implementation
- `results/uniform_pilot_adaptive.csv` - Pilot results on Uniform policy
- `figures/auc_trajectory_uniform.png` - AUC over iterations plot
- `docs/tuning_notes_week3.md` - Observations on threshold tuning

**Week 4 (Nov 24-30):**
- `results/adaptive_vs_nonadaptive_all_policies.csv` - Full comparison table
- `figures/adaptive_comparison_*.png` - Comparative plots (4 policies × 3 metrics)
- `results/param_evolution_*.csv` - Parameter trajectories over training
- `docs/final_hyperparameters.md` - Tuned threshold values and rationale

**Finish (Dec 1-3):**
- `scripts/reproduce_adaptive.sh` - One-command reproduction
- `paper_sections/adaptive_adversary_methods.tex` - Methods section for paper
- `paper_sections/adaptive_adversary_results.tex` - Results section for paper

### 6.2 Key Figures for Paper

1. **AUC Trajectory Plot:** AUC over training iterations, with target range shaded
2. **Parameter Evolution:** Stochastic params (e.g., timing_range) over iterations
3. **Adaptive vs. Non-Adaptive Comparison:** Side-by-side bar charts for Sharpe, MaxDD, ΔIS
4. **Adjustment Frequency Heatmap:** Which policies/regimes required most adjustments
5. **Convergence Analysis:** Iterations to reach target AUC range by policy

---

## 7. Risks & Mitigations

### 7.1 Risk: Adaptive Loop Instability

**Problem:** Adjustments cause oscillations (increase → decrease → increase...)

**Mitigation:**
- Add momentum/smoothing: only adjust if AUC outside target for 2+ consecutive iterations
- Implement exponential moving average of AUC scores
- Reduce adjustment factor from 1.2/0.8 to 1.15/0.85 if oscillations observed

### 7.2 Risk: Exposure Constraint Violations

**Problem:** Increased randomness breaks exposure invariance

**Mitigation:**
- Check exposure after every adjustment (P4's `check_exposure_invariance()`)
- Revert adjustment if constraint violated
- Add hard bounds on parameter ranges (e.g., timing_range ∈ [0.5, 6.0] hours)

### 7.3 Risk: No Improvement vs. Non-Adaptive

**Problem:** Adaptive adversary shows no benefit (null result)

**Mitigation:**
- Document findings honestly - comparative analysis valuable even if null
- Investigate regime-specific effects (may help in high-vol regimes only)
- Include in Discussion section as informative negative result

### 7.4 Risk: Computational Budget

**Problem:** Adaptive training requires ~20 iterations per policy (vs. 1 for non-adaptive)

**Mitigation:**
- P3 manages parallel runs and caching
- Prioritize most informative sweeps (coordinate with P5)
- Use smaller symbol universe for adaptive pilots (Week 3)

---

## 8. Coordination Points

### 8.1 Dependencies

**P4 (Randomization Modules) - Week 1:**
- Confirm `adjust_stochasticity()` API works as specified
- Agree on parameter bounds and adjustment factors
- Test exposure checks with toy examples

**P6 (Standard Adversary) - Week 2:**
- Align on adversary training code (can P7 reuse P6's implementation?)
- Confirm AUC computation matches (same sklearn function, same CV folds)
- Share feature engineering code to avoid duplication

**P5 (Experiments & Statistics) - Week 3:**
- Coordinate on metrics computation (Sharpe, MaxDD, ΔIS)
- Align on confidence interval methods (block bootstrap parameters)
- Decide on regime definitions for per-regime analysis

**P3 (Infrastructure) - Ongoing:**
- Support for parallel runs (adaptive vs. non-adaptive)
- Logging schema for adjustment history
- CSV output formats for P7-specific results

### 8.2 Weekly Sync Agenda Items

**Week 1:** Review this design doc; get sign-off on thresholds  
**Week 2:** Demo v0.1 skeleton; discuss integration issues  
**Week 3:** Present Uniform pilot results; adjust thresholds if needed  
**Week 4:** Review full comparative results; finalize paper figures

---

## 9. Open Questions for Team Review

1. **AUC thresholds:** Are 0.75 (high) and 0.55 (low) reasonable starting points? Should we be more/less aggressive?

2. **Adjustment factor:** Is 1.2/0.8 (20% change) appropriate, or should we start with 1.1/0.9 (10% change)?

3. **Convergence criteria:** Should we stop after 5 iterations in target range, or continue to n_iterations for comparability?

4. **Regime-specific tuning:** Should adaptive thresholds differ by regime (e.g., higher threshold in low-vol regimes)?

5. **P6 code reuse:** Can P7 directly import and extend P6's adversary training code, or should it be reimplemented?

6. **Market hours constraint:** Should timing adjustments respect market hours (9:30 AM - 4:00 PM), or can we perturb outside RTH?

---

## 10. Success Criteria

The adaptive adversary framework is successful if:

✅ **Technical:** Achieves stable convergence (AUC in target range) for at least 2 out of 3 policies (Uniform, OU, Pink)

✅ **Reproducibility:** All results regenerate from one command with fixed seeds

✅ **Comparative:** Produces clean head-to-head comparison (adaptive vs. non-adaptive) across all policies

✅ **Financial:** Does not violate exposure constraints or significantly degrade Sharpe ratio vs. non-adaptive

✅ **Scientific:** Generates publication-quality figures and clear interpretation for paper Discussion section

**Stretch Goal:** Adaptive adversary improves ΔIS by >5% vs. non-adaptive while maintaining Sharpe ratio within 95% CI

---

## 11. Timeline Checkpoint

**Day 0 (Today) - Completed:**
- ✅ Reviewed project timeline and team roles
- ✅ Studied P4's RNG API specification
- ✅ Studied P6's standard adversary plan
- ✅ Studied P2's baseline strategy spec
- ✅ Studied P5's metrics definitions
- ✅ Drafted adaptive adversary framework design (this document)

---

**Document Status:** ✅ Day 0 Complete - Ready for Team Review

**Author:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Version:** 1.0  
