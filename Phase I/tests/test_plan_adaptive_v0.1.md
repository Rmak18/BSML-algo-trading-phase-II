# Test Plan for Adaptive Adversary v0.1

**Project:** BSML - Randomized Execution Research  
**Owner:** P7 (Adaptive Adversary Framework)  
**Date:** November 14, 2025  
**Status:** Week 1 Deliverable

---

## 1. Overview

This document specifies the test plan for the Week 2 skeleton implementation (`adaptive_adversary_v0.1.py`). The goal is to validate core adaptive logic on toy data before integrating with real backtests in Week 3.

**Testing Philosophy:**
- Start simple: test individual components in isolation
- Build up: test component interactions
- Validate: test end-to-end skeleton on toy data
- No real data required for v0.1 - use synthetic/mock data

---

## 2. Test Scope

### 2.1 In Scope for v0.1

✅ **Adjustment logic:**
- AUC threshold evaluation
- Action decision (INCREASE, DECREASE, HOLD, NUDGE)
- Multiplier application

✅ **P4 integration:**
- `adjust_stochasticity()` API calls
- Parameter modification
- Boundary enforcement

✅ **Convergence detection:**
- Target range checking
- Consecutive iteration counting
- Early stopping logic

✅ **Oscillation detection:**
- Pattern recognition (INCREASE→DECREASE→INCREASE)
- Warning flags

✅ **Logging:**
- Adjustment record structure
- History tracking

### 2.2 Out of Scope for v0.1

❌ **Real backtesting:** Use mock trades, not P3's full runner

❌ **Real adversary training:** Use mock AUC scores, not P6's classifier

❌ **Real metrics:** Use placeholder Sharpe/MaxDD/ΔIS, not P5's computation

❌ **Multiple policies:** Focus on Uniform only in v0.1

❌ **Parallel runs:** Single-threaded execution only

**Rationale:** v0.1 is a skeleton to validate adaptive logic. Full integration happens in v1.0 (Week 3).

---

## 3. Unit Tests

### 3.1 Adjustment Decision Logic

#### Test 3.1.1: High AUC triggers INCREASE

```python
def test_high_auc_triggers_increase():
    """AUC > 0.75 should trigger INCREASE action."""
    from adaptive_adversary import decide_adjustment
    
    action, multiplier = decide_adjustment(auc_score=0.82)
    
    assert action == 'INCREASE'
    assert multiplier == 1.20
```

**Expected behavior:** When adversary predicts too well (AUC > 0.75), system should increase randomness.

---

#### Test 3.1.2: Low AUC triggers DECREASE

```python
def test_low_auc_triggers_decrease():
    """AUC < 0.55 should trigger DECREASE action."""
    from adaptive_adversary import decide_adjustment
    
    action, multiplier = decide_adjustment(auc_score=0.52)
    
    assert action == 'DECREASE'
    assert multiplier == 0.80
```

**Expected behavior:** When adversary cannot predict (AUC < 0.55), system should decrease randomness.

---

#### Test 3.1.3: Target range triggers HOLD

```python
def test_target_range_triggers_hold():
    """AUC in [0.60, 0.70] should trigger HOLD action."""
    from adaptive_adversary import decide_adjustment
    
    # Test lower bound
    action, multiplier = decide_adjustment(auc_score=0.60)
    assert action == 'HOLD'
    assert multiplier is None
    
    # Test midpoint
    action, multiplier = decide_adjustment(auc_score=0.65)
    assert action == 'HOLD'
    assert multiplier is None
    
    # Test upper bound
    action, multiplier = decide_adjustment(auc_score=0.70)
    assert action == 'HOLD'
    assert multiplier is None
```

**Expected behavior:** When AUC in optimal range, no adjustment needed.

---

#### Test 3.1.4: Between thresholds triggers NUDGE

```python
def test_between_thresholds_triggers_nudge():
    """AUC between thresholds should trigger gentle nudge."""
    from adaptive_adversary import decide_adjustment
    
    # Slightly above target (0.70 < AUC < 0.75)
    action, multiplier = decide_adjustment(auc_score=0.72)
    assert action == 'NUDGE_UP'
    assert multiplier == 1.20
    
    # Slightly below target (0.55 < AUC < 0.60)
    action, multiplier = decide_adjustment(auc_score=0.57)
    assert action == 'NUDGE_DOWN'
    assert multiplier == 0.80
```

**Expected behavior:** Between thresholds, nudge toward target midpoint (0.65).

---

#### Test 3.1.5: Edge cases

```python
def test_edge_cases():
    """Test boundary values exactly at thresholds."""
    from adaptive_adversary import decide_adjustment
    
    # Exactly at high threshold
    action, _ = decide_adjustment(auc_score=0.75)
    assert action == 'INCREASE'  # >= 0.75 triggers increase
    
    # Exactly at low threshold
    action, _ = decide_adjustment(auc_score=0.55)
    assert action == 'DECREASE'  # <= 0.55 triggers decrease
    
    # Just above high threshold
    action, _ = decide_adjustment(auc_score=0.751)
    assert action == 'INCREASE'
    
    # Just below low threshold
    action, _ = decide_adjustment(auc_score=0.549)
    assert action == 'DECREASE'
```

---

### 3.2 Parameter Adjustment

#### Test 3.2.1: INCREASE multiplies by 1.2

```python
def test_increase_multiplier():
    """INCREASE should multiply adjustable params by 1.2."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    policy.adjust_stochasticity(auc_score=0.82, direction='increase')
    
    assert policy.params['timing_range_hours'] == 2.4  # 2.0 × 1.2
    assert policy.params['threshold_pct'] == 0.12      # 0.10 × 1.2
```

---

#### Test 3.2.2: DECREASE multiplies by 0.8

```python
def test_decrease_multiplier():
    """DECREASE should multiply adjustable params by 0.8."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    policy.adjust_stochasticity(auc_score=0.52, direction='decrease')
    
    assert policy.params['timing_range_hours'] == 1.6  # 2.0 × 0.8
    assert policy.params['threshold_pct'] == 0.08      # 0.10 × 0.8
```

---

#### Test 3.2.3: Multiple adjustments compound

```python
def test_multiple_adjustments_compound():
    """Sequential adjustments should compound multiplicatively."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    # First increase: 2.0 × 1.2 = 2.4
    policy.adjust_stochasticity(0.82, 'increase')
    assert abs(policy.params['timing_range_hours'] - 2.4) < 0.01
    
    # Second increase: 2.4 × 1.2 = 2.88
    policy.adjust_stochasticity(0.85, 'increase')
    assert abs(policy.params['timing_range_hours'] - 2.88) < 0.01
    
    # Decrease: 2.88 × 0.8 = 2.304
    policy.adjust_stochasticity(0.52, 'decrease')
    assert abs(policy.params['timing_range_hours'] - 2.304) < 0.01
```

---

### 3.3 Boundary Enforcement

#### Test 3.3.1: Min boundary clips

```python
def test_min_boundary_clips():
    """Parameters should not go below minimum bounds."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 0.6})
    
    # Decrease: 0.6 × 0.8 = 0.48, should clip to 0.5
    policy.adjust_stochasticity(0.52, 'decrease')
    
    assert policy.params['timing_range_hours'] == 0.5
```

---

#### Test 3.3.2: Max boundary clips

```python
def test_max_boundary_clips():
    """Parameters should not go above maximum bounds."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 5.5})
    
    # Increase: 5.5 × 1.2 = 6.6, should clip to 6.0
    policy.adjust_stochasticity(0.82, 'increase')
    
    assert policy.params['timing_range_hours'] == 6.0
```

---

#### Test 3.3.3: Both params respect bounds independently

```python
def test_independent_boundary_enforcement():
    """Each parameter should enforce its own bounds independently."""
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 5.8,  # Near max (6.0)
        'threshold_pct': 0.06       # Near min (0.05)
    })
    
    # Increase
    policy.adjust_stochasticity(0.82, 'increase')
    
    # timing_range: 5.8 × 1.2 = 6.96 → clip to 6.0
    assert policy.params['timing_range_hours'] == 6.0
    
    # threshold_pct: 0.06 × 1.2 = 0.072 → no clipping
    assert abs(policy.params['threshold_pct'] - 0.072) < 0.001
    
    # Decrease
    policy.adjust_stochasticity(0.52, 'decrease')
    
    # timing_range: 6.0 × 0.8 = 4.8 → no clipping
    assert abs(policy.params['timing_range_hours'] - 4.8) < 0.01
    
    # threshold_pct: 0.072 × 0.8 = 0.0576 → no clipping
    assert abs(policy.params['threshold_pct'] - 0.0576) < 0.001
```

---

### 3.4 Convergence Detection

#### Test 3.4.1: Not converged initially

```python
def test_not_converged_initially():
    """Should not be converged with insufficient history."""
    from adaptive_adversary import check_convergence
    
    auc_history = [0.65, 0.68]  # Only 2 iterations
    
    converged, conv_iter = check_convergence(auc_history, patience=5)
    
    assert converged == False
    assert conv_iter is None
```

---

#### Test 3.4.2: Converged after 5 consecutive in range

```python
def test_converged_after_patience():
    """Should detect convergence after 5 consecutive iterations in target."""
    from adaptive_adversary import check_convergence
    
    auc_history = [
        0.75,  # Outside
        0.72,  # Outside
        0.65,  # In range (start counting)
        0.68,  # In range (2)
        0.62,  # In range (3)
        0.67,  # In range (4)
        0.63   # In range (5) → converged!
    ]
    
    converged, conv_iter = check_convergence(auc_history, patience=5)
    
    assert converged == True
    assert conv_iter == 3  # Started counting at iteration 2 (0-indexed)
```

---

#### Test 3.4.3: Not converged if leaves range

```python
def test_not_converged_if_leaves_range():
    """Should reset counter if AUC leaves target range."""
    from adaptive_adversary import check_convergence
    
    auc_history = [
        0.65,  # In range (1)
        0.68,  # In range (2)
        0.62,  # In range (3)
        0.78,  # OUT OF RANGE → reset counter
        0.66,  # In range (1)
        0.64   # In range (2) → not yet 5
    ]
    
    converged, conv_iter = check_convergence(auc_history, patience=5)
    
    assert converged == False
```

---

### 3.5 Oscillation Detection

#### Test 3.5.1: Detect INCREASE-DECREASE-INCREASE pattern

```python
def test_detect_increase_decrease_increase():
    """Should detect oscillation in adjustment actions."""
    from adaptive_adversary import detect_oscillation
    
    adjustment_history = [
        {'action': 'HOLD'},
        {'action': 'INCREASE'},
        {'action': 'DECREASE'},
        {'action': 'INCREASE'}  # Oscillation!
    ]
    
    is_oscillating = detect_oscillation(adjustment_history, window=3)
    
    assert is_oscillating == True
```

---

#### Test 3.5.2: Detect DECREASE-INCREASE-DECREASE pattern

```python
def test_detect_decrease_increase_decrease():
    """Should detect reverse oscillation pattern."""
    from adaptive_adversary import detect_oscillation
    
    adjustment_history = [
        {'action': 'DECREASE'},
        {'action': 'INCREASE'},
        {'action': 'DECREASE'}  # Oscillation!
    ]
    
    is_oscillating = detect_oscillation(adjustment_history, window=3)
    
    assert is_oscillating == True
```

---

#### Test 3.5.3: No oscillation with consistent direction

```python
def test_no_oscillation_consistent_direction():
    """Should not detect oscillation with consistent adjustments."""
    from adaptive_adversary import detect_oscillation
    
    adjustment_history = [
        {'action': 'INCREASE'},
        {'action': 'INCREASE'},
        {'action': 'INCREASE'}
    ]
    
    is_oscillating = detect_oscillation(adjustment_history, window=3)
    
    assert is_oscillating == False
```

---

#### Test 3.5.4: No oscillation with HOLD actions

```python
def test_no_oscillation_with_holds():
    """Should not detect oscillation when HOLD actions present."""
    from adaptive_adversary import detect_oscillation
    
    adjustment_history = [
        {'action': 'INCREASE'},
        {'action': 'HOLD'},
        {'action': 'DECREASE'}
    ]
    
    is_oscillating = detect_oscillation(adjustment_history, window=3)
    
    assert is_oscillating == False
```

---

### 3.6 Adjustment Record Structure

#### Test 3.6.1: Record contains required fields

```python
def test_adjustment_record_structure():
    """Adjustment record should contain all required fields."""
    from adaptive_adversary import adaptive_step
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    record = adaptive_step(
        auc_score=0.82,
        policy=policy,
        iteration=5,
        adjustment_history=[]
    )
    
    # Check required fields exist
    assert 'iteration' in record
    assert 'auc' in record
    assert 'action' in record
    assert 'rationale' in record
    assert 'params_before' in record
    assert 'params_after' in record
    
    # Check values are correct types
    assert isinstance(record['iteration'], int)
    assert isinstance(record['auc'], float)
    assert isinstance(record['action'], str)
    assert isinstance(record['rationale'], str)
    assert isinstance(record['params_before'], dict)
    assert isinstance(record['params_after'], dict)
```

---

#### Test 3.6.2: Params before and after captured correctly

```python
def test_params_captured_correctly():
    """Should capture parameter values before and after adjustment."""
    from adaptive_adversary import adaptive_step
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    record = adaptive_step(
        auc_score=0.82,
        policy=policy,
        iteration=0,
        adjustment_history=[]
    )
    
    # Check params_before
    assert record['params_before']['timing_range_hours'] == 2.0
    
    # Check params_after (should be increased by 1.2)
    assert record['params_after']['timing_range_hours'] == 2.4
    
    # Verify they're different
    assert record['params_before'] != record['params_after']
```

---

## 4. Integration Tests

### 4.1 Adaptive Step with P4

#### Test 4.1.1: Full adaptive step workflow

```python
def test_full_adaptive_step():
    """Test complete adaptive step with real P4 policy."""
    from adaptive_adversary import adaptive_step
    from bsml.randomization import UniformPolicy
    
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    adjustment_history = []
    
    # Step 1: High AUC triggers increase
    record = adaptive_step(0.82, policy, 0, adjustment_history)
    
    assert record['action'] == 'INCREASE'
    assert policy.params['timing_range_hours'] == 2.4
    assert policy.params['threshold_pct'] == 0.12
    
    adjustment_history.append(record)
    
    # Step 2: Still high, increase again
    record = adaptive_step(0.78, policy, 1, adjustment_history)
    
    assert record['action'] == 'INCREASE'
    assert abs(policy.params['timing_range_hours'] - 2.88) < 0.01
    
    adjustment_history.append(record)
    
    # Step 3: Now in target, hold
    record = adaptive_step(0.65, policy, 2, adjustment_history)
    
    assert record['action'] == 'HOLD'
    assert abs(policy.params['timing_range_hours'] - 2.88) < 0.01  # Unchanged
```

---

### 4.2 Mock Training Loop

#### Test 4.2.1: Simple training loop with mock AUC

```python
def test_mock_training_loop():
    """Test adaptive loop with predefined mock AUC scores."""
    from adaptive_adversary import adaptive_adversary_training_loop
    from bsml.randomization import UniformPolicy
    
    # Mock AUC sequence: high → high → target → target → target (converge)
    mock_auc_sequence = [0.82, 0.78, 0.65, 0.68, 0.63, 0.67, 0.66]
    
    def mock_train_and_evaluate(policy, iteration):
        """Return mock AUC from sequence."""
        return mock_auc_sequence[iteration]
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    results = adaptive_adversary_training_loop(
        policy_initial=policy,
        n_iterations=len(mock_auc_sequence),
        mock_auc_fn=mock_train_and_evaluate
    )
    
    # Should converge after iteration 6 (5 consecutive in target)
    assert results['converged_iteration'] is not None
    assert len(results['auc_history']) == len(mock_auc_sequence)
    
    # First two iterations should INCREASE
    assert results['adjustment_log'][0]['action'] == 'INCREASE'
    assert results['adjustment_log'][1]['action'] == 'INCREASE'
    
    # Remaining should HOLD (in target range)
    for i in range(2, len(mock_auc_sequence)):
        assert results['adjustment_log'][i]['action'] == 'HOLD'
```

---

#### Test 4.2.2: Training loop with oscillation

```python
def test_training_loop_with_oscillation():
    """Test that oscillation is detected during training."""
    from adaptive_adversary import adaptive_adversary_training_loop
    from bsml.randomization import UniformPolicy
    
    # Mock AUC: oscillate between high and low
    mock_auc_sequence = [0.82, 0.52, 0.82, 0.52, 0.82]
    
    def mock_train_and_evaluate(policy, iteration):
        return mock_auc_sequence[iteration]
    
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    results = adaptive_adversary_training_loop(
        policy_initial=policy,
        n_iterations=len(mock_auc_sequence),
        mock_auc_fn=mock_train_and_evaluate
    )
    
    # Check oscillation warning appears
    oscillation_detected = any(
        rec.get('oscillation_warning', False)
        for rec in results['adjustment_log']
    )
    
    assert oscillation_detected == True
```

---

#### Test 4.2.3: Training loop hits boundary

```python
def test_training_loop_hits_boundary():
    """Test training when parameter hits max boundary."""
    from adaptive_adversary import adaptive_adversary_training_loop
    from bsml.randomization import UniformPolicy
    
    # Start near max, all high AUC → will hit boundary
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 5.5})
    
    mock_auc_sequence = [0.85, 0.88, 0.90, 0.92]  # All very high
    
    def mock_train_and_evaluate(policy, iteration):
        return mock_auc_sequence[iteration]
    
    results = adaptive_adversary_training_loop(
        policy_initial=policy,
        n_iterations=len(mock_auc_sequence),
        mock_auc_fn=mock_train_and_evaluate
    )
    
    # Final params should be at max
    assert results['final_policy_params']['timing_range_hours'] == 6.0
    
    # Should have boundary hit recorded
    boundary_hits = [
        rec.get('boundary_hit', False)
        for rec in results['adjustment_log']
    ]
    assert any(boundary_hits)
```

---

## 5. End-to-End Skeleton Test

### Test 5.1: Complete v0.1 workflow on toy data

```python
def test_end_to_end_v01_skeleton():
    """
    Test complete v0.1 skeleton workflow:
    1. Initialize policy
    2. Run adaptive loop with mock adversary
    3. Check convergence
    4. Verify results structure
    """
    from adaptive_adversary import adaptive_adversary_training_loop
    from bsml.randomization import UniformPolicy
    import numpy as np
    
    # Mock adversary that gradually improves with more randomness
    def mock_adversary(policy, iteration):
        """
        Simulate adversary AUC based on policy stochasticity.
        More stochasticity → lower AUC (less predictable).
        """
        timing_range = policy.params['timing_range_hours']
        
        # AUC inversely related to timing_range
        # Base AUC starts at 0.85, decreases as range increases
        base_auc = 0.85
        reduction = (timing_range - 2.0) * 0.05  # 0.05 per hour increase
        auc = base_auc - reduction
        
        # Add small random noise
        auc += np.random.normal(0, 0.02)
        
        # Clip to valid range
        return max(0.5, min(1.0, auc))
    
    # Initialize
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    # Run adaptive loop
    results = adaptive_adversary_training_loop(
        policy_initial=policy,
        n_iterations=20,
        mock_auc_fn=mock_adversary,
        early_stop_patience=5
    )
    
    # Assertions
    
    # 1. Should produce adjustment history
    assert len(results['adjustment_log']) > 0
    
    # 2. Should have AUC trajectory
    assert len(results['auc_history']) > 0
    
    # 3. Should eventually converge (with our mock, AUC decreases)
    assert results['converged_iteration'] is not None
    
    # 4. Final params should be different from initial
    assert results['final_policy_params'] != {'timing_range_hours': 2.0, 'threshold_pct': 0.10}
    
    # 5. Check results structure
    assert 'policy_name' in results
    assert 'auc_history' in results
    assert 'adjustment_log' in results
    assert 'final_policy_params' in results
    assert 'converged_iteration' in results
    assert 'n_iterations_run' in results
    
    # 6. AUC should decrease over time (due to increasing randomness)
    first_auc = results['auc_history'][0]
    last_auc = results['auc_history'][-1]
    assert last_auc < first_auc  # Became less predictable
    
    print("✓ End-to-end v0.1 skeleton test passed!")
    print(f"  Converged at iteration: {results['converged_iteration']}")
    print(f"  AUC trajectory: {first_auc:.3f} → {last_auc:.3f}")
    print(f"  Final params: {results['final_policy_params']}")
```

---

## 6. Test Fixtures

### 6.1 Mock Policy

```python
# tests/fixtures.py

class MockPolicy:
    """Mock policy for testing without P4 dependency."""
    
    def __init__(self, seed, params):
        self.seed = seed
        self.params = params.copy()
        self._log = []
    
    def adjust_stochasticity(self, auc_score, direction):
        """Mock adjustment that just multiplies params."""
        multiplier = 1.2 if direction == 'increase' else 0.8
        
        for key in self.params:
            self.params[key] *= multiplier
        
        self._log.append({
            'auc': auc_score,
            'direction': direction,
            'params': self.params.copy()
        })
    
    def check_exposure_invariance(self, pos_before, pos_after):
        """Mock exposure check (always returns True for testing)."""
        return True
```

### 6.2 Mock Adversary

```python
# tests/fixtures.py

def mock_adversary_deterministic(policy, iteration):
    """
    Deterministic mock adversary for reproducible tests.
    
    Returns AUC based on policy params in predictable way.
    """
    timing_range = policy.params.get('timing_range_hours', 2.0)
    
    # Simple inverse relationship
    auc = 0.90 - (timing_range * 0.05)
    
    return max(0.5, min(1.0, auc))


def mock_adversary_random(policy, iteration, seed=42):
    """
    Stochastic mock adversary with controlled randomness.
    """
    np.random.seed(seed + iteration)
    
    timing_range = policy.params.get('timing_range_hours', 2.0)
    base_auc = 0.85 - (timing_range - 2.0) * 0.05
    noise = np.random.normal(0, 0.03)
    
    auc = base_auc + noise
    
    return max(0.5, min(1.0, auc))
```

### 6.3 Mock Data Generator

```python
# tests/fixtures.py

def generate_mock_trades(n=1000, seed=42):
    """Generate mock trade DataFrame for testing."""
    np.random.seed(seed)
    
    trades = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n),
        'side': np.random.choice(['BUY', 'SELL'], n),
        'qty': np.random.randint(10, 1000, n),
        'ref_price': np.random.uniform(100, 200, n),
        'exec_price': np.random.uniform(100, 200, n)
    })
    
    return trades
```

---

## 7. Acceptance Criteria

For v0.1 to be considered complete and ready for Week 3:

### 7.1 Code Quality

- [ ] All unit tests pass (Section 3)
- [ ] All integration tests pass (Section 4)
- [ ] End-to-end skeleton test passes (Section 5)
- [ ] Code coverage > 80% for adaptive_adversary.py
- [ ] No linting errors (flake8, black)
- [ ] Type hints added to all public functions

### 7.2 Functionality

- [ ] Adjustment logic correctly implements all 5 actions (INCREASE, DECREASE, HOLD, NUDGE_UP, NUDGE_DOWN)
- [ ] P4 integration works (mock or real UniformPolicy)
- [ ] Convergence detection works correctly
- [ ] Oscillation detection works correctly
- [ ] Boundary enforcement prevents out-of-bounds params
- [ ] Logging captures all required fields

### 7.3 Documentation

- [ ] Docstrings for all functions
- [ ] Type hints in function signatures
- [ ] README.md with usage examples
- [ ] This test plan executed and results documented

### 7.4 Integration Readiness

- [ ] API compatible with P6's adversary training (can swap in real adversary in v1.0)
- [ ] API compatible with P3's backtest runner (can swap in real trades in v1.0)
- [ ] Logging format matches P3's schema expectations
- [ ] Output structure matches paper requirements

---

## 8. Test Execution Plan

### Week 2 Schedule

**Monday-Tuesday (Nov 10-11):**
- Implement `adaptive_adversary_v0.1.py` skeleton
- Write unit tests (Section 3.1-3.6)
- Aim for 100% unit test pass rate

**Wednesday (Nov 12):**
- Implement integration with P4 (real UniformPolicy)
- Write integration tests (Section 4.1-4.2)
- Fix any P4 API issues discovered

**Thursday (Nov 13):**
- Implement end-to-end skeleton test (Section 5)
- Test with mock adversary and mock data
- Refine convergence and oscillation logic

**Friday (Nov 14):**
- Code review with P4 (Neel) and P6
- Address feedback
- Finalize v0.1 for Week 3 integration
- Document any API changes needed

### Test Environment

```bash
# Setup test environment
python -m pytest tests/test_adaptive_v01.py -v --cov=adaptive_adversary

# Run specific test category
pytest tests/test_adaptive_v01.py::test_adjustment_decision_logic -v

# Run with coverage report
pytest tests/test_adaptive_v01.py --cov=adaptive_adversary --cov-report=html
```

---

## 9. Success Metrics

**v0.1 is successful if:**

1. **All tests pass:** 100% of tests in this plan execute successfully
2. **No P4 integration issues:** Can call `adjust_stochasticity()` without errors
3. **Mock adversary works:** Can run 20 iterations with deterministic mock AUC
4. **Convergence detected:** Successfully detects when AUC stable in target range
5. **Ready for v1.0:** Can swap in real P6 adversary and real P3 backtests with minimal changes

**Stretch goals for v0.1:**
- Test with real UniformPolicy on real toy data (not required, but nice to have)
- Add visualization of AUC trajectory in test output
- Benchmark performance (can run 20 iterations in < 1 minute on mock data)

---

## 10. Test Data Requirements

### 10.1 Minimal Test Data

For v0.1, only need:

- Mock AUC sequences (predefined arrays)
- Mock policy params (simple dicts)
- No real price data required
- No real trade execution required

### 10.2 Optional Enhanced Testing

If time permits, can test with:

- Toy price data from P3 (toy_prices_baseline.csv)
- Small backtest (10 symbols, 1 month)
- Real UniformPolicy with bounded params

**Decision:** Keep v0.1 pure unit/integration tests with mocks. Real data in v1.0.

---

## 11. Known Limitations of v0.1

What v0.1 **does not** test (deferred to v1.0):

- ❌ Real adversary training (P6 integration)
- ❌ Real backtest execution (P3 integration)
- ❌ Real metrics computation (P5 integration)
- ❌ OU and Pink policies (only Uniform in v0.1)
- ❌ Multiple seeds (only single seed)
- ❌ Regime splits (only single run)
- ❌ Parallel execution (only sequential)
- ❌ Exposure invariance with real trades

**Rationale:** v0.1 is a skeleton to validate adaptive logic. Full integration happens Week 3+.

---

**END OF TEST PLAN**

**Status:** Week 1 Deliverable  
**Owner:** P7  
**Date:** November 14, 2025  
**Next:** Execute this plan during Week 2 implementation
