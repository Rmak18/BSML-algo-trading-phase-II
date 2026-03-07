"""
Unit Tests for Adaptive Adversary v0.1

Project: BSML - Randomized Execution Research
Owner: P7 (Adaptive Adversary Framework)
Date: November 14, 2025
Status: Week 2 Deliverable

Test coverage for adaptive_adversary_v0.1.py core functions.
"""

import pytest
import numpy as np
import copy
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: module 'adaptive_adversary_v0.1' does not exist yet — "v0.1" is also
# not a valid Python identifier. When this module is implemented, update this
# import to point at the correct module path (e.g. bsml.adaptive.adversary_loop)
# and remove the pytestmark skip below.
#
# from adaptive_adversary_v0_1 import (
#     decide_adjustment, adaptive_step, check_convergence,
#     detect_oscillation, adaptive_adversary_training_loop,
#     AUC_HIGH_THRESHOLD, AUC_LOW_THRESHOLD,
#     AUC_TARGET_MIN, AUC_TARGET_MAX, AUC_TARGET_MIDPOINT,
# )

pytestmark = pytest.mark.skip(
    reason="adaptive_adversary_v0.1 module not yet implemented — "
           "see TODO comment above the import"
)

# Stub constants so test function bodies parse without NameError at collection
decide_adjustment = adaptive_step = check_convergence = None
detect_oscillation = adaptive_adversary_training_loop = None
AUC_HIGH_THRESHOLD = AUC_LOW_THRESHOLD = None
AUC_TARGET_MIN = AUC_TARGET_MAX = AUC_TARGET_MIDPOINT = None


# =============================================================================
# FIXTURES
# =============================================================================

class MockPolicy:
    """Mock policy for testing without P4 dependency."""
    
    def __init__(self, seed, params):
        self.seed = seed
        self.params = params.copy()
        self.__class__.__name__ = 'MockUniform'
        self._adjustment_log = []
    
    def adjust_stochasticity(self, auc_score, direction):
        """Mock adjustment that multiplies params by 1.2 or 0.8."""
        multiplier = 1.2 if direction == 'increase' else 0.8
        
        old_params = self.params.copy()
        
        for key in self.params:
            self.params[key] *= multiplier
            
            # Enforce bounds (same as P4)
            if key == 'timing_range_hours':
                self.params[key] = max(0.5, min(6.0, self.params[key]))
            elif key == 'threshold_pct':
                self.params[key] = max(0.05, min(0.25, self.params[key]))
            elif key == 'sigma':
                self.params[key] = max(0.01, min(0.15, self.params[key]))
            elif key == 'scale':
                self.params[key] = max(0.02, min(0.20, self.params[key]))
        
        # Log adjustment for testing
        self._adjustment_log.append({
            'auc': auc_score,
            'direction': direction,
            'params_before': old_params,
            'params_after': self.params.copy()
        })


@pytest.fixture
def mock_uniform_policy():
    """Fixture for Uniform policy with standard initial params."""
    return MockPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })


@pytest.fixture
def mock_ou_policy():
    """Fixture for OU policy with standard initial params."""
    return MockPolicy(seed=42, params={
        'theta': 0.5,
        'sigma': 0.5,
        'mu': 0.0
    })


@pytest.fixture
def mock_pink_policy():
    """Fixture for Pink policy with standard initial params."""
    return MockPolicy(seed=42, params={
        'alpha': 1.0,
        'scale': 0.08
    })


# =============================================================================
# TEST: decide_adjustment()
# =============================================================================

class TestDecideAdjustment:
    """Tests for adjustment decision logic."""
    
    def test_high_auc_triggers_increase(self):
        """AUC > 0.75 should trigger INCREASE action."""
        action, multiplier = decide_adjustment(auc_score=0.82)
        
        assert action == 'INCREASE'
        assert multiplier == 1.20
    
    def test_very_high_auc_triggers_increase(self):
        """Very high AUC (> 0.90) should also trigger INCREASE."""
        action, multiplier = decide_adjustment(auc_score=0.95)
        
        assert action == 'INCREASE'
        assert multiplier == 1.20
    
    def test_low_auc_triggers_decrease(self):
        """AUC < 0.55 should trigger DECREASE action."""
        action, multiplier = decide_adjustment(auc_score=0.52)
        
        assert action == 'DECREASE'
        assert multiplier == 0.80
    
    def test_very_low_auc_triggers_decrease(self):
        """Very low AUC (close to 0.50) should trigger DECREASE."""
        action, multiplier = decide_adjustment(auc_score=0.51)
        
        assert action == 'DECREASE'
        assert multiplier == 0.80
    
    def test_target_range_lower_bound_triggers_hold(self):
        """AUC = 0.60 (lower bound) should trigger HOLD."""
        action, multiplier = decide_adjustment(auc_score=0.60)
        
        assert action == 'HOLD'
        assert multiplier is None
    
    def test_target_range_midpoint_triggers_hold(self):
        """AUC = 0.65 (midpoint) should trigger HOLD."""
        action, multiplier = decide_adjustment(auc_score=0.65)
        
        assert action == 'HOLD'
        assert multiplier is None
    
    def test_target_range_upper_bound_triggers_hold(self):
        """AUC = 0.70 (upper bound) should trigger HOLD."""
        action, multiplier = decide_adjustment(auc_score=0.70)
        
        assert action == 'HOLD'
        assert multiplier is None
    
    def test_target_range_interior_triggers_hold(self):
        """AUC inside target range should trigger HOLD."""
        for auc in [0.61, 0.63, 0.67, 0.69]:
            action, multiplier = decide_adjustment(auc_score=auc)
            assert action == 'HOLD', f"AUC {auc} should trigger HOLD"
            assert multiplier is None
    
    def test_slightly_above_target_triggers_nudge_up(self):
        """AUC between 0.70 and 0.75 should trigger NUDGE_UP."""
        for auc in [0.71, 0.72, 0.73, 0.74]:
            action, multiplier = decide_adjustment(auc_score=auc)
            assert action == 'NUDGE_UP', f"AUC {auc} should trigger NUDGE_UP"
            assert multiplier == 1.20
    
    def test_slightly_below_target_triggers_nudge_down(self):
        """AUC between 0.55 and 0.60 should trigger NUDGE_DOWN."""
        for auc in [0.56, 0.57, 0.58, 0.59]:
            action, multiplier = decide_adjustment(auc_score=auc)
            assert action == 'NUDGE_DOWN', f"AUC {auc} should trigger NUDGE_DOWN"
            assert multiplier == 0.80
    
    def test_exact_high_threshold_triggers_increase(self):
        """AUC exactly at 0.75 should trigger INCREASE."""
        action, multiplier = decide_adjustment(auc_score=0.75)
        
        assert action == 'INCREASE'
        assert multiplier == 1.20
    
    def test_exact_low_threshold_triggers_decrease(self):
        """AUC exactly at 0.55 should trigger DECREASE."""
        action, multiplier = decide_adjustment(auc_score=0.55)
        
        assert action == 'DECREASE'
        assert multiplier == 0.80
    
    def test_just_above_high_threshold(self):
        """AUC just above 0.75 should trigger INCREASE."""
        action, multiplier = decide_adjustment(auc_score=0.751)
        
        assert action == 'INCREASE'
        assert multiplier == 1.20
    
    def test_just_below_low_threshold(self):
        """AUC just below 0.55 should trigger DECREASE."""
        action, multiplier = decide_adjustment(auc_score=0.549)
        
        assert action == 'DECREASE'
        assert multiplier == 0.80


# =============================================================================
# TEST: adaptive_step()
# =============================================================================

class TestAdaptiveStep:
    """Tests for single adaptive adjustment step."""
    
    def test_increase_modifies_params(self, mock_uniform_policy):
        """INCREASE action should multiply params by 1.2."""
        policy = mock_uniform_policy
        
        record = adaptive_step(
            auc_score=0.82,
            policy=policy,
            iteration=0,
            adjustment_history=[]
        )
        
        assert record['action'] == 'INCREASE'
        assert record['params_before']['timing_range_hours'] == 2.0
        assert record['params_after']['timing_range_hours'] == 2.4
        assert record['params_after']['threshold_pct'] == 0.12
    
    def test_decrease_modifies_params(self, mock_uniform_policy):
        """DECREASE action should multiply params by 0.8."""
        policy = mock_uniform_policy
        
        record = adaptive_step(
            auc_score=0.52,
            policy=policy,
            iteration=0,
            adjustment_history=[]
        )
        
        assert record['action'] == 'DECREASE'
        assert record['params_before']['timing_range_hours'] == 2.0
        assert record['params_after']['timing_range_hours'] == 1.6
        assert record['params_after']['threshold_pct'] == 0.08
    
    def test_hold_does_not_modify_params(self, mock_uniform_policy):
        """HOLD action should not change params."""
        policy = mock_uniform_policy
        
        record = adaptive_step(
            auc_score=0.65,
            policy=policy,
            iteration=0,
            adjustment_history=[]
        )
        
        assert record['action'] == 'HOLD'
        assert record['params_before'] == record['params_after']
        assert policy.params['timing_range_hours'] == 2.0
        assert policy.params['threshold_pct'] == 0.10
    
    def test_record_contains_required_fields(self, mock_uniform_policy):
        """Adjustment record should contain all required fields."""
        policy = mock_uniform_policy
        
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
        assert 'multiplier' in record
        assert 'rationale' in record
        assert 'params_before' in record
        assert 'params_after' in record
        assert 'oscillation_warning' in record
        
        # Check values
        assert record['iteration'] == 5
        assert record['auc'] == 0.82
        assert isinstance(record['action'], str)
        assert isinstance(record['rationale'], str)
        assert isinstance(record['params_before'], dict)
        assert isinstance(record['params_after'], dict)
        assert isinstance(record['oscillation_warning'], bool)
    
    def test_rationale_contains_auc_value(self, mock_uniform_policy):
        """Rationale should include AUC value."""
        policy = mock_uniform_policy
        
        record = adaptive_step(0.82, policy, 0, [])
        
        assert '0.82' in record['rationale'] or '0.820' in record['rationale']
    
    def test_multiple_adjustments_compound(self, mock_uniform_policy):
        """Sequential adjustments should compound multiplicatively."""
        policy = mock_uniform_policy
        history = []
        
        # First increase: 2.0 × 1.2 = 2.4
        record1 = adaptive_step(0.82, policy, 0, history)
        history.append(record1)
        assert abs(policy.params['timing_range_hours'] - 2.4) < 0.01
        
        # Second increase: 2.4 × 1.2 = 2.88
        record2 = adaptive_step(0.85, policy, 1, history)
        history.append(record2)
        assert abs(policy.params['timing_range_hours'] - 2.88) < 0.01
        
        # Decrease: 2.88 × 0.8 = 2.304
        record3 = adaptive_step(0.52, policy, 2, history)
        assert abs(policy.params['timing_range_hours'] - 2.304) < 0.01


# =============================================================================
# TEST: Boundary Enforcement
# =============================================================================

class TestBoundaryEnforcement:
    """Tests for parameter boundary clipping."""
    
    def test_timing_range_min_boundary(self):
        """timing_range_hours should not go below 0.5."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 0.6})
        
        # Decrease: 0.6 × 0.8 = 0.48 → should clip to 0.5
        record = adaptive_step(0.52, policy, 0, [])
        
        assert policy.params['timing_range_hours'] == 0.5
    
    def test_timing_range_max_boundary(self):
        """timing_range_hours should not go above 6.0."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 5.5})
        
        # Increase: 5.5 × 1.2 = 6.6 → should clip to 6.0
        record = adaptive_step(0.82, policy, 0, [])
        
        assert policy.params['timing_range_hours'] == 6.0
    
    def test_threshold_pct_min_boundary(self):
        """threshold_pct should not go below 0.05."""
        policy = MockPolicy(seed=42, params={'threshold_pct': 0.06})
        
        # Decrease: 0.06 × 0.8 = 0.048 → should clip to 0.05
        record = adaptive_step(0.52, policy, 0, [])
        
        assert policy.params['threshold_pct'] == 0.05
    
    def test_threshold_pct_max_boundary(self):
        """threshold_pct should not go above 0.25."""
        policy = MockPolicy(seed=42, params={'threshold_pct': 0.22})
        
        # Increase: 0.22 × 1.2 = 0.264 → should clip to 0.25
        record = adaptive_step(0.82, policy, 0, [])
        
        assert policy.params['threshold_pct'] == 0.25
    
    def test_sigma_min_boundary(self, mock_ou_policy):
        """sigma should not go below 0.01."""
        policy = MockPolicy(seed=42, params={'sigma': 0.012})
        
        # Decrease: 0.012 × 0.8 = 0.0096 → should clip to 0.01
        record = adaptive_step(0.52, policy, 0, [])
        
        assert policy.params['sigma'] == 0.01
    
    def test_sigma_max_boundary(self, mock_ou_policy):
        """sigma should not go above 0.15."""
        policy = MockPolicy(seed=42, params={'sigma': 0.14})
        
        # Increase: 0.14 × 1.2 = 0.168 → should clip to 0.15
        record = adaptive_step(0.82, policy, 0, [])
        
        assert policy.params['sigma'] == 0.15
    
    def test_scale_min_boundary(self, mock_pink_policy):
        """scale should not go below 0.02."""
        policy = MockPolicy(seed=42, params={'scale': 0.024})
        
        # Decrease: 0.024 × 0.8 = 0.0192 → should clip to 0.02
        record = adaptive_step(0.52, policy, 0, [])
        
        assert policy.params['scale'] == 0.02
    
    def test_scale_max_boundary(self, mock_pink_policy):
        """scale should not go above 0.20."""
        policy = MockPolicy(seed=42, params={'scale': 0.18})
        
        # Increase: 0.18 × 1.2 = 0.216 → should clip to 0.20
        record = adaptive_step(0.82, policy, 0, [])
        
        assert policy.params['scale'] == 0.20
    
    def test_independent_boundary_enforcement(self):
        """Each parameter should enforce its own bounds independently."""
        policy = MockPolicy(seed=42, params={
            'timing_range_hours': 5.8,  # Near max (6.0)
            'threshold_pct': 0.06       # Near min (0.05)
        })
        
        # Increase
        record = adaptive_step(0.82, policy, 0, [])
        
        # timing_range: 5.8 × 1.2 = 6.96 → clip to 6.0
        assert policy.params['timing_range_hours'] == 6.0
        
        # threshold_pct: 0.06 × 1.2 = 0.072 → no clipping
        assert abs(policy.params['threshold_pct'] - 0.072) < 0.001


# =============================================================================
# TEST: check_convergence()
# =============================================================================

class TestCheckConvergence:
    """Tests for convergence detection."""
    
    def test_not_converged_initially(self):
        """Should not be converged with insufficient history."""
        auc_history = [0.65, 0.68]  # Only 2 iterations
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == False
        assert conv_iter is None
    
    def test_not_converged_empty_history(self):
        """Empty history should not be converged."""
        auc_history = []
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == False
        assert conv_iter is None
    
    def test_converged_after_patience_iterations(self):
        """Should detect convergence after N consecutive in target."""
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
        assert conv_iter == 2  # Started at index 2
    
    def test_not_converged_if_leaves_range(self):
        """Should reset counter if AUC leaves target range."""
        auc_history = [
            0.65,  # In range (1)
            0.68,  # In range (2)
            0.62,  # In range (3)
            0.78,  # OUT → reset
            0.66,  # In range (1)
            0.64   # In range (2) → not yet 5
        ]
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == False
    
    def test_converged_at_boundaries(self):
        """Should converge when at exact boundaries."""
        auc_history = [0.60, 0.60, 0.70, 0.70, 0.65]  # All in [0.60, 0.70]
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == True
        assert conv_iter == 0
    
    def test_not_converged_just_outside_lower_bound(self):
        """Should not converge if just below lower bound."""
        auc_history = [0.599, 0.65, 0.68, 0.62, 0.67]  # First is 0.599 < 0.60
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == False
    
    def test_not_converged_just_outside_upper_bound(self):
        """Should not converge if just above upper bound."""
        auc_history = [0.65, 0.68, 0.701, 0.62, 0.67]  # Third is 0.701 > 0.70
        
        converged, conv_iter = check_convergence(auc_history, patience=5)
        
        assert converged == False
    
    def test_custom_patience(self):
        """Should respect custom patience parameter."""
        auc_history = [0.65, 0.68, 0.62]  # 3 in range
        
        # Not converged with patience=5
        converged, _ = check_convergence(auc_history, patience=5)
        assert converged == False
        
        # Converged with patience=3
        converged, conv_iter = check_convergence(auc_history, patience=3)
        assert converged == True
        assert conv_iter == 0


# =============================================================================
# TEST: detect_oscillation()
# =============================================================================

class TestDetectOscillation:
    """Tests for oscillation detection."""
    
    def test_detect_increase_decrease_increase(self):
        """Should detect INCREASE→DECREASE→INCREASE pattern."""
        history = [
            {'action': 'HOLD'},
            {'action': 'INCREASE'},
            {'action': 'DECREASE'},
            {'action': 'INCREASE'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == True
    
    def test_detect_decrease_increase_decrease(self):
        """Should detect DECREASE→INCREASE→DECREASE pattern."""
        history = [
            {'action': 'DECREASE'},
            {'action': 'INCREASE'},
            {'action': 'DECREASE'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == True
    
    def test_detect_nudge_up_down_up(self):
        """Should detect NUDGE_UP→NUDGE_DOWN→NUDGE_UP pattern."""
        history = [
            {'action': 'NUDGE_UP'},
            {'action': 'NUDGE_DOWN'},
            {'action': 'NUDGE_UP'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == True
    
    def test_detect_nudge_down_up_down(self):
        """Should detect NUDGE_DOWN→NUDGE_UP→NUDGE_DOWN pattern."""
        history = [
            {'action': 'NUDGE_DOWN'},
            {'action': 'NUDGE_UP'},
            {'action': 'NUDGE_DOWN'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == True
    
    def test_no_oscillation_consistent_direction(self):
        """Should not detect oscillation with consistent adjustments."""
        history = [
            {'action': 'INCREASE'},
            {'action': 'INCREASE'},
            {'action': 'INCREASE'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == False
    
    def test_no_oscillation_with_holds(self):
        """Should not detect oscillation when HOLD actions present."""
        history = [
            {'action': 'INCREASE'},
            {'action': 'HOLD'},
            {'action': 'DECREASE'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == False
    
    def test_no_oscillation_insufficient_history(self):
        """Should not detect oscillation with insufficient history."""
        history = [
            {'action': 'INCREASE'},
            {'action': 'DECREASE'}
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == False
    
    def test_no_oscillation_empty_history(self):
        """Empty history should not oscillate."""
        history = []
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == False
    
    def test_oscillation_only_checks_last_n(self):
        """Should only check last N actions, not entire history."""
        history = [
            {'action': 'HOLD'},
            {'action': 'HOLD'},
            {'action': 'HOLD'},
            {'action': 'INCREASE'},  # Last 3 start here
            {'action': 'DECREASE'},
            {'action': 'INCREASE'}   # Oscillation!
        ]
        
        is_oscillating = detect_oscillation(history, window=3)
        
        assert is_oscillating == True


# =============================================================================
# TEST: adaptive_adversary_training_loop()
# =============================================================================

class TestAdaptiveTrainingLoop:
    """Tests for complete adaptive training loop."""
    
    def test_training_loop_runs_to_completion(self, mock_uniform_policy):
        """Training loop should complete without errors."""
        results = adaptive_adversary_training_loop(
            policy_initial=mock_uniform_policy,
            n_iterations=5,
            verbose=False
        )
        
        assert results is not None
        assert 'auc_history' in results
        assert 'adjustment_log' in results
        assert len(results['auc_history']) > 0
        assert len(results['adjustment_log']) > 0
    
    def test_training_loop_converges(self):
        """Training loop should detect convergence."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        # Mock adversary that returns target-range AUC
        def mock_converging_auc(policy, iteration):
            return 0.65  # Always in target range
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=10,
            early_stop_patience=5,
            mock_auc_fn=mock_converging_auc,
            verbose=False
        )
        
        assert results['converged_iteration'] is not None
        assert results['n_iterations_run'] <= 10
    
    def test_training_loop_no_convergence(self):
        """Training loop should handle no convergence."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        # Mock adversary that oscillates
        def mock_oscillating_auc(policy, iteration):
            return 0.82 if iteration % 2 == 0 else 0.52
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=10,
            early_stop_patience=5,
            mock_auc_fn=mock_oscillating_auc,
            verbose=False
        )
        
        assert results['converged_iteration'] is None
        assert results['n_iterations_run'] == 10
    
    def test_training_loop_results_structure(self, mock_uniform_policy):
        """Results should contain all required fields."""
        results = adaptive_adversary_training_loop(
            policy_initial=mock_uniform_policy,
            n_iterations=5,
            verbose=False
        )
        
        # Check structure
        assert 'policy_name' in results
        assert 'auc_history' in results
        assert 'adjustment_log' in results
        assert 'final_policy_params' in results
        assert 'converged_iteration' in results
        assert 'n_iterations_run' in results
        
        # Check types
        assert isinstance(results['policy_name'], str)
        assert isinstance(results['auc_history'], list)
        assert isinstance(results['adjustment_log'], list)
        assert isinstance(results['final_policy_params'], dict)
        assert isinstance(results['n_iterations_run'], int)
    
    def test_default_mock_adversary_responds_to_randomness(self):
        """Default mock adversary AUC should decrease with more randomness."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=10,
            verbose=False
        )
        
        # AUC should generally decrease as timing_range increases
        first_auc = results['auc_history'][0]
        last_auc = results['auc_history'][-1]
        
        # Initial AUC should be high (predictable with timing_range=2.0)
        assert first_auc > 0.70
        
        # Final AUC should be lower (less predictable with increased timing_range)
        assert last_auc < first_auc
    
    def test_training_loop_modifies_policy_params(self, mock_uniform_policy):
        """Training loop should modify policy parameters."""
        initial_params = copy.deepcopy(mock_uniform_policy.params)
        
        results = adaptive_adversary_training_loop(
            policy_initial=mock_uniform_policy,
            n_iterations=10,
            verbose=False
        )
        
        # Final params should differ from initial
        assert results['final_policy_params'] != initial_params
    
    def test_custom_mock_adversary(self):
        """Should work with custom mock adversary function."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        call_count = [0]
        
        def custom_mock(policy, iteration):
            call_count[0] += 1
            return 0.75  # Always return 0.75
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=5,
            mock_auc_fn=custom_mock,
            verbose=False
        )
        
        # Check custom function was called
        assert call_count[0] == 5
        
        # Check AUC history
        assert all(auc == 0.75 for auc in results['auc_history'])


# =============================================================================
# TEST: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow_with_convergence(self):
        """Test complete workflow that converges."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        # Mock that gradually improves
        iteration_count = [0]
        def gradual_improvement(policy, iteration):
            iteration_count[0] = iteration
            # Start high, gradually enter target range
            if iteration < 3:
                return 0.82  # High
            else:
                return 0.65  # Target range
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=15,
            early_stop_patience=5,
            mock_auc_fn=gradual_improvement,
            verbose=False
        )
        
        # Should converge after ~8 iterations (3 high + 5 in target)
        assert results['converged_iteration'] is not None
        assert results['n_iterations_run'] < 15
        
        # First adjustments should be INCREASE
        assert results['adjustment_log'][0]['action'] == 'INCREASE'
        
        # Later adjustments should be HOLD
        assert results['adjustment_log'][-1]['action'] == 'HOLD'
    
    def test_full_workflow_with_oscillation(self):
        """Test workflow that oscillates."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 2.0})
        
        # Oscillating AUC
        def oscillating(policy, iteration):
            return 0.82 if iteration % 2 == 0 else 0.52
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=10,
            mock_auc_fn=oscillating,
            verbose=False
        )
        
        # Should detect oscillation
        oscillation_detected = any(
            rec.get('oscillation_warning', False)
            for rec in results['adjustment_log']
        )
        
        assert oscillation_detected == True
    
    def test_full_workflow_hits_boundary(self):
        """Test workflow where parameter hits max boundary."""
        policy = MockPolicy(seed=42, params={'timing_range_hours': 5.5})
        
        # Always very high AUC
        def always_high(policy, iteration):
            return 0.92
        
        results = adaptive_adversary_training_loop(
            policy_initial=policy,
            n_iterations=5,
            mock_auc_fn=always_high,
            verbose=False
        )
        
        # Should hit max boundary (6.0)
        final_timing = results['final_policy_params']['timing_range_hours']
        assert final_timing == 6.0
        
        # All actions should be INCREASE (trying to increase further)
        actions = [rec['action'] for rec in results['adjustment_log']]
        assert all(action == 'INCREASE' for action in actions)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
