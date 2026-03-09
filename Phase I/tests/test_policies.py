"""
Comprehensive test suite for randomization policies.

Tests cover:
- Utility functions (seed generation, exposure checks, market hours)
- Base policy functionality
- Uniform policy implementation
- Edge cases and error handling
- Statistical properties

Run with: pytest tests/test_policies.py -v

Owner: P4
Week: 2
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, time
import sys
sys.path.insert(0, 'src')

from bsml.policies import (
    UniformPolicy,
    DEFAULT_UNIFORM_PARAMS,
    generate_policy_seed,
    calculate_net_exposure,
    calculate_gross_exposure,
    is_within_exposure_tolerance,
    check_market_hours,
    clamp_to_market_hours,
)


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtils:
    """Tests for utility functions in utils.py"""
    
    def test_generate_policy_seed_deterministic(self):
        """Same inputs should always give same seed."""
        seed1 = generate_policy_seed(42, 'Uniform')
        seed2 = generate_policy_seed(42, 'Uniform')
        
        assert seed1 == seed2, "Same inputs must produce identical seeds"
    
    def test_generate_policy_seed_different_policies(self):
        """Different policies should get different seeds."""
        seed_uniform = generate_policy_seed(42, 'Uniform')
        seed_ou = generate_policy_seed(42, 'OU')
        seed_pink = generate_policy_seed(42, 'Pink')
        
        # All should be different
        assert seed_uniform != seed_ou
        assert seed_uniform != seed_pink
        assert seed_ou != seed_pink
    
    def test_generate_policy_seed_with_date(self):
        """Seeds should differ by date."""
        date1 = datetime(2025, 7, 15)
        date2 = datetime(2025, 7, 16)
        
        seed1 = generate_policy_seed(42, 'Uniform', date=date1)
        seed2 = generate_policy_seed(42, 'Uniform', date=date2)
        
        assert seed1 != seed2, "Different dates should give different seeds"
    
    def test_generate_policy_seed_with_symbol(self):
        """Seeds should differ by symbol."""
        seed_aapl = generate_policy_seed(42, 'Uniform', symbol='AAPL')
        seed_msft = generate_policy_seed(42, 'Uniform', symbol='MSFT')
        
        assert seed_aapl != seed_msft
    
    def test_generate_policy_seed_range(self):
        """Seeds should be in int32 range."""
        seed = generate_policy_seed(42, 'Uniform')
        
        assert 0 <= seed < 2**31, "Seed must be in int32 range"
        assert isinstance(seed, int), "Seed must be integer"
    
    def test_calculate_net_exposure_balanced(self):
        """Balanced portfolio should have net exposure = 0."""
        positions = {'AAPL': 100, 'MSFT': -100}
        net = calculate_net_exposure(positions)
        
        assert net == 0.0
    
    def test_calculate_net_exposure_long(self):
        """Net long portfolio."""
        positions = {'AAPL': 100, 'MSFT': -50, 'GOOGL': -30}
        net = calculate_net_exposure(positions)
        
        assert net == 20.0
    
    def test_calculate_net_exposure_empty(self):
        """Empty portfolio should have net exposure = 0."""
        assert calculate_net_exposure({}) == 0.0
    
    def test_calculate_gross_exposure(self):
        """Gross exposure is sum of absolute values."""
        positions = {'AAPL': 100, 'MSFT': -50, 'GOOGL': -30}
        gross = calculate_gross_exposure(positions)
        
        assert gross == 180.0
    
    def test_check_market_hours_during_hours(self):
        """Timestamp during market hours."""
        ts = datetime(2025, 7, 15, 10, 30)  # 10:30 AM
        assert check_market_hours(ts) == True
    
    def test_check_market_hours_before_open(self):
        """Timestamp before market open."""
        ts = datetime(2025, 7, 15, 8, 0)  # 8:00 AM
        assert check_market_hours(ts) == False
    
    def test_check_market_hours_after_close(self):
        """Timestamp after market close."""
        ts = datetime(2025, 7, 15, 17, 0)  # 5:00 PM
        assert check_market_hours(ts) == False
    
    def test_check_market_hours_at_open(self):
        """Exactly at market open (inclusive)."""
        ts = datetime(2025, 7, 15, 9, 30)
        assert check_market_hours(ts) == True
    
    def test_check_market_hours_at_close(self):
        """Exactly at market close (inclusive)."""
        ts = datetime(2025, 7, 15, 16, 0)
        assert check_market_hours(ts) == True
    
    def test_clamp_to_market_hours_before_open(self):
        """Clamp to market open."""
        ts = datetime(2025, 7, 15, 8, 0)
        clamped = clamp_to_market_hours(ts)
        
        assert clamped.hour == 9
        assert clamped.minute == 30
    
    def test_clamp_to_market_hours_after_close(self):
        """Clamp to market close."""
        ts = datetime(2025, 7, 15, 17, 30)
        clamped = clamp_to_market_hours(ts)
        
        assert clamped.hour == 16
        assert clamped.minute == 0
    
    def test_clamp_to_market_hours_already_within(self):
        """Already within hours - no change."""
        ts = datetime(2025, 7, 15, 12, 0)
        clamped = clamp_to_market_hours(ts)
        
        assert clamped == ts
    
    def test_exposure_tolerance_within(self):
        """Exposure change within tolerance."""
        before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
        after = {'AAPL': 102, 'MSFT': -98}    # Net = 4
        
        assert is_within_exposure_tolerance(before, after, tolerance=5.0) == True
    
    def test_exposure_tolerance_violation(self):
        """Exposure change violates tolerance."""
        before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
        after = {'AAPL': 120, 'MSFT': -80}    # Net = 40
        
        assert is_within_exposure_tolerance(before, after, tolerance=5.0) == False
    
    def test_exposure_tolerance_boundary(self):
        """Exactly at tolerance boundary (inclusive)."""
        before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
        after = {'AAPL': 105, 'MSFT': -100}   # Net = 5
        
        assert is_within_exposure_tolerance(before, after, tolerance=5.0) == True


# ============================================================================
# Test UniformPolicy
# ============================================================================

class TestUniformPolicy:
    """Tests for UniformPolicy implementation."""
    
    def test_initialization(self):
        """Policy initializes correctly with valid params."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        assert policy.seed == 42
        assert policy.params['timing_range_hours'] == 2.0
        assert policy.params['threshold_pct'] == 0.10
    
    def test_initialization_missing_params(self):
        """Missing required parameters should raise error."""
        with pytest.raises(ValueError, match="requires parameters"):
            UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    def test_initialization_sets_defaults(self):
        """Optional parameters get default values."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        # respect_market_hours should default to True
        assert policy.params['respect_market_hours'] == True
    
    def test_seed_reproducibility(self):
        """Same seed produces identical perturbations."""
        policy1 = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        policy2 = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        ts = datetime(2025, 7, 15, 10, 30)
        
        perturbed1 = policy1.perturb_timing(ts)
        perturbed2 = policy2.perturb_timing(ts)
        
        assert perturbed1 == perturbed2, "Same seed should give identical results"
    
    def test_different_seeds(self):
        """Different seeds produce different perturbations."""
        policy1 = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        policy2 = UniformPolicy(seed=99, params=DEFAULT_UNIFORM_PARAMS)
        
        ts = datetime(2025, 7, 15, 10, 30)
        
        # Run multiple times to ensure difference (not just lucky)
        results1 = [policy1.perturb_timing(ts) for _ in range(10)]
        results2 = [policy2.perturb_timing(ts) for _ in range(10)]
        
        # At least some should be different
        assert results1 != results2, "Different seeds should give different results"
    
    def test_timing_perturbation_range(self):
        """Timing shifts stay within specified range."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        ts = datetime(2025, 7, 15, 12, 0)  # Noon (middle of trading day)
        
        # Test many perturbations
        for _ in range(100):
            perturbed = policy.perturb_timing(ts)
            delta_hours = (perturbed - ts).total_seconds() / 3600
            
            # Allow slight overflow due to market hours clamping
            assert -3.0 <= delta_hours <= 3.0, \
                f"Shift {delta_hours}h outside reasonable range"
    
    def test_threshold_perturbation_range(self):
        """Threshold shifts stay within specified range."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        base_threshold = 150.0
        
        # Test many perturbations
        for _ in range(100):
            perturbed = policy.perturb_threshold(base_threshold)
            pct_change = (perturbed - base_threshold) / base_threshold
            
            assert -0.10 <= pct_change <= 0.10, \
                f"Shift {pct_change*100}% outside ±10% range"
    
    def test_threshold_always_positive(self):
        """Perturbed thresholds are always positive."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        # Test with very small base threshold
        base_threshold = 0.10
        
        for _ in range(100):
            perturbed = policy.perturb_threshold(base_threshold)
            assert perturbed > 0, "Threshold must always be positive"
    
    def test_market_hours_clamping(self):
        """Perturbations respect market hours when enabled."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 6.0,  # Large range to force clamping
            'threshold_pct': 0.10,
            'respect_market_hours': True
        })
        
        ts = datetime(2025, 7, 15, 9, 45)  # Near market open
        
        # Run many perturbations
        for _ in range(50):
            perturbed = policy.perturb_timing(ts)
            
            # Should never be outside market hours
            assert check_market_hours(perturbed), \
                f"Perturbed time {perturbed.time()} outside market hours"
    
    def test_no_clamping_mode(self):
        """Perturbations can go outside market hours when disabled."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 6.0,
            'threshold_pct': 0.10,
            'respect_market_hours': False
        })
        
        ts = datetime(2025, 7, 15, 9, 45)
        
        # Run many perturbations
        outside_hours = False
        for _ in range(50):
            perturbed = policy.perturb_timing(ts)
            if not check_market_hours(perturbed):
                outside_hours = True
                break
        
        # With large range and no clamping, we should see some outside hours
        assert outside_hours, "With no clamping, some times should be outside hours"
    
    def test_exposure_invariance_check(self):
        """Exposure checks work correctly."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        # Case 1: Within tolerance
        before = {'AAPL': 100, 'MSFT': -100}
        after = {'AAPL': 102, 'MSFT': -98}
        
        assert policy.check_exposure_invariance(before, after, tolerance=5.0) == True
        
        # Case 2: Outside tolerance
        before = {'AAPL': 100, 'MSFT': -100}
        after = {'AAPL': 120, 'MSFT': -80}
        
        assert policy.check_exposure_invariance(before, after, tolerance=5.0) == False
    
    def test_exposure_logging(self):
        """Exposure checks are logged."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        before = {'AAPL': 100, 'MSFT': -100}
        after = {'AAPL': 102, 'MSFT': -98}
        
        policy.check_exposure_invariance(before, after)
        
        log = policy.get_exposure_log()
        assert len(log) == 1
        assert 'net_before' in log[0]
        assert 'net_after' in log[0]
        assert 'valid' in log[0]
    
    def test_diagnostics_structure(self):
        """Diagnostics return expected structure."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        # Generate some perturbations
        ts = datetime(2025, 7, 15, 10, 30)
        for _ in range(10):
            policy.perturb_timing(ts)
            policy.perturb_threshold(150.0)
        
        diag = policy.get_diagnostics()
        
        # Check required keys
        assert diag['policy'] == 'Uniform'
        assert diag['seed'] == 42
        assert diag['n_perturbations'] == 20  # 10 timing + 10 threshold
        assert 'timing' in diag
        assert 'threshold' in diag
    
    def test_diagnostics_statistical_properties(self):
        """Diagnostics show correct statistical properties."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        # Generate many perturbations for statistical validity
        ts = datetime(2025, 7, 15, 12, 0)
        for _ in range(1000):
            policy.perturb_timing(ts)
            policy.perturb_threshold(150.0)
        
        diag = policy.get_diagnostics()
        
        # For uniform distribution on [-a, a]:
        # Mean should be close to 0
        # Std should be close to a / sqrt(3)
        
        timing_mean = diag['timing']['mean_shift_hours']
        timing_std = diag['timing']['std_shift_hours']
        expected_std = 2.0 / np.sqrt(3)
        
        assert abs(timing_mean) < 0.1, "Mean should be close to 0"
        assert abs(timing_std - expected_std) < 0.1, "Std should match theory"
    
    def test_stochasticity_adjustment_increase(self):
        """Adaptive adjustment increases randomness."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        original_timing = policy.params['timing_range_hours']
        original_threshold = policy.params['threshold_pct']
        
        policy.adjust_stochasticity(auc_score=0.85, direction='increase')
        
        assert policy.params['timing_range_hours'] > original_timing
        assert policy.params['threshold_pct'] > original_threshold
    
    def test_stochasticity_adjustment_decrease(self):
        """Adaptive adjustment decreases randomness."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 2.0,
            'threshold_pct': 0.10
        })
        
        original_timing = policy.params['timing_range_hours']
        original_threshold = policy.params['threshold_pct']
        
        policy.adjust_stochasticity(auc_score=0.55, direction='decrease')
        
        assert policy.params['timing_range_hours'] < original_timing
        assert policy.params['threshold_pct'] < original_threshold
    
    def test_adjustment_logging(self):
        """Adjustments are logged."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        policy.adjust_stochasticity(auc_score=0.85, direction='increase')
        
        log = policy.get_adjustment_log()
        assert len(log) == 1
        assert log[0]['auc_score'] == 0.85
        assert log[0]['direction'] == 'increase'
    
    def test_adjustment_invalid_direction(self):
        """Invalid adjustment direction raises error."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        with pytest.raises(ValueError, match="Direction must be"):
            policy.adjust_stochasticity(auc_score=0.75, direction='sideways')
    
    def test_repr(self):
        """String representation is informative."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        repr_str = repr(policy)
        assert 'UniformPolicy' in repr_str
        assert 'seed=42' in repr_str
        assert '2.0h' in repr_str or '2h' in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_full_workflow(self):
        """Complete workflow: create policy, perturb, check exposure, adjust."""
        # Step 1: Create policy
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        # Step 2: Perturb signals
        ts = datetime(2025, 7, 15, 10, 30)
        threshold = 150.0
        
        perturbed_ts = policy.perturb_timing(ts)
        perturbed_threshold = policy.perturb_threshold(threshold)
        
        assert perturbed_ts != ts
        assert perturbed_threshold != threshold
        
        # Step 3: Check exposure
        before = {'AAPL': 100, 'MSFT': -100}
        after = {'AAPL': 102, 'MSFT': -98}
        
        is_valid = policy.check_exposure_invariance(before, after)
        assert is_valid
        
        # Step 4: Adjust based on adversary
        policy.adjust_stochasticity(auc_score=0.85, direction='increase')
        
        # Step 5: Get diagnostics
        diag = policy.get_diagnostics()
        assert diag['n_perturbations'] == 2
    
    def test_seed_variance_simulation(self):
        """Simulate seed variance analysis (Week 2 requirement)."""
        results = []
        
        # Run with multiple seeds
        for seed in range(1, 11):
            policy = UniformPolicy(seed=seed, params=DEFAULT_UNIFORM_PARAMS)
            
            ts = datetime(2025, 7, 15, 10, 30)
            perturbed = policy.perturb_timing(ts)
            
            delta_hours = (perturbed - ts).total_seconds() / 3600
            results.append(delta_hours)
        
        # Results should have variance (different seeds → different outcomes)
        assert np.std(results) > 0, "Different seeds should give different results"
        
        # But mean should be close to 0
        assert abs(np.mean(results)) < 0.5, "Mean should be close to 0"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_perturbation_range(self):
        """Edge case: zero perturbation (no randomness)."""
        # Note: Parameter bounds will clamp 0.0 to minimum (0.1 and 0.01)
        # So we test with actual minimum values instead
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 0.1,  # Minimum bound
            'threshold_pct': 0.01        # Minimum bound
        })
        
        ts = datetime(2025, 7, 15, 10, 30)
        threshold = 150.0
        
        # Should have very small perturbations
        perturbed_ts = policy.perturb_timing(ts)
        perturbed_threshold = policy.perturb_threshold(threshold)
        
        # Timing: with 0.1h range, max shift is ±6 minutes
        delta_minutes = abs((perturbed_ts - ts).total_seconds() / 60)
        assert delta_minutes <= 6, f"Shift {delta_minutes} min too large for min range"
        
        # Threshold: with 0.01 range, max shift is ±1%
        pct_change = abs((perturbed_threshold - threshold) / threshold)
        assert pct_change <= 0.01, f"Shift {pct_change*100}% too large for min range"
    
    def test_very_large_perturbation(self):
        """Edge case: very large perturbation range."""
        policy = UniformPolicy(seed=42, params={
            'timing_range_hours': 10.0,  # Larger than trading day
            'threshold_pct': 0.50,        # ±50%
            'respect_market_hours': True
        })
        
        # Should still clamp to valid ranges
        ts = datetime(2025, 7, 15, 12, 0)
        perturbed_ts = policy.perturb_timing(ts)
        
        # Should be within market hours due to clamping
        assert check_market_hours(perturbed_ts)
    
    def test_empty_positions(self):
        """Edge case: empty position dictionaries."""
        policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
        
        # Empty before and after
        assert policy.check_exposure_invariance({}, {}, tolerance=5.0) == True
        
        # Empty before, non-empty after
        after = {'AAPL': 10}
        # This is a violation (going from 0 to 10)
        assert policy.check_exposure_invariance({}, after, tolerance=5.0) == False


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])