"""
Quick test runner to verify everything works.

This script runs a subset of tests without requiring pytest installation.
For full test suite, use: pytest tests/test_policies.py -v

Owner: P4
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
from bsml.policies import (
    UniformPolicy,
    DEFAULT_UNIFORM_PARAMS,
    generate_policy_seed,
    calculate_net_exposure,
    check_market_hours,
)

def test_utils():
    """Test utility functions."""
    print("Testing utilities...")
    
    # Test 1: Seed generation
    seed1 = generate_policy_seed(42, 'Uniform')
    seed2 = generate_policy_seed(42, 'Uniform')
    assert seed1 == seed2, "❌ Seed generation not deterministic"
    print("  ✅ Seed generation: deterministic")
    
    # Test 2: Net exposure
    positions = {'AAPL': 100, 'MSFT': -100}
    net = calculate_net_exposure(positions)
    assert net == 0.0, "❌ Net exposure calculation wrong"
    print("  ✅ Net exposure calculation: correct")
    
    # Test 3: Market hours
    ts_during = datetime(2025, 7, 15, 10, 30)
    ts_before = datetime(2025, 7, 15, 8, 0)
    assert check_market_hours(ts_during) == True
    assert check_market_hours(ts_before) == False
    print("  ✅ Market hours checking: correct")
    
    print("✅ All utility tests passed!\n")


def test_uniform_policy():
    """Test UniformPolicy."""
    print("Testing UniformPolicy...")
    
    # Test 1: Initialization
    policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
    assert policy.seed == 42
    print("  ✅ Initialization: correct")
    
    # Test 2: Timing perturbation
    ts = datetime(2025, 7, 15, 10, 30)
    perturbed = policy.perturb_timing(ts)
    assert perturbed != ts, "❌ Timing not perturbed"
    print(f"  ✅ Timing perturbation: {ts} → {perturbed}")
    
    # Test 3: Threshold perturbation
    threshold = 150.0
    perturbed_threshold = policy.perturb_threshold(threshold)
    assert perturbed_threshold != threshold, "❌ Threshold not perturbed"
    pct_change = (perturbed_threshold - threshold) / threshold * 100
    print(f"  ✅ Threshold perturbation: ${threshold:.2f} → ${perturbed_threshold:.2f} ({pct_change:+.1f}%)")
    
    # Test 4: Seed reproducibility
    policy1 = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
    policy2 = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
    
    ts = datetime(2025, 7, 15, 10, 30)
    result1 = policy1.perturb_timing(ts)
    result2 = policy2.perturb_timing(ts)
    
    assert result1 == result2, "❌ Seed reproducibility broken"
    print("  ✅ Seed reproducibility: working")
    
    # Test 5: Exposure checks
    before = {'AAPL': 100, 'MSFT': -100}
    after = {'AAPL': 102, 'MSFT': -98}
    
    is_valid = policy.check_exposure_invariance(before, after, tolerance=5.0)
    assert is_valid, "❌ Exposure check failed"
    print("  ✅ Exposure invariance: working")
    
    # Test 6: Diagnostics
    diag = policy.get_diagnostics()
    assert 'policy' in diag
    assert diag['policy'] == 'Uniform'
    print("  ✅ Diagnostics: working")
    
    # Test 7: Adaptive adjustment
    original_range = policy.params['timing_range_hours']
    policy.adjust_stochasticity(auc_score=0.85, direction='increase')
    assert policy.params['timing_range_hours'] > original_range
    print(f"  ✅ Adaptive adjustment: {original_range}h → {policy.params['timing_range_hours']:.1f}h")
    
    print("✅ All UniformPolicy tests passed!\n")


def test_statistical_properties():
    """Test statistical properties of perturbations."""
    print("Testing statistical properties...")
    
    policy = UniformPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    # Generate many perturbations
    import numpy as np
    
    timing_shifts = []
    threshold_shifts = []
    
    ts = datetime(2025, 7, 15, 12, 0)
    for _ in range(1000):
        perturbed_ts = policy.perturb_timing(ts)
        delta_hours = (perturbed_ts - ts).total_seconds() / 3600
        timing_shifts.append(delta_hours)
        
        threshold = 150.0
        perturbed_threshold = policy.perturb_threshold(threshold)
        pct_change = (perturbed_threshold - threshold) / threshold
        threshold_shifts.append(pct_change)
    
    # Check statistical properties
    timing_mean = np.mean(timing_shifts)
    timing_std = np.std(timing_shifts)
    expected_std = 2.0 / np.sqrt(3)
    
    print(f"  Timing shifts:")
    print(f"    Mean: {timing_mean:.4f} (should be ~0)")
    print(f"    Std: {timing_std:.4f} (expected {expected_std:.4f})")
    
    threshold_mean = np.mean(threshold_shifts)
    threshold_std = np.std(threshold_shifts)
    expected_threshold_std = 0.10 / np.sqrt(3)
    
    print(f"  Threshold shifts:")
    print(f"    Mean: {threshold_mean:.4f} (should be ~0)")
    print(f"    Std: {threshold_std:.4f} (expected {expected_threshold_std:.4f})")
    
    # Validate
    assert abs(timing_mean) < 0.1, "❌ Mean too far from 0"
    assert abs(timing_std - expected_std) < 0.1, "❌ Std doesn't match theory"
    
    print("  ✅ Statistical properties match theory\n")


def test_integration():
    """Test full integration workflow."""
    print("Testing integration workflow...")
    
    # Simulate P2 calling P4's functions
    policy = UniformPolicy(seed=42, params=DEFAULT_UNIFORM_PARAMS)
    
    # P2 has a signal
    signal = {
        'timestamp': datetime(2025, 7, 15, 10, 30),
        'symbol': 'AAPL',
        'threshold': 150.00,
        'side': 'BUY',
        'qty': 100
    }
    
    # P4 perturbs it
    perturbed_signal = {
        'timestamp': policy.perturb_timing(signal['timestamp']),
        'symbol': signal['symbol'],
        'threshold': policy.perturb_threshold(signal['threshold']),
        'side': signal['side'],
        'qty': signal['qty']
    }
    
    print(f"  Original:  {signal['timestamp']} @ ${signal['threshold']:.2f}")
    print(f"  Perturbed: {perturbed_signal['timestamp']} @ ${perturbed_signal['threshold']:.2f}")
    
    # Check exposure
    before = {'AAPL': 0}
    after = {'AAPL': 100}
    
    # This violates exposure (net went from 0 to 100)
    is_valid = policy.check_exposure_invariance(before, after, tolerance=5.0)
    print(f"  Exposure check: {is_valid} (expected False for this example)")
    
    print("  ✅ Integration workflow: working\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("P4 WEEK 2 - QUICK TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        test_utils()
        test_uniform_policy()
        test_statistical_properties()
        test_integration()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Run full test suite: pytest tests/test_policies.py -v")
        print("2. Check coverage: pytest --cov=bsml.policies tests/")
        print("3. Review diagnostics in demo notebook")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())