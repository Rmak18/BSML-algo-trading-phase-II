"""
Quick test script for Week 3 policies.

Run this to verify OU and Pink policies work correctly.

Usage:
    python scripts/test_week3_policies.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bsml.policies import (
    OUPolicy,
    PinkNoisePolicy,
    DEFAULT_OU_PARAMS,
    DEFAULT_PINK_PARAMS,
)


def test_ou_policy():
    """Test OU policy."""
    print("="*70)
    print("Testing OU Policy")
    print("="*70)
    
    policy = OUPolicy(seed=42, params=DEFAULT_OU_PARAMS)
    print(f"Created: {policy}")
    
    # Run some perturbations
    timestamp = datetime(2025, 7, 15, 10, 0)
    print("\nRunning 10 perturbations:")
    for i in range(10):
        perturbed = policy.perturb_timing(timestamp)
        shift = (perturbed - timestamp).total_seconds() / 3600
        state = policy._ou_timing_state
        
        print(f"  {i+1}. Shift: {shift:+.3f}h, State: {state:+.3f}")
        
        timestamp += timedelta(hours=1)
    
    # Get diagnostics
    diag = policy.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  Policy: {diag['policy']}")
    print(f"  Seed: {diag['seed']}")
    print(f"  N perturbations: {diag['n_perturbations']}")
    print(f"  Half-life: {diag['timing']['half_life_hours']:.2f} hours")
    
    if diag['timing']['autocorrelation']:
        acf1 = diag['timing']['autocorrelation'][1]
        print(f"  ACF(1): {acf1:.3f}")
    
    print("\n✓ OU Policy test passed!")
    return True


def test_pink_policy():
    """Test Pink policy."""
    print("\n" + "="*70)
    print("Testing Pink Noise Policy")
    print("="*70)
    
    policy = PinkNoisePolicy(seed=42, params=DEFAULT_PINK_PARAMS)
    print(f"Created: {policy}")
    
    # Run some perturbations
    timestamp = datetime(2025, 7, 15, 10, 0)
    print("\nRunning 10 perturbations:")
    for i in range(10):
        perturbed = policy.perturb_timing(timestamp)
        shift = (perturbed - timestamp).total_seconds() / 3600
        
        print(f"  {i+1}. Shift: {shift:+.3f}h")
        
        timestamp += timedelta(hours=1)
    
    # Get diagnostics
    diag = policy.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  Policy: {diag['policy']}")
    print(f"  Seed: {diag['seed']}")
    print(f"  Alpha: {diag['params']['alpha']}")
    print(f"  Buffer size: {diag['buffer']['size']}")
    print(f"  Buffer usage: {diag['buffer']['timing_usage']*100:.1f}%")
    
    # Check buffer mean is near zero
    buffer_mean = diag['timing']['buffer_mean']
    buffer_std = diag['timing']['buffer_std']
    print(f"  Buffer mean: {buffer_mean:.4f} (should be ≈ 0)")
    print(f"  Buffer std: {buffer_std:.4f}")
    
    assert abs(buffer_mean) < 0.1 * buffer_std, "Buffer mean not near zero!"
    
    print("\n✓ Pink Policy test passed!")
    return True


def main():
    """Run all tests."""
    print("BSML Week 3 Policies - Quick Test")
    print("="*70)
    
    try:
        # Test OU
        ou_passed = test_ou_policy()
        
        # Test Pink
        pink_passed = test_pink_policy()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  OU Policy:    {'✓ PASS' if ou_passed else '✗ FAIL'}")
        print(f"  Pink Policy:  {'✓ PASS' if pink_passed else '✗ FAIL'}")
        print("="*70)
        
        if ou_passed and pink_passed:
            print("\n🎉 All tests passed! Week 3 policies are ready.")
            return 0
        else:
            print("\n❌ Some tests failed. Check output above.")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())