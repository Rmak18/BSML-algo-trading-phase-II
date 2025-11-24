"""
Test suite for 3-Policy Comparison

Run with:
    cd ~/Randomness-in-Algorithm-Trading---BSML/src
    python -m bsml.adaptive.test_3policies

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bsml.adaptive.bridge_3policies import prepare_three_policy_data
from bsml.adaptive.next_trade_predictor import NextTradePredictor
from bsml.adaptive.adaptive_loop_3policies import (
    ThreePolicyConfig,
    compare_three_policies,
    adaptive_three_policy_loop
)
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices


def create_mock_data(n_days: int = 100, n_symbols: int = 3) -> pd.DataFrame:
    """Create mock price data for testing"""
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Create price data
    prices_list = []
    for symbol in [f'SYM{i}' for i in range(n_symbols)]:
        for date in dates:
            prices_list.append({
                'date': date,
                'symbol': symbol,
                'price': 100 + np.random.randn() * 5
            })
    
    prices = pd.DataFrame(prices_list)
    
    return prices


def test_prepare_three_policy_data():
    """Test: Can we prepare datasets for all 3 policies?"""
    print("\n" + "="*80)
    print("TEST 1: Prepare 3-Policy Datasets")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data")
        prices = create_mock_data(n_days=100, n_symbols=3)
    
    # Create policies with real implementations
    uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
    uniform_params['price_noise'] = 0.03
    uniform_policy = UniformPolicy(params=uniform_params)
    
    pink_policy = PinkNoisePolicy(alpha=1.0, price_scale=0.04, seed=42)
    ou_policy = OUPolicy(theta=0.15, sigma=0.02, price_scale=1.0, seed=42)
    
    datasets = prepare_three_policy_data(
        prices,
        baseline_generate,
        uniform_policy,
        pink_policy,
        ou_policy,
        lookback=5,
        verbose=True
    )
    
    print(f"\n✓ Baseline dataset: {len(datasets['baseline'])} observations")
    print(f"✓ Uniform dataset: {len(datasets['uniform'])} observations")
    print(f"✓ Pink Noise dataset: {len(datasets['pink_noise'])} observations")
    print(f"✓ OU dataset: {len(datasets['ou'])} observations")
    
    assert len(datasets['baseline']) > 0, "Baseline dataset should have observations"
    assert len(datasets['uniform']) > 0, "Uniform dataset should have observations"
    
    print("✅ PASSED")
    return True


def test_compare_three_policies():
    """Test: Can we compare all 3 policies?"""
    print("\n" + "="*80)
    print("TEST 2: Compare 3 Policies")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data")
        prices = create_mock_data(n_days=100, n_symbols=3)
    
    config = ThreePolicyConfig(
        lookback=5,
        max_iterations=1,
        initial_price_noise=0.03,
        initial_time_noise=30.0,
        pink_alpha=1.0,
        pink_price_scale=0.04,
        ou_theta=0.15,
        ou_sigma=0.02,
        ou_price_scale=1.0
    )
    
    results = compare_three_policies(
        prices,
        price_noise=0.03,
        time_noise=30.0,
        config=config,
        verbose=True
    )
    
    assert results['success'], "Comparison should succeed"
    assert 'comparison' in results, "Should have comparison"
    
    comp = results['comparison']
    
    print(f"\n✓ Baseline AUC: {comp['baseline_auc']:.4f}")
    print(f"✓ Uniform AUC: {comp['uniform_auc']:.4f}")
    print(f"✓ Pink Noise AUC: {comp['pink_auc']:.4f}")
    print(f"✓ OU AUC: {comp['ou_auc']:.4f}")
    print(f"✓ Winner: {comp['best_policy']}")
    
    print("✅ PASSED")
    return True


def test_adaptive_loop():
    """Test: Can we run the adaptive 3-policy loop?"""
    print("\n" + "="*80)
    print("TEST 3: Adaptive 3-Policy Loop (2 iterations)")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data")
        prices = create_mock_data(n_days=100, n_symbols=3)
    
    config = ThreePolicyConfig(
        lookback=5,
        max_iterations=2,  # Just 2 iterations for testing
        initial_price_noise=0.03,
        initial_time_noise=30.0,
        pink_alpha=1.0,
        pink_price_scale=0.04,
        ou_theta=0.15,
        ou_sigma=0.02,
        ou_price_scale=1.0,
        convergence_patience=1
    )
    
    results_df = adaptive_three_policy_loop(prices, config, verbose=True)
    
    assert len(results_df) > 0, "Should have results"
    assert 'baseline_auc' in results_df.columns, "Should have baseline AUC"
    assert 'uniform_auc' in results_df.columns, "Should have uniform AUC"
    assert 'pink_auc' in results_df.columns, "Should have pink AUC"
    assert 'ou_auc' in results_df.columns, "Should have OU AUC"
    assert 'best_policy' in results_df.columns, "Should have best policy"
    
    print(f"\n✓ Completed {len(results_df)} iterations")
    print(f"✓ Final winner: {results_df.iloc[-1]['best_policy']}")
    
    print("✅ PASSED")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("3-POLICY COMPARISON TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_prepare_three_policy_data,
        test_compare_three_policies,
        test_adaptive_loop
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {test_fn.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
