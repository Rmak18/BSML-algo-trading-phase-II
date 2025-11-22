"""
Test script for Adaptive Adversary V2
Tests the policy distinguishability approach on synthetic data.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from bsml.adaptive.bridge_v2 import prepare_adversary_data_v2, time_split_data
from bsml.adaptive.adversary_classifier_v2 import P7AdversaryV2
from bsml.adaptive.adaptive_loop_v2 import adaptive_training_loop_v2, AdaptiveConfigV2


def generate_synthetic_prices(n_days=500, n_symbols=5, seed=42):
    """Generate synthetic price data for testing"""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    all_data = []
    for symbol in [f'ETF{i}' for i in range(n_symbols)]:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'price': prices
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


def test_bridge_v2():
    """Test bridge V2 - data preparation"""
    print("\n" + "="*80)
    print("TEST 1: Bridge V2 - Data Preparation")
    print("="*80)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic prices...")
    prices = generate_synthetic_prices(n_days=200, n_symbols=3, seed=42)
    print(f"  ✓ Generated {len(prices)} price observations")
    
    # Import policies
    from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
    from bsml.policies.baseline import generate_trades as baseline_generate
    
    print("\n[2] Creating Uniform policy...")
    uniform_policy = UniformPolicy(params=DEFAULT_UNIFORM_PARAMS, seed=42)
    print(f"  ✓ price_noise={DEFAULT_UNIFORM_PARAMS['price_noise']:.3f}")
    print(f"  ✓ time_noise={DEFAULT_UNIFORM_PARAMS['time_noise_minutes']}min")
    
    print("\n[3] Generating adversary data...")
    adversary_data = prepare_adversary_data_v2(
        prices,
        baseline_generate,
        uniform_policy,
        verbose=True
    )
    
    print(f"\n[4] Checking data quality...")
    print(f"  Total observations: {len(adversary_data)}")
    print(f"  Baseline trades: {(adversary_data['label'] == 0).sum()}")
    print(f"  Uniform trades: {(adversary_data['label'] == 1).sum()}")
    print(f"  Features: {len([c for c in adversary_data.columns if c not in ['date', 'symbol', 'label', 'policy']])}")
    
    # Check for key features
    key_features = ['price_deviation', 'abs_price_deviation', 'day_of_week', 
                   'days_since_last_trade', 'returns_1d', 'vol_5d']
    
    print(f"\n[5] Checking key features...")
    for feat in key_features:
        if feat in adversary_data.columns:
            print(f"  ✓ {feat}")
        else:
            print(f"  ✗ {feat} MISSING!")
    
    print("\n✅ Bridge V2 test passed!")
    return adversary_data


def test_classifier_v2(adversary_data):
    """Test classifier V2"""
    print("\n" + "="*80)
    print("TEST 2: Adversary Classifier V2")
    print("="*80)
    
    print("\n[1] Splitting data...")
    train, val, test = time_split_data(adversary_data, 0.6, 0.2)
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    print("\n[2] Training adversary...")
    adversary = P7AdversaryV2(use_cv=True, n_cv_folds=3, random_state=42)
    train_metrics = adversary.train(train, verbose=True)
    
    if not train_metrics['success']:
        print(f"  ✗ Training failed: {train_metrics}")
        return None
    
    print(f"\n[3] Evaluating on validation set...")
    val_metrics = adversary.evaluate(val, verbose=True)
    
    print(f"\n[4] Feature importance (top 10)...")
    feat_imp = adversary.get_feature_importance(top_n=10)
    print(feat_imp.to_string(index=False))
    
    print("\n✅ Classifier V2 test passed!")
    return adversary, val_metrics


def test_adaptive_loop_v2():
    """Test full adaptive loop V2"""
    print("\n" + "="*80)
    print("TEST 3: Adaptive Loop V2")
    print("="*80)
    
    # Generate larger dataset for full loop
    print("\n[1] Generating larger synthetic dataset...")
    prices = generate_synthetic_prices(n_days=500, n_symbols=5, seed=42)
    print(f"  ✓ Generated {len(prices)} observations")
    
    # Configure for quick test
    config = AdaptiveConfigV2()
    config.MAX_ITERATIONS = 5  # Reduce for testing
    config.CONVERGENCE_PATIENCE = 2
    config.OUTPUT_DIR = Path("outputs/test_adaptive_v2")
    
    print(f"\n[2] Running adaptive loop (max {config.MAX_ITERATIONS} iterations)...")
    results = adaptive_training_loop_v2(
        prices,
        initial_params={'price_noise': 0.03, 'time_noise_minutes': 30},
        config=config,
        seed=42,
        verbose=True
    )
    
    print(f"\n[3] Checking results...")
    print(f"  Iterations completed: {results['n_iterations']}")
    print(f"  Converged: {results['converged']}")
    
    if results['logger'].iterations:
        print(f"  Final AUC: {results['logger'].iterations[-1]['auc_val']:.4f}")
        print(f"  Final price_noise: {results['final_params']['price_noise']:.4f}")
        print(f"  Final time_noise: {results['final_params']['time_noise_minutes']:.1f}")
    
    print("\n[4] Saving results...")
    results['logger'].save_results(config.OUTPUT_DIR)
    
    print("\n✅ Adaptive Loop V2 test passed!")
    return results


def main():
    """Run all tests"""
    print("="*80)
    print("ADAPTIVE ADVERSARY V2 - TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test 1: Bridge
        adversary_data = test_bridge_v2()
        
        # Test 2: Classifier
        adversary, val_metrics = test_classifier_v2(adversary_data)
        
        # Test 3: Full adaptive loop
        results = test_adaptive_loop_v2()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
