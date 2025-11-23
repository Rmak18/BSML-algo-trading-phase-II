"""
Test suite for Next-Trade Prediction

Run with:
    cd ~/Randomness-in-Algorithm-Trading---BSML/src
    python -m bsml.adaptive.test_next_trade

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bsml.adaptive.bridge_next_trade import create_next_trade_dataset, prepare_next_trade_data
from bsml.adaptive.next_trade_predictor import NextTradePredictor
from bsml.adaptive.adaptive_loop_next_trade import (
    NextTradePredictionConfig,
    run_next_trade_experiment,
    adaptive_next_trade_loop
)
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices


def create_mock_data(n_days: int = 100, n_symbols: int = 3) -> tuple:
    """Create mock price and trade data for testing"""
    
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
    
    # Create trade data (trades every 2-5 days)
    trades_list = []
    for symbol in [f'SYM{i}' for i in range(n_symbols)]:
        current_date = dates[0]
        while current_date < dates[-1]:
            trades_list.append({
                'date': current_date,
                'symbol': symbol,
                'side': np.random.choice(['BUY', 'SELL']),
                'qty': np.random.randint(100, 1000),
                'price': 100 + np.random.randn() * 5,
                'ref_price': 100 + np.random.randn() * 5
            })
            # Next trade in 2-5 days
            current_date += timedelta(days=np.random.randint(2, 6))
    
    trades = pd.DataFrame(trades_list)
    
    return prices, trades


def test_create_next_trade_dataset():
    """Test: Can we create next-trade prediction dataset?"""
    print("\n" + "="*80)
    print("TEST 1: Create Next-Trade Dataset")
    print("="*80)
    
    prices, trades = create_mock_data(n_days=100, n_symbols=2)
    
    print(f"Mock data: {len(prices)} price rows, {len(trades)} trades")
    
    dataset = create_next_trade_dataset(trades, prices, lookback=5)
    
    print(f"✓ Created dataset: {len(dataset)} observations")
    print(f"  Features: {len([c for c in dataset.columns if c not in ['symbol', 'date', 'label']])} features")
    print(f"  Label distribution:")
    print(f"    Trade tomorrow: {(dataset['label'] == 1).sum()} ({(dataset['label'] == 1).mean()*100:.1f}%)")
    print(f"    No trade: {(dataset['label'] == 0).sum()} ({(dataset['label'] == 0).mean()*100:.1f}%)")
    
    assert len(dataset) > 0, "Dataset should have observations"
    assert 'label' in dataset.columns, "Dataset should have labels"
    
    print("✅ PASSED")
    return True


def test_next_trade_predictor():
    """Test: Can we train and evaluate a next-trade predictor?"""
    print("\n" + "="*80)
    print("TEST 2: Train Next-Trade Predictor")
    print("="*80)
    
    prices, trades = create_mock_data(n_days=150, n_symbols=3)
    dataset = create_next_trade_dataset(trades, prices, lookback=5)
    
    print(f"Dataset: {len(dataset)} observations")
    
    # Split
    train_size = int(len(dataset) * 0.7)
    train = dataset.iloc[:train_size]
    val = dataset.iloc[train_size:]
    
    print(f"Train: {len(train)}, Val: {len(val)}")
    
    # Train
    predictor = NextTradePredictor()
    train_metrics = predictor.train(train, verbose=True)
    
    assert train_metrics['success'], "Training should succeed"
    print(f"\n✓ Training succeeded: {train_metrics['n_features']} features")
    
    # Evaluate
    val_metrics = predictor.evaluate(val, verbose=True)
    
    assert val_metrics['success'], "Evaluation should succeed"
    assert 0 <= val_metrics['auc'] <= 1, "AUC should be between 0 and 1"
    
    print(f"\n✓ Evaluation succeeded: AUC = {val_metrics['auc']:.4f}")
    
    print("✅ PASSED")
    return True


def test_prepare_next_trade_data():
    """Test: Can we prepare baseline and uniform datasets?"""
    print("\n" + "="*80)
    print("TEST 3: Prepare Baseline and Uniform Datasets")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)  # Use first 500 rows for speed
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data (real data not available)")
        prices, _ = create_mock_data(n_days=100, n_symbols=3)
    
    uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
    uniform_params['price_noise'] = 0.03
    uniform_policy = UniformPolicy(params=uniform_params)
    
    baseline_data, uniform_data = prepare_next_trade_data(
        prices,
        baseline_generate,
        uniform_policy,
        lookback=5,
        verbose=True
    )
    
    print(f"\n✓ Baseline dataset: {len(baseline_data)} observations")
    print(f"✓ Uniform dataset: {len(uniform_data)} observations")
    
    assert len(baseline_data) > 0, "Baseline dataset should have observations"
    assert len(uniform_data) > 0, "Uniform dataset should have observations"
    
    print("✅ PASSED")
    return True


def test_run_experiment():
    """Test: Can we run a single experiment comparing policies?"""
    print("\n" + "="*80)
    print("TEST 4: Run Single Experiment")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data")
        prices, _ = create_mock_data(n_days=100, n_symbols=3)
    
    config = NextTradePredictionConfig(
        lookback=5,
        max_iterations=1,
        initial_price_noise=0.03,
        initial_time_noise=30.0
    )
    
    results = run_next_trade_experiment(
        prices,
        price_noise=0.03,
        time_noise=30.0,
        config=config,
        verbose=True
    )
    
    assert results['success'], "Experiment should succeed"
    assert 'baseline' in results, "Should have baseline results"
    assert 'uniform' in results, "Should have uniform results"
    assert 'comparison' in results, "Should have comparison"
    
    baseline_auc = results['comparison']['baseline_auc']
    uniform_auc = results['comparison']['uniform_auc']
    
    print(f"\n✓ Baseline AUC: {baseline_auc:.4f}")
    print(f"✓ Uniform AUC: {uniform_auc:.4f}")
    print(f"✓ AUC Reduction: {results['comparison']['auc_reduction']:.4f}")
    
    print("✅ PASSED")
    return True


def test_adaptive_loop():
    """Test: Can we run the adaptive loop (limited iterations)?"""
    print("\n" + "="*80)
    print("TEST 5: Adaptive Loop (2 iterations)")
    print("="*80)
    
    # Use smaller real data
    try:
        prices = load_prices("data/ALL_backtest.csv")
        prices = prices.head(500)
        print(f"Loaded real data: {len(prices)} rows")
    except:
        print("Using mock data")
        prices, _ = create_mock_data(n_days=100, n_symbols=3)
    
    config = NextTradePredictionConfig(
        lookback=5,
        max_iterations=2,  # Just 2 iterations for testing
        initial_price_noise=0.03,
        initial_time_noise=30.0,
        convergence_patience=1
    )
    
    results_df = adaptive_next_trade_loop(prices, config, verbose=True)
    
    assert len(results_df) > 0, "Should have results"
    assert 'baseline_auc' in results_df.columns, "Should have baseline AUC"
    assert 'uniform_auc' in results_df.columns, "Should have uniform AUC"
    
    print(f"\n✓ Completed {len(results_df)} iterations")
    print(f"✓ Final uniform AUC: {results_df.iloc[-1]['uniform_auc']:.4f}")
    
    print("✅ PASSED")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("NEXT-TRADE PREDICTION TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_create_next_trade_dataset,
        test_next_trade_predictor,
        test_prepare_next_trade_data,
        test_run_experiment,
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
