"""
P7 Adaptive Loop - Next-Trade Prediction Task

Compare predictability of baseline vs uniform policies.
Lower AUC = less predictable = better randomization.

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices

from .bridge_next_trade import prepare_next_trade_data, time_split_data
from .next_trade_predictor import NextTradePredictor


@dataclass
class NextTradePredictionConfig:
    """Configuration for next-trade prediction experiment"""
    lookback: int = 5  # Number of past trades to use as features
    max_iterations: int = 10
    initial_price_noise: float = 0.03
    initial_time_noise: float = 30.0
    noise_increase_rate: float = 1.25
    target_auc_high: float = 0.65  # If above this, increase noise
    target_auc_low: float = 0.55   # If below this, decrease noise
    convergence_patience: int = 3


def run_next_trade_experiment(
    prices: pd.DataFrame,
    price_noise: float,
    time_noise: float,
    config: NextTradePredictionConfig,
    verbose: bool = True
) -> Dict:
    """
    Run experiment: train predictors on baseline and uniform, compare AUCs.
    
    Returns:
        Dict with results for both policies
    """
    
    # Create uniform policy with current parameters
    uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
    uniform_params['price_noise'] = price_noise
    uniform_params['time_noise_minutes'] = time_noise
    uniform_policy = UniformPolicy(params=uniform_params)
    
    if verbose:
        print(f"[Experiment] Generating datasets (lookback={config.lookback})...")
    
    # Prepare datasets
    baseline_data, uniform_data = prepare_next_trade_data(
        prices,
        baseline_generate,
        uniform_policy,
        lookback=config.lookback,
        verbose=verbose
    )
    
    if len(baseline_data) == 0 or len(uniform_data) == 0:
        return {
            'baseline_auc': 0.50,
            'uniform_auc': 0.50,
            'success': False,
            'reason': 'insufficient_data'
        }
    
    results = {}
    
    # =========================================================================
    # BASELINE PREDICTOR
    # =========================================================================
    
    if verbose:
        print("\n" + "="*80)
        print("BASELINE POLICY - NEXT-TRADE PREDICTION")
        print("="*80)
    
    baseline_train, baseline_val, baseline_test = time_split_data(baseline_data)
    
    if verbose:
        print(f"[Baseline] Train: {len(baseline_train)}, Val: {len(baseline_val)}, Test: {len(baseline_test)}")
    
    baseline_predictor = NextTradePredictor()
    baseline_train_metrics = baseline_predictor.train(baseline_train, verbose=verbose)
    baseline_val_metrics = baseline_predictor.evaluate(baseline_val, verbose=verbose)
    
    results['baseline'] = {
        'train_metrics': baseline_train_metrics,
        'val_metrics': baseline_val_metrics,
        'auc': baseline_val_metrics['auc'],
        'n_samples': len(baseline_data)
    }
    
    # =========================================================================
    # UNIFORM PREDICTOR
    # =========================================================================
    
    if verbose:
        print("\n" + "="*80)
        print("UNIFORM POLICY - NEXT-TRADE PREDICTION")
        print("="*80)
    
    uniform_train, uniform_val, uniform_test = time_split_data(uniform_data)
    
    if verbose:
        print(f"[Uniform] Train: {len(uniform_train)}, Val: {len(uniform_val)}, Test: {len(uniform_test)}")
    
    uniform_predictor = NextTradePredictor()
    uniform_train_metrics = uniform_predictor.train(uniform_train, verbose=verbose)
    uniform_val_metrics = uniform_predictor.evaluate(uniform_val, verbose=verbose)
    
    results['uniform'] = {
        'train_metrics': uniform_train_metrics,
        'val_metrics': uniform_val_metrics,
        'auc': uniform_val_metrics['auc'],
        'n_samples': len(uniform_data)
    }
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    
    baseline_auc = baseline_val_metrics['auc']
    uniform_auc = uniform_val_metrics['auc']
    auc_reduction = baseline_auc - uniform_auc
    auc_reduction_pct = (auc_reduction / baseline_auc * 100) if baseline_auc > 0 else 0
    
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Baseline AUC:  {baseline_auc:.4f}")
        print(f"Uniform AUC:   {uniform_auc:.4f}")
        print(f"AUC Reduction: {auc_reduction:.4f} ({auc_reduction_pct:.1f}%)")
        
        if uniform_auc < baseline_auc:
            print("✓ Uniform is LESS predictable than baseline (good!)")
        else:
            print("⚠️  Uniform is MORE predictable than baseline (unexpected)")
    
    results['comparison'] = {
        'baseline_auc': float(baseline_auc),
        'uniform_auc': float(uniform_auc),
        'auc_reduction': float(auc_reduction),
        'auc_reduction_pct': float(auc_reduction_pct)
    }
    
    results['success'] = True
    
    return results


def adaptive_next_trade_loop(
    prices: pd.DataFrame,
    config: NextTradePredictionConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Adaptive loop: Increase noise until uniform's AUC is sufficiently low.
    
    Goal: Find noise parameters where uniform trades become unpredictable.
    """
    
    if verbose:
        print("="*80)
        print("P7 ADAPTIVE NEXT-TRADE PREDICTION")
        print("="*80)
        print(f"Task: Compare predictability of baseline vs uniform policies")
        print(f"Goal: Find noise parameters where uniform is unpredictable (AUC < {config.target_auc_low})")
        print(f"Max iterations: {config.max_iterations}")
        print(f"Initial params: price_noise={config.initial_price_noise:.4f}, time_noise={config.initial_time_noise:.1f}min")
        print()
    
    # Track results
    iterations = []
    
    price_noise = config.initial_price_noise
    time_noise = config.initial_time_noise
    
    converged = False
    in_target_count = 0
    
    for iteration in range(1, config.max_iterations + 1):
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iteration}/{config.max_iterations}")
            print("="*80)
            print(f"Testing: price_noise={price_noise:.4f}, time_noise={time_noise:.1f}min")
            print()
        
        # Run experiment
        results = run_next_trade_experiment(
            prices, price_noise, time_noise, config, verbose=verbose
        )
        
        if not results['success']:
            if verbose:
                print(f"✗ Iteration {iteration} failed: {results.get('reason', 'unknown')}")
            continue
        
        baseline_auc = results['comparison']['baseline_auc']
        uniform_auc = results['comparison']['uniform_auc']
        auc_reduction = results['comparison']['auc_reduction']
        
        # Decide action
        if uniform_auc > config.target_auc_high:
            action = "INCREASE"
            reason = f"Uniform still too predictable (AUC={uniform_auc:.3f})"
            multiplier = config.noise_increase_rate
            in_target_count = 0
        elif uniform_auc < config.target_auc_low:
            action = "DECREASE"
            reason = f"Uniform too unpredictable (AUC={uniform_auc:.3f}), may be losing signal"
            multiplier = 1.0 / config.noise_increase_rate
            in_target_count = 0
        else:
            action = "HOLD"
            reason = f"Uniform in target range (AUC={uniform_auc:.3f})"
            multiplier = 1.0
            in_target_count += 1
        
        if verbose:
            print("\n" + "-"*80)
            print(f"RESULT: Uniform AUC = {uniform_auc:.4f}")
            print(f"ACTION: {action}")
            print(f"REASON: {reason}")
        
        # Record iteration
        iterations.append({
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'price_noise': price_noise,
            'time_noise_minutes': time_noise,
            'baseline_auc': baseline_auc,
            'uniform_auc': uniform_auc,
            'auc_reduction': auc_reduction,
            'auc_reduction_pct': results['comparison']['auc_reduction_pct'],
            'action': action,
            'reason': reason,
            'multiplier': multiplier
        })
        
        # Check convergence
        if in_target_count >= config.convergence_patience:
            converged = True
            if verbose:
                print(f"\n🎉 CONVERGED after {iteration} iterations!")
            break
        
        # Adjust parameters
        if action != "HOLD":
            price_noise *= multiplier
            time_noise *= multiplier
            
            if verbose:
                print(f"\nAdjustment:")
                print(f"  price_noise: {price_noise:.4f}")
                print(f"  time_noise: {time_noise:.1f} min")
    
    # Summary
    results_df = pd.DataFrame(iterations)
    
    if verbose:
        print("\n" + "="*80)
        print("ITERATION SUMMARY")
        print("="*80)
        print(results_df[['iteration', 'baseline_auc', 'uniform_auc', 'auc_reduction', 'action']].to_string(index=False))
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Iterations: {len(iterations)}")
        if len(iterations) > 0:
            final = iterations[-1]
            print(f"Final Baseline AUC: {final['baseline_auc']:.4f}")
            print(f"Final Uniform AUC: {final['uniform_auc']:.4f}")
            print(f"Final AUC Reduction: {final['auc_reduction']:.4f} ({final['auc_reduction_pct']:.1f}%)")
            print(f"Final params: price_noise={final['price_noise']:.4f}, time_noise={final['time_noise_minutes']:.1f}min")
            
            if final['uniform_auc'] < config.target_auc_low:
                print("\n✓ Good: Uniform is unpredictable (low AUC)")
            elif final['uniform_auc'] > config.target_auc_high:
                print("\n⚠️  Warning: Uniform still too predictable")
            else:
                print("\n✓ Good: Uniform in target range")
    
    return results_df


def main():
    """Main entry point"""
    
    print("="*80)
    print("P7 WEEK 3: NEXT-TRADE PREDICTION EXPERIMENT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Task: Compare baseline vs uniform policy predictability")
    print("Method: Train next-trade predictors, compare AUCs")
    print("Goal: Lower AUC = Less predictable = Better randomization")
    print()
    
    # Load data
    print("[1/3] Loading data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    print()
    
    # Configure experiment
    config = NextTradePredictionConfig(
        lookback=5,
        max_iterations=10,
        initial_price_noise=0.03,
        initial_time_noise=30.0,
        target_auc_high=0.65,
        target_auc_low=0.55,
        convergence_patience=3
    )
    
    # Run adaptive loop
    print("[2/3] Running adaptive next-trade prediction loop...")
    print()
    
    results_df = adaptive_next_trade_loop(prices, config, verbose=True)
    
    # Save results
    print("\n[3/3] Saving results...")
    output_dir = Path("outputs/adaptive_runs/next_trade_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "next_trade_results.csv", index=False)
    results_df.to_json(output_dir / "next_trade_results.json", orient='records', indent=2)
    
    print(f"  ✓ Results saved to {output_dir}")
    print()
    
    print("="*80)
    print("✅ COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Final summary
    if len(results_df) > 0:
        final = results_df.iloc[-1]
        print("Final Results:")
        print(f"  Baseline AUC:  {final['baseline_auc']:.4f} (predictability of baseline)")
        print(f"  Uniform AUC:   {final['uniform_auc']:.4f} (predictability of uniform)")
        print(f"  AUC Reduction: {final['auc_reduction']:.4f} ({final['auc_reduction_pct']:.1f}%)")
        print()
        print(f"Interpretation:")
        if final['uniform_auc'] < final['baseline_auc']:
            print(f"  ✓ Randomization REDUCED predictability by {final['auc_reduction_pct']:.1f}%")
        else:
            print(f"  ⚠️  Randomization did NOT reduce predictability")


if __name__ == "__main__":
    main()
