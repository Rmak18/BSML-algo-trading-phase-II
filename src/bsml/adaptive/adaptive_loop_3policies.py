"""
P7 Adaptive Loop - 3-Policy Comparison (Next-Trade Prediction)

Compare predictability of Baseline vs 3 randomization strategies:
- Uniform (independent random noise)
- Pink Noise (1/f correlated noise - persistent drifts)
- Ornstein-Uhlenbeck (mean-reverting noise)

Goal: Identify which randomization best evades adversarial pattern detection.

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices

from .bridge_3policies import prepare_three_policy_data, time_split_data
from .next_trade_predictor import NextTradePredictor


@dataclass
class ThreePolicyConfig:
    """Configuration for 3-policy comparison experiment"""
    lookback: int = 5
    max_iterations: int = 10
    
    # Uniform noise parameters
    initial_price_noise: float = 0.03
    initial_time_noise: float = 30.0
    
    # Pink Noise parameters
    pink_alpha: float = 1.0  # 1/f^alpha, alpha=1 is classic pink noise
    pink_price_scale: float = 0.04
    
    # OU parameters
    ou_theta: float = 0.15  # Mean reversion speed
    ou_sigma: float = 0.02  # Volatility
    ou_price_scale: float = 1.0
    
    # Adjustment strategy
    noise_increase_rate: float = 1.25
    target_auc_high: float = 0.65  # If above this, still too predictable
    target_auc_low: float = 0.55   # If below this, achieved target
    convergence_patience: int = 3


def compare_three_policies(
    prices: pd.DataFrame,
    price_noise: float,
    time_noise: float,
    config: ThreePolicyConfig,
    verbose: bool = True
) -> Dict:
    """
    Train predictor on baseline, test on all 3 randomization policies.
    
    Args:
        prices: Market price data
        price_noise: Price noise level for Uniform
        time_noise: Time noise level for Uniform
        config: Configuration with policy-specific parameters
        verbose: Print detailed output
    
    Returns:
        {
            'baseline': {...},
            'uniform': {...},
            'pink_noise': {...},
            'ou': {...},
            'comparison': {
                'baseline_auc': float,
                'uniform_auc': float,
                'pink_auc': float,
                'ou_auc': float,
                'best_policy': str,
                'ranking': [...]
            }
        }
    """
    
    # =========================================================================
    # CREATE POLICIES
    # =========================================================================
    
    # Uniform Policy
    uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
    uniform_params['price_noise'] = price_noise
    uniform_params['time_noise_minutes'] = time_noise
    uniform_policy = UniformPolicy(params=uniform_params)
    
    # Pink Noise Policy (independent - doesn't use price_noise/time_noise)
    pink_policy = PinkNoisePolicy(
        alpha=config.pink_alpha,
        price_scale=config.pink_price_scale,
        seed=42
    )
    
    # OU Policy (independent - doesn't use price_noise/time_noise)
    ou_policy = OUPolicy(
        theta=config.ou_theta,
        sigma=config.ou_sigma,
        price_scale=config.ou_price_scale,
        seed=42
    )
    
    if verbose:
        print(f"[3-Policy] Generating datasets (lookback={config.lookback})...")
        print(f"  Uniform: price_noise={price_noise:.4f}, time_noise={time_noise:.1f}min")
        print(f"  Pink Noise: alpha={config.pink_alpha}, price_scale={config.pink_price_scale}")
        print(f"  OU: theta={config.ou_theta}, sigma={config.ou_sigma}, price_scale={config.ou_price_scale}")
    
    # Prepare all datasets
    datasets = prepare_three_policy_data(
        prices,
        baseline_generate,
        uniform_policy,
        pink_policy,
        ou_policy,
        lookback=config.lookback,
        verbose=verbose
    )
    
    baseline_data = datasets['baseline']
    
    if len(baseline_data) == 0:
        return {
            'success': False,
            'reason': 'insufficient_baseline_data'
        }
    
    results = {}
    
    # =========================================================================
    # TRAIN PREDICTOR ON BASELINE ONLY
    # =========================================================================
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING PREDICTOR ON BASELINE DATA")
        print("="*80)
    
    baseline_train, baseline_val, baseline_test = time_split_data(baseline_data)
    
    if verbose:
        print(f"[Training] Train: {len(baseline_train)}, Val: {len(baseline_val)}, Test: {len(baseline_test)}")
    
    predictor = NextTradePredictor()
    train_metrics = predictor.train(baseline_train, verbose=verbose)
    
    if not train_metrics['success']:
        return {
            'success': False,
            'reason': 'training_failed'
        }
    
    # Test on baseline
    if verbose:
        print("\n" + "="*80)
        print("TESTING ON BASELINE POLICY")
        print("="*80)
    
    baseline_val_metrics = predictor.evaluate(baseline_val, verbose=verbose)
    results['baseline'] = {
        'auc': baseline_val_metrics['auc'],
        'n_samples': len(baseline_data)
    }
    
    # =========================================================================
    # TEST ON ALL 3 RANDOMIZATION POLICIES
    # =========================================================================
    
    policy_aucs = {'baseline': baseline_val_metrics['auc']}
    
    for policy_name in ['uniform', 'pink_noise', 'ou']:
        policy_data = datasets[policy_name]
        
        if verbose:
            print("\n" + "="*80)
            print(f"TESTING ON {policy_name.upper().replace('_', ' ')} POLICY")
            print("="*80)
        
        if len(policy_data) == 0:
            if verbose:
                print(f"[Warning] No {policy_name} data available")
            policy_auc = 0.50
        else:
            _, policy_val, _ = time_split_data(policy_data)
            
            if verbose:
                print(f"[Testing] Using {policy_name} validation set: {len(policy_val)} samples")
            
            policy_val_metrics = predictor.evaluate(policy_val, verbose=verbose)
            policy_auc = policy_val_metrics['auc']
            
            results[policy_name] = {
                'auc': policy_auc,
                'n_samples': len(policy_data)
            }
        
        policy_aucs[policy_name] = policy_auc
    
    # =========================================================================
    # COMPARISON & RANKING
    # =========================================================================
    
    baseline_auc = policy_aucs['baseline']
    uniform_auc = policy_aucs['uniform']
    pink_auc = policy_aucs['pink_noise']
    ou_auc = policy_aucs['ou']
    
    # Rank policies by AUC (lower = better)
    randomization_aucs = {
        'uniform': uniform_auc,
        'pink_noise': pink_auc,
        'ou': ou_auc
    }
    ranking = sorted(randomization_aucs.keys(), key=lambda k: randomization_aucs[k])
    best_policy = ranking[0]
    
    # Calculate reductions
    uniform_reduction = baseline_auc - uniform_auc
    pink_reduction = baseline_auc - pink_auc
    ou_reduction = baseline_auc - ou_auc
    
    if verbose:
        print("\n" + "="*80)
        print("3-POLICY COMPARISON")
        print("="*80)
        print(f"Baseline AUC:   {baseline_auc:.4f} (highly predictable)")
        print(f"Uniform AUC:    {uniform_auc:.4f} (reduction: {uniform_reduction:.4f} = {uniform_reduction/baseline_auc*100:.1f}%)")
        print(f"Pink Noise AUC: {pink_auc:.4f} (reduction: {pink_reduction:.4f} = {pink_reduction/baseline_auc*100:.1f}%)")
        print(f"OU AUC:         {ou_auc:.4f} (reduction: {ou_reduction:.4f} = {ou_reduction/baseline_auc*100:.1f}%)")
        print()
        print(f"Ranking (best to worst):")
        for i, policy in enumerate(ranking, 1):
            auc = randomization_aucs[policy]
            reduction = baseline_auc - auc
            print(f"  {i}. {policy.replace('_', ' ').title()}: AUC={auc:.4f} (↓ {reduction:.4f})")
        print()
        print(f"🏆 WINNER: {best_policy.replace('_', ' ').title()} (lowest AUC = least predictable)")
    
    results['comparison'] = {
        'baseline_auc': float(baseline_auc),
        'uniform_auc': float(uniform_auc),
        'pink_auc': float(pink_auc),
        'ou_auc': float(ou_auc),
        'uniform_reduction': float(uniform_reduction),
        'pink_reduction': float(pink_reduction),
        'ou_reduction': float(ou_reduction),
        'best_policy': best_policy,
        'ranking': ranking
    }
    
    results['success'] = True
    
    return results


def adaptive_three_policy_loop(
    prices: pd.DataFrame,
    config: ThreePolicyConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Adaptive loop: Compare all 3 policies, adjust Uniform noise to find sweet spot.
    
    Note: Pink Noise and OU parameters are fixed, only Uniform parameters adapt.
    
    Goal: Find Uniform noise level where all policies achieve target unpredictability.
    """
    
    if verbose:
        print("="*80)
        print("P7 ADAPTIVE 3-POLICY COMPARISON")
        print("="*80)
        print(f"Task: Compare Baseline vs Uniform vs Pink Noise vs OU")
        print(f"Goal: Find noise level where policies are unpredictable (AUC < {config.target_auc_low})")
        print(f"Max iterations: {config.max_iterations}")
        print(f"Initial Uniform params: price_noise={config.initial_price_noise:.4f}, time_noise={config.initial_time_noise:.1f}min")
        print(f"Fixed Pink params: alpha={config.pink_alpha}, price_scale={config.pink_price_scale}")
        print(f"Fixed OU params: theta={config.ou_theta}, sigma={config.ou_sigma}, price_scale={config.ou_price_scale}")
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
            print(f"Testing Uniform: price_noise={price_noise:.4f}, time_noise={time_noise:.1f}min")
            print()
        
        # Run comparison
        results = compare_three_policies(
            prices, price_noise, time_noise, config, verbose=verbose
        )
        
        if not results['success']:
            if verbose:
                print(f"✗ Iteration {iteration} failed: {results.get('reason', 'unknown')}")
            continue
        
        comp = results['comparison']
        baseline_auc = comp['baseline_auc']
        uniform_auc = comp['uniform_auc']
        pink_auc = comp['pink_auc']
        ou_auc = comp['ou_auc']
        best_policy = comp['best_policy']
        
        # Get best (lowest) AUC among randomization policies
        best_auc = min(uniform_auc, pink_auc, ou_auc)
        
        # Decide action (FIXED LOGIC!)
        if best_auc > config.target_auc_high:
            action = "INCREASE"
            reason = f"Policies still too predictable (best AUC={best_auc:.3f}), increase noise"
            multiplier = config.noise_increase_rate
            in_target_count = 0
        elif best_auc < config.target_auc_low:
            action = "SUCCESS"
            reason = f"Achieved target unpredictability (best AUC={best_auc:.3f})"
            multiplier = 1.0
            in_target_count += 1
        else:
            action = "HOLD"
            reason = f"In target range (best AUC={best_auc:.3f})"
            multiplier = 1.0
            in_target_count += 1
        
        if verbose:
            print("\n" + "-"*80)
            print(f"RESULT: Best AUC = {best_auc:.4f} ({best_policy.replace('_', ' ').title()})")
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
            'pink_auc': pink_auc,
            'ou_auc': ou_auc,
            'best_policy': best_policy,
            'best_auc': best_auc,
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
        
        # Adjust Uniform parameters only
        if action == "INCREASE":
            price_noise *= multiplier
            time_noise *= multiplier
            
            if verbose:
                print(f"\nAdjustment (Uniform only):")
                print(f"  price_noise: {price_noise:.4f}")
                print(f"  time_noise: {time_noise:.1f} min")
    
    # Summary
    results_df = pd.DataFrame(iterations)
    
    if verbose:
        print("\n" + "="*80)
        print("ITERATION SUMMARY")
        print("="*80)
        print(results_df[['iteration', 'baseline_auc', 'uniform_auc', 'pink_auc', 'ou_auc', 'best_policy', 'action']].to_string(index=False))
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Iterations: {len(iterations)}")
        
        if len(iterations) > 0:
            final = iterations[-1]
            print(f"Final Baseline AUC: {final['baseline_auc']:.4f}")
            print(f"Final Uniform AUC: {final['uniform_auc']:.4f}")
            print(f"Final Pink AUC: {final['pink_auc']:.4f}")
            print(f"Final OU AUC: {final['ou_auc']:.4f}")
            print(f"Winner: {final['best_policy'].replace('_', ' ').title()}")
            print(f"Final Uniform params: price_noise={final['price_noise']:.4f}, time_noise={final['time_noise_minutes']:.1f}min")
            
            if final['best_auc'] < config.target_auc_low:
                print("\n✓ SUCCESS: Achieved target unpredictability")
            elif final['best_auc'] > config.target_auc_high:
                print("\n⚠️  Policies still too predictable, need more noise")
            else:
                print("\n✓ Good: In target range")
    
    return results_df


def main():
    """Main entry point"""
    
    print("="*80)
    print("P7 WEEK 3: 3-POLICY COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Task: Compare predictability across randomization strategies")
    print("Policies: Uniform, Pink Noise (1/f), Ornstein-Uhlenbeck (mean-reverting)")
    print("Method: Train predictor on baseline, test on all 3 policies")
    print("Goal: Identify which randomization best evades detection")
    print()
    
    # Load data
    print("[1/3] Loading data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    print()
    
    # Configure experiment
    config = ThreePolicyConfig(
        lookback=5,
        max_iterations=10,
        
        # Uniform params (will adapt)
        initial_price_noise=0.03,
        initial_time_noise=30.0,
        
        # Pink params (fixed)
        pink_alpha=1.0,  # Classic 1/f pink noise
        pink_price_scale=0.04,
        
        # OU params (fixed)
        ou_theta=0.15,
        ou_sigma=0.02,
        ou_price_scale=1.0,
        
        # Convergence
        target_auc_high=0.65,
        target_auc_low=0.55,
        convergence_patience=3
    )
    
    # Run adaptive loop
    print("[2/3] Running adaptive 3-policy comparison...")
    print()
    
    results_df = adaptive_three_policy_loop(prices, config, verbose=True)
    
    # Save results
    print("\n[3/3] Saving results...")
    output_dir = Path("outputs/adaptive_runs/3policies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "3policy_results.csv", index=False)
    results_df.to_json(output_dir / "3policy_results.json", orient='records', indent=2)
    
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
        print(f"  Baseline AUC: {final['baseline_auc']:.4f} (predictable baseline)")
        print(f"  Uniform AUC:  {final['uniform_auc']:.4f} (independent noise)")
        print(f"  Pink AUC:     {final['pink_auc']:.4f} (1/f noise - persistent drifts)")
        print(f"  OU AUC:       {final['ou_auc']:.4f} (mean-reverting noise)")
        print()
        print(f"🏆 Winner: {final['best_policy'].replace('_', ' ').title()}")
        print(f"   (Best AUC = {final['best_auc']:.4f}, lowest = hardest to predict)")


if __name__ == "__main__":
    main()
