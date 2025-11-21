"""
P7 Adaptive Adversary Framework - Week 3 Pilot

Implements adaptive feedback loop:
1. Generate trades with current policy
2. Train adversary to predict trades
3. Measure predictability (AUC)
4. Adjust policy parameters based on AUC
5. Repeat until convergence or max iterations

Owner: P7
Week: 3
Target: Pilot on Uniform policy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import enrich_trades_for_adversary
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_trades


# === AUC THRESHOLDS (from Week 2 design) ===
AUC_HIGH_THRESHOLD = 0.75      # Too predictable → increase randomness
AUC_LOW_THRESHOLD = 0.55       # Too random → decrease randomness
AUC_TARGET_MIN = 0.60          # Target range lower bound
AUC_TARGET_MAX = 0.70          # Target range upper bound
AUC_TARGET_MID = 0.65          # Midpoint for nudging

# === ADJUSTMENT FACTORS ===
FACTOR_INCREASE = 1.20         # Multiply by 1.2 to increase stochasticity
FACTOR_DECREASE = 0.80         # Multiply by 0.8 to decrease stochasticity


def decide_adjustment(auc: float) -> tuple:
    """
    Decide parameter adjustment based on AUC score.
    
    Decision tree:
    - AUC > 0.75 → INCREASE randomness (too predictable)
    - AUC < 0.55 → DECREASE randomness (too random)
    - 0.60 ≤ AUC ≤ 0.70 → HOLD (in target)
    - Otherwise → NUDGE toward target
    
    Args:
        auc: Validation AUC score
    
    Returns:
        (action_name, multiplier)
    """
    if auc > AUC_HIGH_THRESHOLD:
        return 'INCREASE', FACTOR_INCREASE
    elif auc < AUC_LOW_THRESHOLD:
        return 'DECREASE', FACTOR_DECREASE
    elif AUC_TARGET_MIN <= auc <= AUC_TARGET_MAX:
        return 'HOLD', 1.0
    else:
        # Nudge toward target
        if auc > AUC_TARGET_MID:
            return 'NUDGE_UP', FACTOR_INCREASE
        else:
            return 'NUDGE_DOWN', FACTOR_DECREASE


def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Dict = None,
    max_iterations: int = 5,
    convergence_patience: int = 3,
    seed: int = 42
) -> Dict:
    """
    Main adaptive training loop.
    
    Args:
        prices_df: Price data
        initial_params: Starting policy parameters
        max_iterations: Maximum iterations
        convergence_patience: Iterations to stay in target for convergence
        seed: Random seed
    
    Returns:
        Dictionary with results
    """
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    # Initialize
    params = initial_params.copy()
    policy = UniformPolicy(params=params, seed=seed)
    results = []
    hold_count = 0
    
    print(f"\n{'='*70}")
    print(f"ADAPTIVE TRAINING LOOP")
    print(f"{'='*70}")
    print(f"Initial params: {params}")
    print(f"Max iterations: {max_iterations}")
    print(f"Convergence patience: {convergence_patience}")
    
    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*70}")
        print(f"Current params:")
        print(f"  price_noise = {params['price_noise']:.4f}")
        print(f"  time_noise = {params['time_noise_minutes']:.1f} min")
        
        # =====================================================================
        # STEP 1: Generate trades
        # =====================================================================
        print("\n[1/5] Generating trades...")
        trades = policy.generate_trades(prices_df)
        print(f"  → Generated {len(trades)} trades")
        
        if len(trades) == 0:
            print("  ERROR: No trades generated!")
            break
        
        # =====================================================================
        # STEP 2: Enrich for adversary
        # =====================================================================
        print("[2/5] Enriching trades...")
        enriched = enrich_trades_for_adversary(
            trades, 
            prices_df, 
            policy_id=f'uniform_iter{iteration+1}'
        )
        print(f"  → Enriched {len(enriched)} rows")
        
        # =====================================================================
        # STEP 3: Split data
        # =====================================================================
        print("[3/5] Splitting data (60% train, 20% val, 20% test)...")
        train, val, test = time_split_trades(enriched, train_ratio=0.6, val_ratio=0.2)
        print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        if len(val) < 50:
            print(f"  ERROR: Validation set too small ({len(val)} rows)")
            break
        
        # =====================================================================
        # STEP 4: Train adversary
        # =====================================================================
        print("[4/5] Training adversary...")
        adversary = P7AdaptiveAdversary(window_threshold_days=1.5)
        
        success = adversary.train(train, verbose=True)
        if not success:
            print("  ERROR: Adversary training failed!")
            break
        
        # =====================================================================
        # STEP 5: Evaluate on validation
        # =====================================================================
        print("[5/5] Evaluating on validation set...")
        auc = adversary.evaluate(val, verbose=True)
        
        # =====================================================================
        # Decision & logging
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"RESULT: AUC = {auc:.4f}")
        
        action, multiplier = decide_adjustment(auc)
        print(f"ACTION: {action} (multiplier = {multiplier:.2f})")
        
        # Log results
        results.append({
            'iteration': iteration + 1,
            'auc': float(auc),
            'action': action,
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            'multiplier': float(multiplier),
            'n_train': len(train),
            'n_val': len(val)
        })
        
        # =====================================================================
        # Check convergence
        # =====================================================================
        if action == 'HOLD':
            hold_count += 1
            print(f"✓ In target range ({hold_count}/{convergence_patience})")
            
            if hold_count >= convergence_patience:
                print(f"\n🎉 CONVERGED after {iteration + 1} iterations!")
                break
        else:
            hold_count = 0
            
            # Adjust parameters
            params['price_noise'] = np.clip(
                params['price_noise'] * multiplier,
                0.001,   # Min 0.1%
                0.20     # Max 20%
            )
            params['time_noise_minutes'] = np.clip(
                params['time_noise_minutes'] * multiplier,
                1,       # Min 1 minute
                180      # Max 3 hours
            )
            
            # Create new policy
            policy = UniformPolicy(params=params, seed=seed)
            
            print(f"→ New params: price_noise={params['price_noise']:.4f}, "
                  f"time_noise={params['time_noise_minutes']:.1f}min")
    
    # =========================================================================
    # Return results
    # =========================================================================
    return {
        'results': results,
        'final_params': params,
        'converged': hold_count >= convergence_patience,
        'n_iterations': len(results)
    }


def main():
    """Main entry point for Week 3 pilot"""
    print("="*70)
    print("P7 WEEK 3 PILOT: Adaptive Adversary on Uniform Policy")
    print("="*70)
    
    # Load data
    print("\n[Setup] Loading price data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"  ✓ Loaded {len(prices)} rows")
        print(f"  ✓ Symbols: {prices['symbol'].nunique()}")
        print(f"  ✓ Date range: {prices['date'].min()} to {prices['date'].max()}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return
    
    # Run adaptive loop
    print("\n[Execution] Running adaptive training loop...")
    try:
        output = adaptive_training_loop(
            prices,
            initial_params=DEFAULT_UNIFORM_PARAMS,
            max_iterations=5,
            convergence_patience=3,
            seed=42
        )
    except Exception as e:
        print(f"\n✗ ERROR in adaptive loop: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(output['results'])
    print(results_df[['iteration', 'auc', 'action', 'price_noise', 'time_noise_minutes']].to_string(index=False))
    
    print(f"\nFinal status:")
    print(f"  Converged: {output['converged']}")
    print(f"  Iterations: {output['n_iterations']}")
    print(f"  Final AUC: {output['results'][-1]['auc']:.4f}")
    print(f"  Final params: {output['final_params']}")
    
    # Save results
    print("\n[Output] Saving results...")
    output_dir = Path("outputs/adaptive_runs/uniform_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "adaptive_results.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'adaptive_results.csv'}")
    
    print("\n" + "="*70)
    print("✅ Week 3 pilot complete!")
    print("="*70)


if __name__ == "__main__":
    main()
