"""
P7 Week 4: Master Adversary Training Script

Compares all randomization policies (Uniform, Pink, OU) against
the adaptive adversary classifier.

Outputs comparison table to results/paper/tables/adaptive_adversary.csv

Owner: P7
Week: 4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from bsml.data.loader import load_prices
from bsml.adaptive.adaptive_loop_v2 import (
    adaptive_training_loop,
    AdaptiveConfig,
    POLICY_REGISTRY
)


def compare_policies(
    prices_df: pd.DataFrame,
    policies: list = None,
    config: AdaptiveConfig = None,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run adaptive training on multiple policies and compare results
    
    Args:
        prices_df: Price data
        policies: List of policy names (default: all)
        config: AdaptiveConfig
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Comparison DataFrame
    """
    if policies is None:
        policies = list(POLICY_REGISTRY.keys())
    
    if config is None:
        config = AdaptiveConfig()
    
    results = []
    
    for policy_name in policies:
        print("\n" + "="*80)
        print(f"TESTING POLICY: {policy_name.upper()}")
        print("="*80)
        
        try:
            result = adaptive_training_loop(
                prices_df,
                policy_name=policy_name,
                config=config,
                seed=seed,
                verbose=verbose
            )
            
            logger = result['logger']
            
            if logger.iterations:
                # Get initial and final AUC
                initial_auc = logger.iterations[0]['auc_val']
                final_auc = logger.iterations[-1]['auc_val']
                auc_improvement = final_auc - initial_auc
                
                # Calculate AUC stability (std of last 3 iterations)
                last_aucs = [x['auc_val'] for x in logger.iterations[-3:]]
                auc_stability = np.std(last_aucs) if len(last_aucs) > 1 else 0.0
                
                # Check if in target range
                in_target = (config.AUC_TARGET_MIN <= final_auc <= config.AUC_TARGET_MAX)
                
                results.append({
                    'policy': policy_name,
                    'display_name': POLICY_REGISTRY[policy_name]['display_name'],
                    'description': POLICY_REGISTRY[policy_name]['description'],
                    'initial_auc': initial_auc,
                    'final_auc': final_auc,
                    'auc_improvement': auc_improvement,
                    'auc_stability': auc_stability,
                    'converged': result['converged'],
                    'n_iterations': result['n_iterations'],
                    'in_target_range': in_target,
                    'final_params': str(result['final_params']),
                })
            
            # Save individual results
            output_dir = config.OUTPUT_DIR / policy_name
            logger.save_results(output_dir)
        
        except Exception as e:
            print(f"\n✗ FAILED for {policy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\n✗ No policies completed successfully")
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(results)
    
    # Sort by final AUC (closer to target midpoint is better)
    target_mid = config.AUC_TARGET_MID
    comparison_df['distance_from_target'] = np.abs(comparison_df['final_auc'] - target_mid)
    comparison_df = comparison_df.sort_values('distance_from_target')
    
    return comparison_df


def main():
    """Main entry point for Week 4 testing"""
    print("="*80)
    print("P7 WEEK 4: POLICY COMPARISON - ADAPTIVE ADVERSARY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTesting policies: Uniform, Pink Noise, Ornstein-Uhlenbeck")
    
    config = AdaptiveConfig()
    
    print("\n[1/3] Loading data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return
    
    print("\n[2/3] Running policy comparison...")
    try:
        comparison_df = compare_policies(
            prices,
            policies=['uniform', 'pink', 'ou'],
            config=config,
            seed=42,
            verbose=True
        )
    except Exception as e:
        print(f"\n✗ FATAL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if comparison_df.empty:
        print("\n✗ No results to save")
        return
    
    print("\n[3/3] Saving comparison table...")
    try:
        # Save to paper tables directory
        table_dir = Path("results/paper/tables")
        table_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = table_dir / "adaptive_adversary.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path}")
        
        # Also save formatted version
        display_cols = [
            'display_name', 'initial_auc', 'final_auc', 'auc_improvement',
            'converged', 'n_iterations', 'in_target_range'
        ]
        print("\n" + "="*80)
        print("POLICY COMPARISON RESULTS")
        print("="*80)
        print(comparison_df[display_cols].to_string(index=False, float_format='%.4f'))
        
        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        
        best_policy = comparison_df.iloc[0]
        print(f"\n✓ Best Policy: {best_policy['display_name']}")
        print(f"  Final AUC: {best_policy['final_auc']:.4f}")
        print(f"  Converged: {best_policy['converged']}")
        print(f"  In Target: {best_policy['in_target_range']}")
        
        if best_policy['final_auc'] > 0.70:
            print("\n⚠️  Even best policy is highly predictable")
            print("   Recommendation: Need more aggressive randomization techniques")
        elif best_policy['final_auc'] < 0.55:
            print("\n⚠️  Strategies too random")
            print("   Recommendation: Reduce randomization intensity")
        else:
            print("\n✓ Achieved good unpredictability balance")
    
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ WEEK 4 TESTING COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
