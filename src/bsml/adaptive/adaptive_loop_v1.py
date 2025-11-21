"""P7 Week 3: Adaptive Adversary v1.0"""
import numpy as np
import pandas as pd
from pathlib import Path

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import enrich_trades_for_adversary
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_trades

AUC_HIGH = 0.75
AUC_LOW = 0.55
AUC_TARGET_MIN = 0.60
AUC_TARGET_MAX = 0.70

def decide_adjustment(auc):
    """Decide parameter adjustment based on AUC"""
    if auc > AUC_HIGH: 
        return 'INCREASE', 1.20
    elif auc < AUC_LOW: 
        return 'DECREASE', 0.80
    elif AUC_TARGET_MIN <= auc <= AUC_TARGET_MAX: 
        return 'HOLD', 1.0
    else: 
        return ('NUDGE_UP', 1.20) if auc > 0.65 else ('NUDGE_DOWN', 0.80)

def adaptive_training_loop(prices_df, max_iter=5):
    """Main adaptive loop"""
    params = DEFAULT_UNIFORM_PARAMS.copy()
    policy = UniformPolicy(params=params, seed=42)
    results = []
    
    for i in range(max_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}/{max_iter}")
        print(f"{'='*60}")
        print(f"Params: price_noise={params['price_noise']:.4f}, "
              f"time_noise={params['time_noise_minutes']:.1f}min")
        
        # Generate trades
        print("\n[1/4] Generating trades...")
        trades = policy.generate_trades(prices_df)
        print(f"  Generated {len(trades)} raw trades")
        
        # Enrich for adversary
        print("[2/4] Enriching trades...")
        enriched = enrich_trades_for_adversary(trades, prices_df)
        print(f"  Enriched to {len(enriched)} rows")
        
        # Split data
        print("[3/4] Splitting data...")
        train, val, test = time_split_trades(enriched, train_ratio=0.6, val_ratio=0.2)
        
        if len(val) < 50:
            print(f"  ERROR: Only {len(val)} validation samples. Need at least 50.")
            break
        
        # Train and evaluate adversary
        print("[4/4] Training adversary...")
        adversary = P7AdaptiveAdversary(window_size=1)
        adversary.train(train)
        
        if adversary.model is None:
            print("  ERROR: Adversary failed to train. Stopping.")
            break
        
        auc = adversary.evaluate(val)
        
        # Decision
        print(f"\n>>> RESULT: AUC = {auc:.4f}")
        action, mult = decide_adjustment(auc)
        print(f">>> ACTION: {action} (multiplier: {mult:.2f})")
        
        results.append({
            'iter': i+1, 
            'auc': auc, 
            'action': action,
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes'])
        })
        
        # Adjust parameters
        if action != 'HOLD':
            params['price_noise'] = np.clip(
                params['price_noise'] * mult, 
                0.001, 0.20
            )
            params['time_noise_minutes'] = np.clip(
                params['time_noise_minutes'] * mult, 
                1, 180
            )
            policy = UniformPolicy(params=params, seed=42)
            print(f">>> NEW PARAMS: price_noise={params['price_noise']:.4f}, "
                  f"time_noise={params['time_noise_minutes']:.1f}min")
        else:
            print(f">>> HOLDING at current parameters")
    
    return results

def main():
    """Main entry point"""
    print("="*60)
    print("P7 WEEK 3 PILOT: Adaptive Adversary")
    print("="*60)
    
    # Load data
    print("\n[Setup] Loading price data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  Loaded {len(prices)} price rows")
    print(f"  Symbols: {prices['symbol'].nunique()}")
    print(f"  Date range: {prices['date'].min()} to {prices['date'].max()}")
    
    # Run adaptive loop
    print("\n[Execution] Starting adaptive training loop...")
    results = adaptive_training_loop(prices, max_iter=5)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    print("\n[Output] Saving results...")
    output_dir = Path("outputs/adaptive_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "results.csv", index=False)
    print(f"  Saved to: {output_dir / 'results.csv'}")
    
    print("\n Week 3 pilot complete!")

if __name__ == "__main__":
    main()
