"""
P7 Week 3: Adaptive Adversary v1.0 - Pilot on Uniform Policy

Integrates:
- P4's UniformPolicy
- P6's Skeleton_Adversary_Model (extract_features, train_adversary_classifier, compute_auc)
- P3's backtest infrastructure
- P5's metrics utilities

This implements the adaptive feedback loop:
1. Generate trades with current policy parameters
2. Train adversary to predict trades
3. Measure predictability (AUC)
4. Adjust policy parameters based on AUC
5. Repeat until convergence

Owner: P7
Week: 3
Date: November 2025
Status: v1.0 - Pilot Implementation
"""

import sys
from pathlib import Path

# Add tests directory to path to import Skeleton_Adversary_Model
sys.path.insert(0, str(Path(__file__).parents[3] / 'tests'))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# P4 imports
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.utils import generate_policy_seed

# P6 imports (from tests/Skeleton_Adversary_Model.py)
from Skeleton_Adversary_Model import (
    extract_features,
    generate_labels,
    train_adversary_classifier,
    compute_auc,
    make_time_splits,
    ROLLING_WINDOWS
)

# P3 imports
from bsml.data.loader import load_prices

# Local bridge
from .bridge import enrich_trades_for_adversary


# =============================================================================
# CONSTANTS (from your Week 2 design)
# =============================================================================

AUC_HIGH_THRESHOLD = 0.75      # Too predictable
AUC_LOW_THRESHOLD = 0.55       # Too random
AUC_TARGET_MIN = 0.60          # Lower bound of optimal range
AUC_TARGET_MAX = 0.70          # Upper bound of optimal range
AUC_TARGET_MIDPOINT = 0.65     # Midpoint for nudging

ADJUSTMENT_FACTOR_INCREASE = 1.20  # Multiply by 1.2 to increase stochasticity
ADJUSTMENT_FACTOR_DECREASE = 0.80  # Multiply by 0.8 to decrease stochasticity

DEFAULT_CONVERGENCE_PATIENCE = 5   # Iterations to stay in target for convergence
DEFAULT_MAX_ITERATIONS = 20        # Maximum training iterations


# =============================================================================
# CORE ADJUSTMENT LOGIC (from your Week 2 design)
# =============================================================================

def decide_adjustment(auc_score: float) -> Tuple[str, float]:
    """
    Decide what adjustment action to take based on AUC score.
    
    This implements your Week 2 decision tree:
    - AUC > 0.75 → INCREASE randomness (too predictable)
    - AUC < 0.55 → DECREASE randomness (too random)
    - 0.60 ≤ AUC ≤ 0.70 → HOLD (in target range)
    - 0.65 < AUC < 0.75 → NUDGE_UP (slightly high)
    - 0.55 < AUC < 0.65 → NUDGE_DOWN (slightly low)
    
    Args:
        auc_score: Adversary AUC on validation fold [0.5, 1.0]
    
    Returns:
        (action, multiplier) tuple where:
        - action: str, one of ['INCREASE', 'DECREASE', 'HOLD', 'NUDGE_UP', 'NUDGE_DOWN']
        - multiplier: float, parameter adjustment factor (or 1.0 for HOLD)
    """
    if auc_score > AUC_HIGH_THRESHOLD:
        return 'INCREASE', ADJUSTMENT_FACTOR_INCREASE
    
    elif auc_score < AUC_LOW_THRESHOLD:
        return 'DECREASE', ADJUSTMENT_FACTOR_DECREASE
    
    elif AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
        return 'HOLD', 1.0
    
    else:
        # Between thresholds - nudge toward target
        if auc_score > AUC_TARGET_MIDPOINT:
            return 'NUDGE_UP', ADJUSTMENT_FACTOR_INCREASE
        else:
            return 'NUDGE_DOWN', ADJUSTMENT_FACTOR_DECREASE


# =============================================================================
# ADAPTIVE TRAINING LOOP
# =============================================================================

def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Main adaptive training loop for Week 3 pilot.
    
    This implements the full adaptive adversary feedback cycle:
    
    For each iteration:
        1. Generate trades with current policy parameters
        2. Enrich trades for adversary (bridge P4 → P6 format)
        3. Extract features (P6)
        4. Generate labels (P6)
        5. Train adversary classifier (P6)
        6. Compute AUC on validation set (P6)
        7. Decide parameter adjustment based on AUC (P7)
        8. Update policy parameters (P7)
        9. Check convergence
    
    Args:
        prices_df: Price data with columns [date, symbol, price]
        initial_params: Starting policy parameters (uses DEFAULT_UNIFORM_PARAMS if None)
        max_iterations: Maximum number of iterations
        convergence_patience: Number of iterations to stay in target range before converging
        seed: Master random seed for reproducibility
        verbose: Print detailed progress logs
    
    Returns:
        Dictionary with:
        - final_params: Final policy parameters
        - auc_history: List of AUC scores per iteration
        - param_history: List of parameter dicts per iteration
        - action_history: List of action dicts per iteration
        - converged: Boolean, whether loop converged
        - n_iterations: Number of iterations run
    """
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    # Initialize policy with starting parameters
    current_params = initial_params.copy()
    policy = UniformPolicy(params=current_params, seed=seed)
    
    # Tracking structures
    auc_history = []
    param_history = []
    action_history = []
    
    in_target_count = 0
    
    # Main adaptive loop
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            print(f"Current params: {current_params}")
        
        # =====================================================================
        # STEP 1: Generate trades with current policy
        # =====================================================================
        if verbose:
            print("\n[1/7] Generating trades...")
        
        trades_df = policy.generate_trades(prices_df)
        
        if verbose:
            print(f"  Generated {len(trades_df)} trades")
        
        if len(trades_df) == 0:
            print("  ERROR: No trades generated. Stopping.")
            break
        
        # =====================================================================
        # STEP 2: Enrich trades for adversary (CRITICAL BRIDGE STEP)
        # =====================================================================
        if verbose:
            print("\n[2/7] Enriching trades for adversary...")
        
        enriched_trades = enrich_trades_for_adversary(
            trades_df, 
            prices_df,
            policy_id=f'adaptive_uniform_iter{iteration+1}'
        )
        
        if verbose:
            print(f"  Enriched to {len(enriched_trades)} rows with {len(enriched_trades.columns)} columns")
        
        # =====================================================================
        # STEP 3: Extract features using P6's function
        # =====================================================================
        if verbose:
            print("\n[3/7] Extracting features...")
        
        features_df = extract_features(enriched_trades, rolling_windows=ROLLING_WINDOWS)
        
        # =====================================================================
        # STEP 4: Generate labels using P6's function
        # =====================================================================
        if verbose:
            print("\n[4/7] Generating labels...")
        
        labels = generate_labels(features_df, delta_steps=1)
        features_df['label'] = labels
        
        # Clean: drop rows with NaN labels or features
        features_df = features_df.dropna(subset=['label'])
        
        if len(features_df) < 100:
            print(f"  WARNING: Only {len(features_df)} valid samples after cleaning. Stopping.")
            break
        
        # =====================================================================
        # STEP 5: Time splits (train/val/test)
        # =====================================================================
        if verbose:
            print("\n[5/7] Creating time splits...")
        
        train_df, val_df, test_df = make_time_splits(
            features_df,
            train_end="2024-06-30",  # Adjust based on your data
            val_end="2024-09-30"
        )
        
        if verbose:
            print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Fallback: if val set too small, use test set
        if len(val_df) < 50:
            if verbose:
                print("  WARNING: Validation set too small. Using test set instead.")
            val_df = test_df
        
        if len(val_df) < 50:
            print("  ERROR: Not enough validation data. Stopping.")
            break
        
        # Select feature columns (exclude metadata)
        exclude = {
            "timestamp", "symbol", "policy_id", "label", "pnl", 
            "side", "qty", "ref_price", "date", "exec_flag",
            "action_side", "action_size", "is_market_order"
        }
        
        feature_cols = [
            c for c in features_df.columns
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        if verbose:
            print(f"  Using {len(feature_cols)} feature columns")
        
        # =====================================================================
        # STEP 6: Train adversary classifier
        # =====================================================================
        if verbose:
            print("\n[6/7] Training adversary...")
        
        X_train = train_df[feature_cols].fillna(0)  # Fill NaN with 0
        y_train = train_df['label'].values
        
        # Check label distribution
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print("  WARNING: Labels are all 0 or all 1. Cannot train classifier.")
            print("  Using dummy AUC = 0.50 (random guessing)")
            auc_score = 0.50
        else:
            # Train classifier
            model = train_adversary_classifier(
                features=X_train,
                labels=y_train,
                model='histgb',
                params={'max_depth': 6, 'learning_rate': 0.05, 'max_iter': 300}
            )
            
            # =================================================================
            # STEP 7: Compute AUC on validation set
            # =================================================================
            if verbose:
                print("\n[7/7] Computing validation AUC...")
            
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df['label'].values
            
            # Check validation label distribution
            if y_val.sum() == 0 or y_val.sum() == len(y_val):
                print("  WARNING: Validation labels are all 0 or all 1.")
                print("  Using dummy AUC = 0.50")
                auc_score = 0.50
            else:
                auc_score = compute_auc(model, X_val, y_val)
        
        if verbose:
            print(f"\n>>> Validation AUC: {auc_score:.4f}")
        
        # =====================================================================
        # Log iteration results
        # =====================================================================
        auc_history.append(auc_score)
        param_history.append(current_params.copy())
        
        # =====================================================================
        # STEP 8: Decide adjustment based on AUC
        # =====================================================================
        action, multiplier = decide_adjustment(auc_score)
        
        action_history.append({
            'iteration': iteration + 1,
            'action': action,
            'auc': auc_score,
            'multiplier': multiplier,
            'params': current_params.copy()
        })
        
        if verbose:
            print(f">>> Action: {action} (multiplier: {multiplier:.2f})")
        
        # =====================================================================
        # Check convergence
        # =====================================================================
        if action == 'HOLD':
            in_target_count += 1
            if verbose:
                print(f">>> In target range ({in_target_count}/{convergence_patience})")
            
            if in_target_count >= convergence_patience:
                if verbose:
                    print(f"\n🎉 CONVERGED after {iteration + 1} iterations!")
                break
        else:
            # Not in target - reset counter
            in_target_count = 0
            
            # ================================================================
            # STEP 9: Adjust policy parameters
            # ================================================================
            current_params['price_noise'] *= multiplier
            current_params['time_noise_minutes'] *= multiplier
            
            # Clamp to reasonable bounds (prevent extreme values)
            current_params['price_noise'] = np.clip(
                current_params['price_noise'], 
                0.001,  # Min: 0.1% noise
                0.20    # Max: 20% noise
            )
            current_params['time_noise_minutes'] = np.clip(
                current_params['time_noise_minutes'],
                1,      # Min: 1 minute
                180     # Max: 3 hours
            )
            
            # Recreate policy with new parameters
            policy = UniformPolicy(params=current_params, seed=seed)
            
            if verbose:
                print(f">>> New params: {current_params}")
    
    # =========================================================================
    # Return results
    # =========================================================================
    results = {
        'final_params': current_params,
        'auc_history': auc_history,
        'param_history': param_history,
        'action_history': action_history,
        'converged': in_target_count >= convergence_patience,
        'n_iterations': len(auc_history)
    }
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run Week 3 pilot: Adaptive Adversary on Uniform Policy
    
    This is the main entry point for testing the adaptive loop.
    """
    print("="*70)
    print("P7 WEEK 3 PILOT: Adaptive Adversary on Uniform Policy")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load price data
    print("\n[Setup] Loading price data...")
    try:
        prices_df = load_prices("data/ALL_backtest.csv")
        print(f"  Loaded {len(prices_df)} price rows")
        print(f"  Symbols: {prices_df['symbol'].nunique()}")
        print(f"  Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return
    
    # Run adaptive loop
    print("\n[Execution] Running adaptive training loop...")
    print(f"  Initial params: {DEFAULT_UNIFORM_PARAMS}")
    print(f"  Max iterations: 10")
    print(f"  Convergence patience: 3")
    
    try:
        results = adaptive_training_loop(
            prices_df,
            initial_params=DEFAULT_UNIFORM_PARAMS,
            max_iterations=10,  # Start with fewer iterations for testing
            convergence_patience=3,
            seed=42,
            verbose=True
        )
    except Exception as e:
        print(f"\n  ERROR in adaptive loop: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Final AUC: {results['auc_history'][-1]:.4f}")
    print(f"Final params: {results['final_params']}")
    print(f"\nAUC trajectory: {[f'{x:.3f}' for x in results['auc_history']]}")
    
    # Save results
    print("\n[Output] Saving results...")
    output_dir = Path("outputs/adaptive_runs/uniform_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save AUC trajectory
    auc_df = pd.DataFrame({
        'iteration': range(1, len(results['auc_history']) + 1),
        'auc': results['auc_history']
    })
    auc_path = output_dir / "auc_trajectory.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"  Saved: {auc_path}")
    
    # Save action history
    action_df = pd.DataFrame(results['action_history'])
    action_path = output_dir / "action_history.csv"
    action_df.to_csv(action_path, index=False)
    print(f"  Saved: {action_path}")
    
    # Save parameter history
    param_df = pd.DataFrame(results['param_history'])
    param_df.insert(0, 'iteration', range(1, len(param_df) + 1))
    param_path = output_dir / "param_history.csv"
    param_df.to_csv(param_path, index=False)
    print(f"  Saved: {param_path}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Week 3 pilot complete!")


if __name__ == "__main__":
    main()
```

---

## **NO MODIFICATIONS NEEDED TO EXISTING FILES! ✅**

All existing code stays exactly as is:
- ✅ `src/bsml/policies/uniform_policy.py` - **NO CHANGES**
- ✅ `src/bsml/policies/base_policy.py` - **NO CHANGES**
- ✅ `src/bsml/data/loader.py` - **NO CHANGES**
- ✅ `tests/Skeleton_Adversary_Model.py` - **NO CHANGES**
- ✅ All other P3, P4, P5, P6 code - **NO CHANGES**

---

## **DIRECTORY STRUCTURE AFTER ADDING FILES:**
```
repo/
├── src/bsml/
│   ├── adaptive/               # ⭐ NEW DIRECTORY
│   │   ├── __init__.py        # ⭐ NEW FILE
│   │   ├── bridge.py          # ⭐ NEW FILE
│   │   └── adaptive_loop_v1.py # ⭐ NEW FILE
│   ├── policies/
│   ├── data/
│   ├── analysis/
│   ├── core/
│   └── ...
├── tests/
│   └── Skeleton_Adversary_Model.py
├── data/
│   └── ALL_backtest.csv
└── outputs/
    └── adaptive_runs/         # Created automatically
        └── uniform_pilot/     # Created automatically
