"""
P7 Adaptive Adversary Framework - Week 3 Production Version

Advanced adaptive training loop with:
- 10-minute prediction window
- Comprehensive logging and metrics
- Convergence tracking
- Multiple adjustment strategies
- Results visualization preparation

Owner: P7
Week: 3
Date: November 2025
Status: Production v1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import enrich_trades_for_adversary
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_trades


# =============================================================================
# CONFIGURATION
# =============================================================================

class AdaptiveConfig:
    """Configuration for adaptive training loop"""
    
    # AUC thresholds (from Week 2 design)
    AUC_HIGH_THRESHOLD = 0.75      # Too predictable
    AUC_LOW_THRESHOLD = 0.55       # Too random
    AUC_TARGET_MIN = 0.60          # Target range lower
    AUC_TARGET_MAX = 0.70          # Target range upper
    AUC_TARGET_MID = 0.65          # Midpoint for nudging
    
    # Adjustment factors
    FACTOR_INCREASE = 1.20         # Increase stochasticity
    FACTOR_DECREASE = 0.80         # Decrease stochasticity
    FACTOR_NUDGE = 1.10            # Smaller nudge adjustment
    
    # Loop parameters
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    # Adversary parameters
    PREDICTION_WINDOW_MINUTES = 10
    USE_SMOTE = True
    USE_CV = True
    N_CV_FOLDS = 3
    
    # Data split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Parameter bounds (safety constraints)
    PRICE_NOISE_MIN = 0.001        # 0.1%
    PRICE_NOISE_MAX = 0.25         # 25%
    TIME_NOISE_MIN = 1             # 1 minute
    TIME_NOISE_MAX = 240           # 4 hours
    
    # Output
    OUTPUT_DIR = Path("outputs/adaptive_runs/uniform_pilot")
    SAVE_DETAILED_LOGS = True


# =============================================================================
# DECISION LOGIC
# =============================================================================

def decide_adjustment(auc: float, config: AdaptiveConfig = None) -> tuple:
    """
    Decide parameter adjustment based on AUC score.
    
    Enhanced decision logic with multiple adjustment levels.
    
    Args:
        auc: Validation AUC score
        config: Configuration object
    
    Returns:
        (action, multiplier, reason)
    """
    if config is None:
        config = AdaptiveConfig()
    
    if auc > config.AUC_HIGH_THRESHOLD:
        return 'INCREASE', config.FACTOR_INCREASE, 'Too predictable (AUC > 0.75)'
    
    elif auc < config.AUC_LOW_THRESHOLD:
        return 'DECREASE', config.FACTOR_DECREASE, 'Too random (AUC < 0.55)'
    
    elif config.AUC_TARGET_MIN <= auc <= config.AUC_TARGET_MAX:
        return 'HOLD', 1.0, 'In target range (0.60-0.70)'
    
    elif auc > config.AUC_TARGET_MAX:
        # Slightly above target
        return 'NUDGE_UP', config.FACTOR_NUDGE, 'Slightly high (0.70-0.75)'
    
    else:
        # Slightly below target
        return 'NUDGE_DOWN', 1.0 / config.FACTOR_NUDGE, 'Slightly low (0.55-0.60)'


def adjust_parameters(
    params: Dict, 
    multiplier: float, 
    config: AdaptiveConfig = None
) -> Dict:
    """
    Adjust policy parameters with safety bounds.
    
    Args:
        params: Current parameters
        multiplier: Adjustment multiplier
        config: Configuration object
    
    Returns:
        New parameters (clipped to bounds)
    """
    if config is None:
        config = AdaptiveConfig()
    
    new_params = params.copy()
    
    # Apply multiplier
    new_params['price_noise'] *= multiplier
    new_params['time_noise_minutes'] *= multiplier
    
    # Clip to safety bounds
    new_params['price_noise'] = np.clip(
        new_params['price_noise'],
        config.PRICE_NOISE_MIN,
        config.PRICE_NOISE_MAX
    )
    new_params['time_noise_minutes'] = np.clip(
        new_params['time_noise_minutes'],
        config.TIME_NOISE_MIN,
        config.TIME_NOISE_MAX
    )
    
    return new_params


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class IterationLogger:
    """Track metrics across iterations"""
    
    def __init__(self):
        self.iterations = []
        self.start_time = datetime.now()
    
    def log_iteration(
        self,
        iteration: int,
        params: Dict,
        auc: float,
        action: str,
        multiplier: float,
        reason: str,
        train_metrics: Dict,
        val_metrics: Dict,
        cv_scores: List[float] = None
    ):
        """Log complete iteration metrics"""
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            
            # Parameters
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            
            # AUC metrics
            'auc_val': float(auc),
            'auc_cv_mean': float(np.mean(cv_scores)) if cv_scores else None,
            'auc_cv_std': float(np.std(cv_scores)) if cv_scores else None,
            
            # Decision
            'action': action,
            'multiplier': float(multiplier),
            'reason': reason,
            
            # Training details
            'n_train': train_metrics.get('n_samples', 0),
            'n_val': val_metrics.get('n_samples', 0),
            'n_features': train_metrics.get('n_features', 0),
            
            # Label distributions
            'train_pos_rate': (
                train_metrics['label_distribution']['positive'] / 
                train_metrics['n_samples'] * 100
                if train_metrics.get('n_samples') else 0
            ),
            'val_pos_rate': (
                val_metrics['label_distribution']['positive'] / 
                val_metrics['n_samples'] * 100
                if val_metrics.get('n_samples') else 0
            ),
            
            # Confusion matrix (if available)
            'confusion_matrix': val_metrics.get('confusion_matrix'),
        }
        
        self.iterations.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert logs to DataFrame"""
        return pd.DataFrame(self.iterations)
    
    def print_summary(self):
        """Print summary table"""
        df = self.to_dataframe()
        
        summary_cols = [
            'iteration', 'auc_val', 'action', 
            'price_noise', 'time_noise_minutes'
        ]
        
        print("\n" + "="*80)
        print("ITERATION SUMMARY")
        print("="*80)
        print(df[summary_cols].to_string(index=False, float_format='%.4f'))
    
    def save_results(self, output_dir: Path):
        """Save all results to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        df = self.to_dataframe()
        df.to_csv(output_dir / "adaptive_results.csv", index=False)
        
        # Save detailed JSON
        with open(output_dir / "adaptive_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        # Save summary stats
        summary = {
            'total_iterations': len(self.iterations),
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'final_auc': float(self.iterations[-1]['auc_val']),
            'final_params': {
                'price_noise': float(self.iterations[-1]['price_noise']),
                'time_noise_minutes': float(self.iterations[-1]['time_noise_minutes'])
            },
            'auc_trajectory': [float(x['auc_val']) for x in self.iterations],
            'actions': [x['action'] for x in self.iterations]
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_dir}")


# =============================================================================
# MAIN ADAPTIVE TRAINING LOOP
# =============================================================================

def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfig] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Production-grade adaptive training loop.
    
    Features:
    - 10-minute prediction window
    - SMOTE resampling for imbalanced data
    - Cross-validation during training
    - Comprehensive logging
    - Multiple adjustment strategies
    - Convergence detection
    
    Args:
        prices_df: Price data
        initial_params: Starting policy parameters
        config: Configuration object
        seed: Random seed
        verbose: Print detailed logs
    
    Returns:
        Dictionary with logger and final results
    """
    # Initialize
    if config is None:
        config = AdaptiveConfig()
    
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    params = initial_params.copy()
    policy = UniformPolicy(params=params, seed=seed)
    logger = IterationLogger()
    
    hold_count = 0
    converged = False
    
    # Print header
    if verbose:
        print("\n" + "="*80)
        print("P7 ADAPTIVE ADVERSARY TRAINING LOOP")
        print("="*80)
        print(f"Configuration:")
        print(f"  Prediction window: {config.PREDICTION_WINDOW_MINUTES} minutes")
        print(f"  Max iterations: {config.MAX_ITERATIONS}")
        print(f"  Convergence patience: {config.CONVERGENCE_PATIENCE}")
        print(f"  AUC target: [{config.AUC_TARGET_MIN}, {config.AUC_TARGET_MAX}]")
        print(f"\nInitial parameters:")
        print(f"  price_noise: {params['price_noise']:.4f}")
        print(f"  time_noise: {params['time_noise_minutes']:.1f} min")
    
    # Main loop
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
            print(f"Parameters: price_noise={params['price_noise']:.4f}, "
                  f"time_noise={params['time_noise_minutes']:.1f}min")
        
        # =====================================================================
        # STEP 1: Generate trades
        # =====================================================================
        if verbose:
            print("\n[1/6] Generating trades...")
        
        try:
            trades = policy.generate_trades(prices_df)
            if verbose:
                print(f"  → {len(trades)} trades generated")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            break
        
        if len(trades) == 0:
            print("  ✗ No trades generated!")
            break
        
        # =====================================================================
        # STEP 2: Enrich trades
        # =====================================================================
        if verbose:
            print("[2/6] Enriching trades...")
        
        try:
            enriched = enrich_trades_for_adversary(
                trades, 
                prices_df,
                policy_id=f'uniform_iter{iter_num}'
            )
            if verbose:
                print(f"  → {len(enriched)} rows enriched")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            break
        
        # =====================================================================
        # STEP 3: Split data
        # =====================================================================
        if verbose:
            print("[3/6] Splitting data...")
        
        try:
            train, val, test = time_split_trades(
                enriched,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO
            )
            if verbose:
                print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            break
        
        # Validate split sizes
        if len(val) < config.MIN_VAL_SAMPLES:
            print(f"  ✗ Validation set too small ({len(val)} < {config.MIN_VAL_SAMPLES})")
            break
        
        # =====================================================================
        # STEP 4: Train adversary
        # =====================================================================
        if verbose:
            print("[4/6] Training adversary...")
        
        try:
            adversary = P7AdaptiveAdversary(
                window_threshold_minutes=config.PREDICTION_WINDOW_MINUTES,
                use_smote=config.USE_SMOTE,
                use_cv=config.USE_CV,
                n_cv_folds=config.N_CV_FOLDS,
                random_state=seed
            )
            
            train_metrics = adversary.train(train, verbose=verbose)
            
            if not train_metrics.get('success', False):
                print(f"  ✗ Training failed: {train_metrics.get('reason', 'unknown')}")
                break
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # =====================================================================
        # STEP 5: Evaluate on validation
        # =====================================================================
        if verbose:
            print("[5/6] Evaluating on validation set...")
        
        try:
            val_metrics = adversary.evaluate(val, verbose=verbose)
            
            if not val_metrics.get('success', False):
                print(f"  ✗ Evaluation failed: {val_metrics.get('reason', 'unknown')}")
                break
            
            auc = val_metrics['auc']
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # =====================================================================
        # STEP 6: Decision and adjustment
        # =====================================================================
        if verbose:
            print("[6/6] Making adjustment decision...")
        
        action, multiplier, reason = decide_adjustment(auc, config)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"RESULT: AUC = {auc:.4f}")
            print(f"ACTION: {action}")
            print(f"REASON: {reason}")
            print(f"MULTIPLIER: {multiplier:.2f}")
        
        # Log iteration
        logger.log_iteration(
            iteration=iter_num,
            params=params,
            auc=auc,
            action=action,
            multiplier=multiplier,
            reason=reason,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            cv_scores=adversary.cv_scores if hasattr(adversary, 'cv_scores') else None
        )
        
        # Check convergence
        if action == 'HOLD':
            hold_count += 1
            if verbose:
                print(f"✓ In target range ({hold_count}/{config.CONVERGENCE_PATIENCE})")
            
            if hold_count >= config.CONVERGENCE_PATIENCE:
                converged = True
                if verbose:
                    print(f"\n🎉 CONVERGED after {iter_num} iterations!")
                break
        else:
            hold_count = 0
            
            # Adjust parameters
            new_params = adjust_parameters(params, multiplier, config)
            
            if verbose:
                print(f"\nParameter adjustment:")
                print(f"  price_noise: {params['price_noise']:.4f} → {new_params['price_noise']:.4f}")
                print(f"  time_noise: {params['time_noise_minutes']:.1f} → {new_params['time_noise_minutes']:.1f} min")
            
            params = new_params
            policy = UniformPolicy(params=params, seed=seed)
    
    # Final summary
    if verbose:
        logger.print_summary()
        
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Total iterations: {len(logger.iterations)}")
        print(f"Final AUC: {logger.iterations[-1]['auc_val']:.4f}")
        print(f"Final parameters:")
        print(f"  price_noise: {params['price_noise']:.4f}")
        print(f"  time_noise: {params['time_noise_minutes']:.1f} min")
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for Week 3 pilot.
    
    Runs adaptive adversary on Uniform policy and saves comprehensive results.
    """
    print("="*80)
    print("P7 WEEK 3 PILOT: ADAPTIVE ADVERSARY ON UNIFORM POLICY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = AdaptiveConfig()
    
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
    print("\n[Execution] Starting adaptive training loop...")
    try:
        results = adaptive_training_loop(
            prices,
            initial_params=DEFAULT_UNIFORM_PARAMS,
            config=config,
            seed=42,
            verbose=True
        )
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    print("\n[Output] Saving results...")
    try:
        results['logger'].save_results(config.OUTPUT_DIR)
    except Exception as e:
        print(f"  ✗ ERROR saving results: {e}")
    
    # Final message
    print("\n" + "="*80)
    if results['converged']:
        print("✅ WEEK 3 PILOT COMPLETE - CONVERGED")
    else:
        print("✅ WEEK 3 PILOT COMPLETE - MAX ITERATIONS REACHED")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick interpretation
    final_auc = results['logger'].iterations[-1]['auc_val']
    print(f"\nInterpretation:")
    if final_auc > 0.75:
        print(f"  ⚠️  High predictability (AUC={final_auc:.4f})")
        print(f"      → Adversary can predict trades with {final_auc*100:.1f}% accuracy")
        print(f"      → Need more aggressive randomization (OU/Pink policies)")
    elif final_auc < 0.55:
        print(f"  ⚠️  Too random (AUC={final_auc:.4f})")
        print(f"      → Strategy may be too noisy")
        print(f"      → Consider reducing randomization")
    else:
        print(f"  ✓ Good balance (AUC={final_auc:.4f})")
        print(f"      → Predictability in target range")
        print(f"      → Randomization is effective")


if __name__ == "__main__":
    main()
