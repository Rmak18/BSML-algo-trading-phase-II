"""
P7 Adaptive Adversary Framework - Week 3 Production Version

Features:
- Auto-detection of prediction window based on actual trade gaps
- SMOTE resampling for balanced training
- Cross-validation
- Comprehensive logging
- Convergence detection

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
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
    
    # AUC thresholds
    AUC_HIGH_THRESHOLD = 0.75
    AUC_LOW_THRESHOLD = 0.55
    AUC_TARGET_MIN = 0.60
    AUC_TARGET_MAX = 0.70
    AUC_TARGET_MID = 0.65
    
    # Adjustment factors
    FACTOR_INCREASE = 1.20
    FACTOR_DECREASE = 0.80
    FACTOR_NUDGE = 1.10
    
    # Loop parameters
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    # Adversary parameters (will be auto-detected)
    PREDICTION_WINDOW_MINUTES = None  # Auto-detect
    USE_SMOTE = True
    USE_CV = True
    N_CV_FOLDS = 3
    
    # Data split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Parameter bounds
    PRICE_NOISE_MIN = 0.001
    PRICE_NOISE_MAX = 0.25
    TIME_NOISE_MIN = 1
    TIME_NOISE_MAX = 240
    
    # Output
    OUTPUT_DIR = Path("outputs/adaptive_runs/uniform_pilot")
    SAVE_DETAILED_LOGS = True


# =============================================================================
# AUTO-DETECTION UTILITIES
# =============================================================================

def analyze_trade_gaps(trades_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Analyze inter-trade time gaps to understand trading frequency.
    
    Args:
        trades_df: Enriched trades DataFrame
        verbose: Print detailed statistics
    
    Returns:
        Dict with gap statistics
    """
    df = trades_df.sort_values(['symbol', 'timestamp']).copy()
    
    # Calculate gaps in minutes
    df['gap_minutes'] = (
        df.groupby('symbol')['timestamp']
        .diff().dt.total_seconds() / 60.0
    )
    
    gaps = df['gap_minutes'].dropna()
    
    if len(gaps) == 0:
        return {'error': 'No gaps found'}
    
    stats = {
        'count': len(gaps),
        'min': float(gaps.min()),
        'q25': float(gaps.quantile(0.25)),
        'median': float(gaps.median()),
        'q75': float(gaps.quantile(0.75)),
        'q90': float(gaps.quantile(0.90)),
        'max': float(gaps.max()),
        'mean': float(gaps.mean()),
        'std': float(gaps.std())
    }
    
    if verbose:
        print(f"\n[Gap Analysis] Trade timing statistics:")
        print(f"  Total gaps: {stats['count']}")
        print(f"  Min gap: {stats['min']:.1f} minutes ({stats['min']/60:.1f} hours)")
        print(f"  25th percentile: {stats['q25']:.1f} minutes ({stats['q25']/60:.1f} hours)")
        print(f"  Median: {stats['median']:.1f} minutes ({stats['median']/60:.1f} hours)")
        print(f"  75th percentile: {stats['q75']:.1f} minutes ({stats['q75']/60:.1f} hours)")
        print(f"  90th percentile: {stats['q90']:.1f} minutes ({stats['q90']/60:.1f} hours)")
        print(f"  Max gap: {stats['max']:.1f} minutes ({stats['max']/60:.1f} hours)")
    
    return stats


def auto_detect_prediction_window(
    trades_df: pd.DataFrame, 
    target_positive_rate: float = 0.30,
    verbose: bool = True
) -> float:
    """
    Auto-detect prediction window to achieve target positive label rate.
    
    Strategy: Find the gap percentile that gives ~30% positive labels.
    This ensures balanced training without being too trivial.
    
    Args:
        trades_df: Enriched trades
        target_positive_rate: Desired % of positive labels (0.3 = 30%)
        verbose: Print details
    
    Returns:
        Suggested window in minutes
    """
    df = trades_df.sort_values(['symbol', 'timestamp']).copy()
    
    # Calculate gaps
    df['gap_minutes'] = (
        df.groupby('symbol')['timestamp']
        .diff().dt.total_seconds() / 60.0
    )
    
    gaps = df['gap_minutes'].dropna().values
    
    if len(gaps) == 0:
        if verbose:
            print("[ERROR] No gaps found! Using default 60 minutes.")
        return 60.0
    
    # Find percentile that gives target positive rate
    # If we want 30% positive, use the 30th percentile as threshold
    suggested_window = np.percentile(gaps, target_positive_rate * 100)
    
    # Validate: compute actual positive rate with this window
    actual_positive_rate = (gaps <= suggested_window).mean()
    
    if verbose:
        print(f"\n[Auto-detect] Prediction window analysis:")
        print(f"  Target positive rate: {target_positive_rate*100:.1f}%")
        print(f"  Suggested window: {suggested_window:.1f} minutes ({suggested_window/60:.2f} hours)")
        print(f"  Actual positive rate: {actual_positive_rate*100:.1f}%")
        print(f"  This means: Predict if next trade within {suggested_window/60:.2f} hours")
    
    # Safety bounds: between 10 minutes and 7 days
    suggested_window = np.clip(suggested_window, 10, 10080)
    
    return float(suggested_window)


# =============================================================================
# DECISION LOGIC
# =============================================================================

def decide_adjustment(auc: float, config: AdaptiveConfig = None) -> Tuple[str, float, str]:
    """Decide parameter adjustment based on AUC"""
    if config is None:
        config = AdaptiveConfig()
    
    if auc > config.AUC_HIGH_THRESHOLD:
        return 'INCREASE', config.FACTOR_INCREASE, f'Too predictable (AUC={auc:.3f} > 0.75)'
    elif auc < config.AUC_LOW_THRESHOLD:
        return 'DECREASE', config.FACTOR_DECREASE, f'Too random (AUC={auc:.3f} < 0.55)'
    elif config.AUC_TARGET_MIN <= auc <= config.AUC_TARGET_MAX:
        return 'HOLD', 1.0, f'In target range (AUC={auc:.3f})'
    elif auc > config.AUC_TARGET_MAX:
        return 'NUDGE_UP', config.FACTOR_NUDGE, f'Slightly high (AUC={auc:.3f})'
    else:
        return 'NUDGE_DOWN', 1.0 / config.FACTOR_NUDGE, f'Slightly low (AUC={auc:.3f})'


def adjust_parameters(params: Dict, multiplier: float, config: AdaptiveConfig = None) -> Dict:
    """Adjust parameters with safety bounds"""
    if config is None:
        config = AdaptiveConfig()
    
    new_params = params.copy()
    new_params['price_noise'] = np.clip(
        new_params['price_noise'] * multiplier,
        config.PRICE_NOISE_MIN,
        config.PRICE_NOISE_MAX
    )
    new_params['time_noise_minutes'] = np.clip(
        new_params['time_noise_minutes'] * multiplier,
        config.TIME_NOISE_MIN,
        config.TIME_NOISE_MAX
    )
    
    return new_params


# =============================================================================
# LOGGING
# =============================================================================

class IterationLogger:
    """Track metrics across iterations"""
    
    def __init__(self):
        self.iterations = []
        self.start_time = datetime.now()
        self.prediction_window = None
        self.gap_stats = None
    
    def set_metadata(self, prediction_window: float, gap_stats: Dict):
        """Store metadata about the run"""
        self.prediction_window = prediction_window
        self.gap_stats = gap_stats
    
    def log_iteration(self, iteration, params, auc, action, multiplier, reason, 
                     train_metrics, val_metrics, cv_scores=None):
        """Log iteration metrics"""
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            'auc_val': float(auc),
            'auc_cv_mean': float(np.mean(cv_scores)) if cv_scores else None,
            'auc_cv_std': float(np.std(cv_scores)) if cv_scores else None,
            'action': action,
            'multiplier': float(multiplier),
            'reason': reason,
            'n_train': train_metrics.get('n_samples', 0),
            'n_val': val_metrics.get('n_samples', 0),
            'n_features': train_metrics.get('n_features', 0),
            'train_pos_rate': (
                train_metrics['label_distribution']['positive'] / train_metrics['n_samples'] * 100
                if train_metrics.get('n_samples') else 0
            ),
            'val_pos_rate': (
                val_metrics['label_distribution']['positive'] / val_metrics['n_samples'] * 100
                if val_metrics.get('n_samples') else 0
            ),
            'confusion_matrix': val_metrics.get('confusion_matrix'),
        }
        self.iterations.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame(self.iterations)
    
    def print_summary(self):
        """Print summary table"""
        if not self.iterations:
            print("\n[No iterations completed]")
            return
        
        df = self.to_dataframe()
        summary_cols = ['iteration', 'auc_val', 'action', 'price_noise', 'time_noise_minutes']
        
        print("\n" + "="*80)
        print("ITERATION SUMMARY")
        print("="*80)
        print(df[summary_cols].to_string(index=False, float_format='%.4f'))
    
    def save_results(self, output_dir: Path):
        """Save results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save iterations
        df = self.to_dataframe()
        df.to_csv(output_dir / "adaptive_results.csv", index=False)
        
        with open(output_dir / "adaptive_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        # Save summary with metadata
        summary = {
            'metadata': {
                'prediction_window_minutes': self.prediction_window,
                'prediction_window_hours': self.prediction_window / 60 if self.prediction_window else None,
                'gap_statistics': self.gap_stats,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            },
            'results': {
                'total_iterations': len(self.iterations),
                'final_auc': float(self.iterations[-1]['auc_val']) if self.iterations else None,
                'final_params': {
                    'price_noise': float(self.iterations[-1]['price_noise']),
                    'time_noise_minutes': float(self.iterations[-1]['time_noise_minutes'])
                } if self.iterations else None,
                'auc_trajectory': [float(x['auc_val']) for x in self.iterations],
                'actions': [x['action'] for x in self.iterations],
            }
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_dir}")


# =============================================================================
# ADAPTIVE TRAINING LOOP
# =============================================================================

def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfig] = None,
    prediction_window: Optional[float] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Production adaptive training loop with auto-detection.
    
    Args:
        prices_df: Price data
        initial_params: Starting policy parameters
        config: Configuration object
        prediction_window: Override auto-detection (minutes)
        seed: Random seed
        verbose: Print logs
    
    Returns:
        Dict with logger and results
    """
    if config is None:
        config = AdaptiveConfig()
    
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    # Use provided window or config's window
    if prediction_window is None:
        prediction_window = config.PREDICTION_WINDOW_MINUTES
    
    params = initial_params.copy()
    policy = UniformPolicy(params=params, seed=seed)
    logger = IterationLogger()
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print("P7 ADAPTIVE ADVERSARY TRAINING LOOP")
        print("="*80)
        print(f"Prediction window: {prediction_window:.1f} minutes ({prediction_window/60:.2f} hours)")
        print(f"Max iterations: {config.MAX_ITERATIONS}")
        print(f"Convergence patience: {config.CONVERGENCE_PATIENCE}")
        print(f"AUC target: [{config.AUC_TARGET_MIN}, {config.AUC_TARGET_MAX}]")
        print(f"Initial params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
    
    # Main loop
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
        
        try:
            # Generate trades
            if verbose:
                print("[1/5] Generating trades...")
            trades = policy.generate_trades(prices_df)
            if len(trades) == 0:
                print("  ✗ No trades generated!")
                break
            if verbose:
                print(f"  → {len(trades)} trades")
            
            # Enrich
            if verbose:
                print("[2/5] Enriching trades...")
            enriched = enrich_trades_for_adversary(trades, prices_df, f'uniform_iter{iter_num}')
            if verbose:
                print(f"  → {len(enriched)} rows")
            
            # Split
            if verbose:
                print("[3/5] Splitting data...")
            train, val, test = time_split_trades(enriched, config.TRAIN_RATIO, config.VAL_RATIO)
            if verbose:
                print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            
            if len(val) < config.MIN_VAL_SAMPLES:
                print(f"  ✗ Validation set too small ({len(val)} < {config.MIN_VAL_SAMPLES})")
                break
            
            # Train adversary
            if verbose:
                print("[4/5] Training adversary...")
            
            adversary = P7AdaptiveAdversary(
                window_threshold_minutes=prediction_window,
                use_smote=config.USE_SMOTE,
                use_cv=config.USE_CV,
                n_cv_folds=config.N_CV_FOLDS,
                random_state=seed
            )
            
            train_metrics = adversary.train(train, verbose=verbose)
            if not train_metrics.get('success', False):
                print(f"  ✗ Training failed: {train_metrics.get('reason', 'unknown')}")
                break
            
            # Evaluate
            if verbose:
                print("[5/5] Evaluating on validation...")
            
            val_metrics = adversary.evaluate(val, verbose=verbose)
            if not val_metrics.get('success', False):
                print(f"  ✗ Evaluation failed: {val_metrics.get('reason', 'unknown')}")
                break
            
            auc = val_metrics['auc']
            
            # Decision
            action, multiplier, reason = decide_adjustment(auc, config)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESULT: AUC = {auc:.4f}")
                print(f"ACTION: {action}")
                print(f"REASON: {reason}")
                print(f"MULTIPLIER: {multiplier:.2f}")
            
            # Log
            logger.log_iteration(
                iter_num, params, auc, action, multiplier, reason,
                train_metrics, val_metrics,
                adversary.cv_scores if hasattr(adversary, 'cv_scores') else None
            )
            
            # Convergence check
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
                params = adjust_parameters(params, multiplier, config)
                policy = UniformPolicy(params=params, seed=seed)
                
                if verbose:
                    print(f"\nParameter adjustment:")
                    print(f"  price_noise: {params['price_noise']:.4f}")
                    print(f"  time_noise: {params['time_noise_minutes']:.1f} min")
        
        except Exception as e:
            print(f"\n✗ ERROR in iteration {iter_num}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    if verbose:
        logger.print_summary()
        
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Total iterations: {len(logger.iterations)}")
        if logger.iterations:
            print(f"Final AUC: {logger.iterations[-1]['auc_val']:.4f}")
            print(f"Final params:")
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
    """Main entry point with auto-detection"""
    print("="*80)
    print("P7 WEEK 3 PILOT: ADAPTIVE ADVERSARY ON UNIFORM POLICY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = AdaptiveConfig()
    
    # Load data
    print("\n[1/4] Loading price data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"  ✓ Loaded {len(prices)} rows")
        print(f"  ✓ Symbols: {prices['symbol'].nunique()}")
        print(f"  ✓ Date range: {prices['date'].min()} to {prices['date'].max()}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return
    
    # Auto-detect prediction window
    print("\n[2/4] Auto-detecting prediction window...")
    try:
        # Generate sample trades
        sample_policy = UniformPolicy(params=DEFAULT_UNIFORM_PARAMS, seed=42)
        sample_trades = sample_policy.generate_trades(prices)
        sample_enriched = enrich_trades_for_adversary(sample_trades, prices, 'sample')
        
        # Analyze gaps
        gap_stats = analyze_trade_gaps(sample_enriched, verbose=True)
        
        # Auto-detect window (target 30% positive rate)
        prediction_window = auto_detect_prediction_window(
            sample_enriched, 
            target_positive_rate=0.30,
            verbose=True
        )
        
        print(f"  ✓ Using prediction window: {prediction_window:.1f} minutes ({prediction_window/60:.2f} hours)")
        
    except Exception as e:
        print(f"  ⚠️ Auto-detection failed: {e}")
        prediction_window = 720.0  # Default: 12 hours
        gap_stats = None
        print(f"  Using default: {prediction_window} minutes")
    
    # Run adaptive loop
    print("\n[3/4] Running adaptive training loop...")
    try:
        results = adaptive_training_loop(
            prices,
            initial_params=DEFAULT_UNIFORM_PARAMS,
            config=config,
            prediction_window=prediction_window,
            seed=42,
            verbose=True
        )
        
        # Store metadata in logger
        results['logger'].set_metadata(prediction_window, gap_stats)
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    print("\n[4/4] Saving results...")
    try:
        results['logger'].save_results(config.OUTPUT_DIR)
    except Exception as e:
        print(f"  ✗ ERROR saving: {e}")
    
    # Final summary
    print("\n" + "="*80)
    if results['converged']:
        print("✅ WEEK 3 PILOT COMPLETE - CONVERGED")
    else:
        print("✅ WEEK 3 PILOT COMPLETE - MAX ITERATIONS REACHED")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Interpretation
    if results['logger'].iterations:
        final_auc = results['logger'].iterations[-1]['auc_val']
        print(f"\nFinal AUC: {final_auc:.4f}")
        print(f"Prediction window: {prediction_window:.1f} minutes ({prediction_window/60:.2f} hours)")
        
        if final_auc > 0.75:
            print("\n⚠️  HIGH PREDICTABILITY")
            print(f"   → Adversary predicts trades with {final_auc*100:.1f}% accuracy")
            print(f"   → Need more aggressive randomization")
            print(f"   → Consider OU or Pink noise policies")
        elif final_auc < 0.55:
            print("\n⚠️  TOO RANDOM")
            print(f"   → Strategy may be too noisy")
            print(f"   → Consider reducing randomization")
        else:
            print("\n✓ GOOD BALANCE")
            print(f"   → Predictability in target range")
            print(f"   → Randomization is effective")


if __name__ == "__main__":
    main()
