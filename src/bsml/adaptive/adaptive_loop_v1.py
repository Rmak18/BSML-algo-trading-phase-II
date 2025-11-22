"""
P7 Adaptive Adversary Framework - STRENGTHENED VERSION

Prediction: "Will a trade occur tomorrow?"
Strengthened: 3 models, 5-fold CV, weighted ensemble

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import prepare_adversary_data
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_data


class AdaptiveConfig:
    """Configuration for adaptive training loop"""
    
    AUC_HIGH_THRESHOLD = 0.60
    AUC_LOW_THRESHOLD = 0.50
    AUC_TARGET_MIN = 0.50
    AUC_TARGET_MAX = 0.60
    AUC_TARGET_MID = 0.55
    
    FACTOR_INCREASE = 1.20
    FACTOR_DECREASE = 0.80
    FACTOR_NUDGE = 1.10
    
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    USE_SMOTE = True
    USE_CV = True
    N_CV_FOLDS = 5  # Increased from 3
    
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    PRICE_NOISE_MIN = 0.001
    PRICE_NOISE_MAX = 0.25
    TIME_NOISE_MIN = 1
    TIME_NOISE_MAX = 240
    
    OUTPUT_DIR = Path("outputs/adaptive_runs/uniform_pilot")


def decide_adjustment(auc: float, config: AdaptiveConfig = None):
    """Decide parameter adjustment based on AUC"""
    if config is None:
        config = AdaptiveConfig()
    
    if auc > config.AUC_HIGH_THRESHOLD:
        return 'INCREASE', config.FACTOR_INCREASE, f'Too predictable (AUC={auc:.3f})'
    elif auc < config.AUC_LOW_THRESHOLD:
        return 'DECREASE', config.FACTOR_DECREASE, f'Too random (AUC={auc:.3f})'
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


class IterationLogger:
    """Track metrics across iterations"""
    
    def __init__(self):
        self.iterations = []
        self.start_time = datetime.now()
    
    def log_iteration(self, iteration, params, auc, action, multiplier, reason, 
                     train_metrics, val_metrics, cv_scores=None):
        """Log iteration metrics"""
        
        cv_mean = None
        cv_std = None
        if cv_scores is not None:
            if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            'auc_val': float(auc),
            'auc_cv_mean': cv_mean,
            'auc_cv_std': cv_std,
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
        
        df = self.to_dataframe()
        df.to_csv(output_dir / "adaptive_results.csv", index=False)
        
        with open(output_dir / "adaptive_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        summary = {
            'metadata': {
                'task': 'predict_next_day_trade',
                'classifier': 'strengthened_3model_ensemble',
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


def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfig] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Adaptive training loop with STRENGTHENED classifier"""
    
    if config is None:
        config = AdaptiveConfig()
    
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    params = initial_params.copy()
    policy = UniformPolicy(params=params, seed=seed)
    logger = IterationLogger()
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print("P7 ADAPTIVE ADVERSARY - STRENGTHENED VERSION")
        print("="*80)
        print(f"Task: Predict 'Will a trade occur tomorrow?'")
        print(f"Classifier: 3-model ensemble (GB + RF + ExtraTrees)")
        print(f"Max iterations: {config.MAX_ITERATIONS}")
        print(f"Convergence patience: {config.CONVERGENCE_PATIENCE}")
        print(f"AUC target: [{config.AUC_TARGET_MIN}, {config.AUC_TARGET_MAX}]")
        print(f"Initial params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
    
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
        
        try:
            if verbose:
                print("[1/5] Generating trades...")
            trades = policy.generate_trades(prices_df)
            if len(trades) == 0:
                print("  ✗ No trades!")
                break
            if verbose:
                print(f"  → {len(trades)} trades")
            
            if verbose:
                print("[2/5] Preparing adversary data...")
            adversary_data = prepare_adversary_data(trades, prices_df)
            if verbose:
                print(f"  → {len(adversary_data)} observations")
                n_trade_days = adversary_data['signal'].sum()
                print(f"  → {n_trade_days} trade days ({n_trade_days/len(adversary_data)*100:.1f}%)")
            
            if verbose:
                print("[3/5] Splitting data...")
            train, val, test = time_split_data(adversary_data, config.TRAIN_RATIO, config.VAL_RATIO)
            if verbose:
                print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            
            if len(val) < config.MIN_VAL_SAMPLES:
                print(f"  ✗ Val set too small ({len(val)})")
                break
            
            if verbose:
                print("[4/5] Training adversary...")
            
            adversary = P7AdaptiveAdversary(
                use_smote=config.USE_SMOTE,
                use_cv=config.USE_CV,
                n_cv_folds=config.N_CV_FOLDS,
                random_state=seed
            )
            
            train_metrics = adversary.train(train, verbose=verbose)
            if not train_metrics.get('success', False):
                print(f"  ✗ Training failed: {train_metrics.get('reason')}")
                break
            
            if verbose:
                print("[5/5] Evaluating...")
            
            val_metrics = adversary.evaluate(val, verbose=verbose)
            if not val_metrics.get('success', False):
                print(f"  ✗ Evaluation failed")
                break
            
            auc = val_metrics['auc']
            
            action, multiplier, reason = decide_adjustment(auc, config)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESULT: AUC = {auc:.4f}")
                print(f"ACTION: {action}")
                print(f"REASON: {reason}")
            
            logger.log_iteration(
                iter_num, params, auc, action, multiplier, reason,
                train_metrics, val_metrics,
                adversary.cv_scores if hasattr(adversary, 'cv_scores') else None
            )
            
            if action == 'HOLD':
                hold_count += 1
                if verbose:
                    print(f"✓ In target ({hold_count}/{config.CONVERGENCE_PATIENCE})")
                
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
                    print(f"\nAdjustment:")
                    print(f"  price_noise: {params['price_noise']:.4f}")
                    print(f"  time_noise: {params['time_noise_minutes']:.1f} min")
        
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if verbose:
        logger.print_summary()
        
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Iterations: {len(logger.iterations)}")
        if logger.iterations:
            print(f"Final AUC: {logger.iterations[-1]['auc_val']:.4f}")
            print(f"Final params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }


def main():
    """Main entry point"""
    print("="*80)
    print("P7 WEEK 3 PILOT: STRENGTHENED ADAPTIVE ADVERSARY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nClassifier: 3-model ensemble (GB + RF + ExtraTrees)")
    print("Task: Predict next-day trade occurrence")
    
    config = AdaptiveConfig()
    
    print("\n[1/3] Loading data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return
    
    print("\n[2/3] Running adaptive loop...")
    try:
        results = adaptive_training_loop(prices, config=config, seed=42, verbose=True)
    except Exception as e:
        print(f"\n✗ FATAL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/3] Saving results...")
    try:
        results['logger'].save_results(config.OUTPUT_DIR)
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
    
    print("\n" + "="*80)
    if results['converged']:
        print("✅ CONVERGED")
    else:
        print("✅ MAX ITERATIONS")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['logger'].iterations:
        final_auc = results['logger'].iterations[-1]['auc_val']
        print(f"\nFinal AUC: {final_auc:.4f}")
        
        if final_auc > 0.75:
            print("⚠️  HIGH PREDICTABILITY - Need more randomization")
        elif final_auc < 0.55:
            print("⚠️  TOO RANDOM - Reducing signal quality")
        else:
            print("✓ GOOD BALANCE - Effective randomization")


if __name__ == "__main__":
    main()
