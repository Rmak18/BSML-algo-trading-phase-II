"""
P7 Adaptive Adversary Framework - POLICY-AGNOSTIC VERSION (V2) - FIXED

Supports multiple randomization policies with proper initialization

Owner: P7
Week: 4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

from bsml.policies.uniform_policy import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkPolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import prepare_adversary_data
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_data


class AdaptiveConfig:
    """Configuration for adaptive training loop"""
    
    # AUC thresholds
    AUC_HIGH_THRESHOLD = 0.60
    AUC_LOW_THRESHOLD = 0.50
    AUC_TARGET_MIN = 0.50
    AUC_TARGET_MAX = 0.60
    AUC_TARGET_MID = 0.55
    
    # Adjustment factors
    FACTOR_INCREASE = 1.20
    FACTOR_DECREASE = 0.80
    FACTOR_NUDGE = 1.10
    
    # Convergence settings
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    # Classifier settings
    USE_SMOTE = True
    USE_CV = True
    N_CV_FOLDS = 5
    
    # Data split
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Policy-specific parameter bounds
    # Uniform
    UNIFORM_PRICE_NOISE_MIN = 0.001
    UNIFORM_PRICE_NOISE_MAX = 0.25
    UNIFORM_TIME_NOISE_MIN = 1
    UNIFORM_TIME_NOISE_MAX = 240
    
    # Pink
    PINK_PRICE_SCALE_MIN = 0.001
    PINK_PRICE_SCALE_MAX = 0.25
    
    # OU
    OU_SIGMA_MIN = 0.001
    OU_SIGMA_MAX = 0.15
    OU_PRICE_SCALE_MIN = 0.001
    OU_PRICE_SCALE_MAX = 2.0
    
    OUTPUT_DIR = Path("outputs/adaptive_runs")


# ============================================================================
# POLICY-SPECIFIC ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_uniform_params(params: Dict, multiplier: float, config: AdaptiveConfig) -> Dict:
    """Adjust Uniform policy parameters"""
    new_params = params.copy()
    new_params['price_noise'] = np.clip(
        new_params['price_noise'] * multiplier,
        config.UNIFORM_PRICE_NOISE_MIN,
        config.UNIFORM_PRICE_NOISE_MAX
    )
    new_params['time_noise_minutes'] = np.clip(
        new_params['time_noise_minutes'] * multiplier,
        config.UNIFORM_TIME_NOISE_MIN,
        config.UNIFORM_TIME_NOISE_MAX
    )
    return new_params


def adjust_pink_params(params: Dict, multiplier: float, config: AdaptiveConfig) -> Dict:
    """Adjust Pink noise policy parameters"""
    new_params = params.copy()
    new_params['price_scale'] = np.clip(
        new_params['price_scale'] * multiplier,
        config.PINK_PRICE_SCALE_MIN,
        config.PINK_PRICE_SCALE_MAX
    )
    return new_params


def adjust_ou_params(params: Dict, multiplier: float, config: AdaptiveConfig) -> Dict:
    """Adjust OU policy parameters"""
    new_params = params.copy()
    new_params['sigma'] = np.clip(
        new_params['sigma'] * multiplier,
        config.OU_SIGMA_MIN,
        config.OU_SIGMA_MAX
    )
    new_params['price_scale'] = np.clip(
        new_params['price_scale'] * multiplier,
        config.OU_PRICE_SCALE_MIN,
        config.OU_PRICE_SCALE_MAX
    )
    return new_params


# ============================================================================
# POLICY INITIALIZATION FUNCTIONS
# ============================================================================

def init_uniform_policy(params: Dict, seed: int):
    """Initialize UniformPolicy (expects params dict + seed)"""
    return UniformPolicy(params=params, seed=seed)


def init_pink_policy(params: Dict, seed: int):
    """Initialize PinkPolicy (expects individual kwargs)"""
    return PinkPolicy(**params, seed=seed)


def init_ou_policy(params: Dict, seed: int):
    """Initialize OUPolicy (expects individual kwargs)"""
    return OUPolicy(**params, seed=seed)


# ============================================================================
# POLICY REGISTRY
# ============================================================================

POLICY_REGISTRY = {
    'uniform': {
        'class': UniformPolicy,
        'init_func': init_uniform_policy,
        'default_params': DEFAULT_UNIFORM_PARAMS,
        'adjust_func': adjust_uniform_params,
        'display_name': 'Uniform Noise',
        'description': 'Independent random noise per trade'
    },
    'pink': {
        'class': PinkPolicy,
        'init_func': init_pink_policy,
        'default_params': DEFAULT_PINK_PARAMS,
        'adjust_func': adjust_pink_params,
        'display_name': 'Pink Noise (1/f)',
        'description': 'Correlated low-frequency noise'
    },
    'ou': {
        'class': OUPolicy,
        'init_func': init_ou_policy,
        'default_params': DEFAULT_OU_PARAMS,
        'adjust_func': adjust_ou_params,
        'display_name': 'Ornstein-Uhlenbeck',
        'description': 'Mean-reverting stochastic process'
    }
}


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


class IterationLogger:
    """Track metrics across iterations"""
    
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
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
        
        # Convert params to loggable format
        params_log = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in params.items()}
        
        entry = {
            'policy': self.policy_name,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'params': params_log,
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
        if not self.iterations:
            return pd.DataFrame()
        
        # Flatten params dict for CSV
        rows = []
        for entry in self.iterations:
            row = {k: v for k, v in entry.items() if k != 'params'}
            row.update(entry['params'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print summary table"""
        if not self.iterations:
            print("\n[No iterations completed]")
            return
        
        df = self.to_dataframe()
        summary_cols = ['iteration', 'auc_val', 'action'] + list(self.iterations[0]['params'].keys())
        available_cols = [c for c in summary_cols if c in df.columns]
        
        print("\n" + "="*80)
        print(f"ITERATION SUMMARY - {self.policy_name.upper()}")
        print("="*80)
        print(df[available_cols].to_string(index=False, float_format='%.4f'))
    
    def save_results(self, output_dir: Path):
        """Save results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_dir / f"{self.policy_name}_results.csv", index=False)
        
        with open(output_dir / f"{self.policy_name}_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        summary = {
            'metadata': {
                'policy': self.policy_name,
                'task': 'predict_next_day_trade',
                'classifier': 'strengthened_3model_ensemble',
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            },
            'results': {
                'total_iterations': len(self.iterations),
                'final_auc': float(self.iterations[-1]['auc_val']) if self.iterations else None,
                'final_params': self.iterations[-1]['params'] if self.iterations else None,
                'auc_trajectory': [float(x['auc_val']) for x in self.iterations],
                'actions': [x['action'] for x in self.iterations],
            }
        }
        
        with open(output_dir / f"{self.policy_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_dir}")


def adaptive_training_loop(
    prices_df: pd.DataFrame,
    policy_name: str,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfig] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Policy-agnostic adaptive training loop
    
    Args:
        prices_df: Price data
        policy_name: 'uniform', 'pink', or 'ou'
        initial_params: Starting parameters (uses defaults if None)
        config: AdaptiveConfig instance
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with logger, final_params, converged, n_iterations
    """
    
    if config is None:
        config = AdaptiveConfig()
    
    # Get policy info from registry
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {policy_name}. Choose from {list(POLICY_REGISTRY.keys())}")
    
    policy_info = POLICY_REGISTRY[policy_name]
    init_func = policy_info['init_func']
    adjust_func = policy_info['adjust_func']
    
    if initial_params is None:
        initial_params = policy_info['default_params'].copy()
    
    params = initial_params.copy()
    policy = init_func(params, seed)
    logger = IterationLogger(policy_name)
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print(f"P7 ADAPTIVE ADVERSARY - {policy_info['display_name'].upper()}")
        print("="*80)
        print(f"Description: {policy_info['description']}")
        print(f"Task: Predict 'Will a trade occur tomorrow?'")
        print(f"Classifier: 3-model ensemble (GB + RF + ExtraTrees)")
        print(f"Max iterations: {config.MAX_ITERATIONS}")
        print(f"AUC target: [{config.AUC_TARGET_MIN}, {config.AUC_TARGET_MAX}]")
        print(f"Initial params: {params}")
    
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
            
            # CRITICAL FIX: Check label distribution
            n_trade_days = adversary_data['signal'].sum()
            n_no_trade_days = len(adversary_data) - n_trade_days
            
            if verbose:
                print(f"  → {len(adversary_data)} observations")
                print(f"  → Trade days: {n_trade_days} ({n_trade_days/len(adversary_data)*100:.1f}%)")
                print(f"  → No-trade days: {n_no_trade_days} ({n_no_trade_days/len(adversary_data)*100:.1f}%)")
            
            # CRITICAL: If all days have trades, the adversary task is trivial
            if n_no_trade_days == 0:
                print(f"\n⚠️  WARNING: Policy generates trades EVERY day!")
                print(f"  → Adversary prediction task is trivial (no negative class)")
                print(f"  → This policy provides NO unpredictability")
                print(f"  → Skipping adversary training")
                
                # Log the issue
                logger.log_iteration(
                    iter_num, params, 1.0, 'SKIP', 1.0, 
                    'All days have trades - trivial prediction',
                    {'success': False, 'n_samples': 0, 'n_features': 0, 
                     'label_distribution': {'positive': n_trade_days, 'negative': 0}},
                    {'success': False, 'n_samples': 0, 
                     'label_distribution': {'positive': 0, 'negative': 0}},
                    None
                )
                break
            
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
                params = adjust_func(params, multiplier, config)
                policy = init_func(params, seed)
                
                if verbose:
                    print(f"\nAdjustment: {params}")
        
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
            print(f"Final params: {params}")
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }
