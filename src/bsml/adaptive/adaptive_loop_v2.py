"""
P7 Adaptive Adversary Framework - CORRECT VERSION

LOGIC:
1. Train adversary on BASELINE trades (deterministic pattern)
2. Test adversary on RANDOMIZED trades (Uniform/Pink/OU)
3. If test AUC high → increase randomization
4. If test AUC ~0.5-0.6 → optimal unpredictability

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
from bsml.policies.baseline import generate_trades as generate_baseline_trades
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
    
    # Prediction task
    PREDICTION_WINDOW_HOURS = 4.0
    
    # Policy bounds
    UNIFORM_PRICE_NOISE_MIN = 0.001
    UNIFORM_PRICE_NOISE_MAX = 0.25
    UNIFORM_TIME_NOISE_MIN = 1
    UNIFORM_TIME_NOISE_MAX = 240
    
    PINK_PRICE_SCALE_MIN = 0.001
    PINK_PRICE_SCALE_MAX = 0.25
    
    OU_SIGMA_MIN = 0.001
    OU_SIGMA_MAX = 0.15
    OU_PRICE_SCALE_MIN = 0.001
    OU_PRICE_SCALE_MAX = 2.0
    
    OUTPUT_DIR = Path("outputs/adaptive_runs")


def adjust_uniform_params(params: Dict, multiplier: float, config: AdaptiveConfig) -> Dict:
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
    new_params = params.copy()
    new_params['price_scale'] = np.clip(
        new_params['price_scale'] * multiplier,
        config.PINK_PRICE_SCALE_MIN,
        config.PINK_PRICE_SCALE_MAX
    )
    return new_params


def adjust_ou_params(params: Dict, multiplier: float, config: AdaptiveConfig) -> Dict:
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


def init_uniform_policy(params: Dict, seed: int):
    return UniformPolicy(params=params, seed=seed)


def init_pink_policy(params: Dict, seed: int):
    return PinkPolicy(**params, seed=seed)


def init_ou_policy(params: Dict, seed: int):
    return OUPolicy(**params, seed=seed)


POLICY_REGISTRY = {
    'uniform': {
        'init_func': init_uniform_policy,
        'default_params': DEFAULT_UNIFORM_PARAMS,
        'adjust_func': adjust_uniform_params,
        'display_name': 'Uniform Noise',
        'description': 'Independent random noise per trade'
    },
    'pink': {
        'init_func': init_pink_policy,
        'default_params': DEFAULT_PINK_PARAMS,
        'adjust_func': adjust_pink_params,
        'display_name': 'Pink Noise (1/f)',
        'description': 'Correlated low-frequency noise'
    },
    'ou': {
        'init_func': init_ou_policy,
        'default_params': DEFAULT_OU_PARAMS,
        'adjust_func': adjust_ou_params,
        'display_name': 'Ornstein-Uhlenbeck',
        'description': 'Mean-reverting stochastic process'
    }
}


def decide_adjustment(auc: float, config: AdaptiveConfig = None):
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
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.iterations = []
        self.start_time = datetime.now()
    
    def log_iteration(self, iteration, params, auc, action, multiplier, reason, 
                     train_metrics, test_metrics, cv_scores=None):
        cv_mean = None
        cv_std = None
        if cv_scores is not None:
            if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
        
        params_log = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in params.items()}
        
        entry = {
            'policy': self.policy_name,
            'iteration': iteration,
            'params': params_log,
            'auc_test': float(auc),
            'auc_cv_mean': cv_mean,
            'auc_cv_std': cv_std,
            'action': action,
            'multiplier': float(multiplier),
            'reason': reason,
            'n_train': train_metrics.get('n_samples', 0),
            'n_test': test_metrics.get('n_samples', 0),
        }
        self.iterations.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.iterations:
            return pd.DataFrame()
        
        rows = []
        for entry in self.iterations:
            row = {k: v for k, v in entry.items() if k != 'params'}
            row.update(entry['params'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        if not self.iterations:
            print("\n[No iterations completed]")
            return
        
        df = self.to_dataframe()
        summary_cols = ['iteration', 'auc_test', 'action'] + list(self.iterations[0]['params'].keys())
        available_cols = [c for c in summary_cols if c in df.columns]
        
        print("\n" + "="*80)
        print(f"ITERATION SUMMARY - {self.policy_name.upper()}")
        print("="*80)
        print(df[available_cols].to_string(index=False, float_format='%.4f'))
    
    def save_results(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_dir / f"{self.policy_name}_results.csv", index=False)
        
        with open(output_dir / f"{self.policy_name}_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
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
    CORRECT LOGIC:
    1. Generate BASELINE trades (deterministic)
    2. Train adversary on baseline
    3. Generate RANDOMIZED trades with policy
    4. Test adversary on randomized trades
    5. Adjust parameters based on test AUC
    """
    
    if config is None:
        config = AdaptiveConfig()
    
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    policy_info = POLICY_REGISTRY[policy_name]
    init_func = policy_info['init_func']
    adjust_func = policy_info['adjust_func']
    
    if initial_params is None:
        initial_params = policy_info['default_params'].copy()
    
    params = initial_params.copy()
    logger = IterationLogger(policy_name)
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print(f"P7 ADAPTIVE ADVERSARY - {policy_info['display_name'].upper()}")
        print("="*80)
        print(f"CORRECT LOGIC: Train on BASELINE, Test on RANDOMIZED")
        print(f"Initial params: {params}")
    
    # STEP 1: Generate BASELINE trades and prepare training data
    if verbose:
        print("\n[SETUP] Generating BASELINE trades for training...")
    
    baseline_trades = generate_baseline_trades(prices_df)
    baseline_data = prepare_adversary_data(
        baseline_trades, 
        prices_df,
        prediction_window_hours=config.PREDICTION_WINDOW_HOURS,
        verbose=verbose
    )
    
    if len(baseline_data) == 0:
        print("  ✗ No baseline data!")
        return {'logger': logger, 'final_params': params, 'converged': False, 'n_iterations': 0}
    
    # Split baseline data for training
    train, val, test = time_split_data(baseline_data, config.TRAIN_RATIO, config.VAL_RATIO)
    
    if verbose:
        print(f"\n[SETUP] Training adversary on BASELINE...")
        print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Train adversary ONCE on baseline
    adversary = P7AdaptiveAdversary(
        use_smote=config.USE_SMOTE,
        use_cv=config.USE_CV,
        n_cv_folds=config.N_CV_FOLDS,
        random_state=seed
    )
    
    train_metrics = adversary.train(train, verbose=verbose)
    if not train_metrics.get('success', False):
        print(f"  ✗ Training failed!")
        return {'logger': logger, 'final_params': params, 'converged': False, 'n_iterations': 0}
    
    if verbose:
        print(f"\n✓ Adversary trained on BASELINE")
        print(f"  Now testing on RANDOMIZED trades...\n")
    
    # STEP 2: Adaptive loop - test on randomized trades
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
        
        try:
            # Generate RANDOMIZED trades
            if verbose:
                print(f"[1/3] Generating RANDOMIZED trades (params={params})...")
            
            policy = init_func(params, seed)
            randomized_trades = policy.generate_trades(prices_df)
            
            if len(randomized_trades) == 0:
                print("  ✗ No trades!")
                break
            
            # Prepare test data from randomized trades
            if verbose:
                print(f"[2/3] Preparing test data from randomized trades...")
            
            test_data = prepare_adversary_data(
                randomized_trades,
                prices_df,
                prediction_window_hours=config.PREDICTION_WINDOW_HOURS,
                verbose=verbose
            )
            
            if len(test_data) < config.MIN_VAL_SAMPLES:
                print(f"  ✗ Test set too small ({len(test_data)})")
                break
            
            # Test adversary on randomized trades
            if verbose:
                print(f"[3/3] Testing adversary on randomized trades...")
            
            test_metrics = adversary.evaluate(test_data, verbose=verbose)
            if not test_metrics.get('success', False):
                print(f"  ✗ Evaluation failed")
                break
            
            auc = test_metrics['auc']
            
            action, multiplier, reason = decide_adjustment(auc, config)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESULT: Test AUC = {auc:.4f} (on randomized)")
                print(f"ACTION: {action}")
                print(f"REASON: {reason}")
            
            logger.log_iteration(
                iter_num, params, auc, action, multiplier, reason,
                train_metrics, test_metrics,
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
            print(f"Final Test AUC: {logger.iterations[-1]['auc_test']:.4f}")
            print(f"Final params: {params}")
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }
