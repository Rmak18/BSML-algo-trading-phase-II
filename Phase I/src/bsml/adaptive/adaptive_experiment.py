"""
Adaptive Experiment - Regression-Based Price Prediction

Instead of binary classification (can we detect?), this uses REGRESSION:
"Can we PREDICT the exact price and PROFIT from it?"

Key Metric: MAE% (Mean Absolute Error as % of price)
- MAE% < 0.5% → Adversary can exploit (adapt policy)
- MAE% > 1.0% → Safe from exploitation

This is more realistic: real adversaries want to PROFIT, not just detect.

Owner: P7
Week: 3
"""

import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from bsml.data.loader import load_prices
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS

# Import our price prediction adversaries
from price_prediction_adversary import (
    BaselineToPinkPredictor,
    BaselineToOUPredictor,
    BaselineToUniformPredictor,
    train_and_evaluate_price_predictor
)


class AdaptiveExperimentRegression:
    """
    Adaptive experiment using regression-based price prediction.
    
    Adversaries try to predict the randomized price from the baseline price.
    If prediction error (MAE%) is low, the policy needs stronger randomization.
    """
    
    def __init__(self, prices: pd.DataFrame, mae_threshold: float = 10.0, random_state: int = 42):
        """
        Initialize experiment.

        Args:
            prices: Market price data
            mae_threshold: MAE% threshold for adaptation (10.0% per paper Section 7)
            random_state: Random seed
        """
        self.prices = prices
        self.mae_threshold = mae_threshold
        self.random_state = random_state

        # Initialize policies with paper-correct parameters (Section 6)
        self.pink_params = {'alpha': 1.0, 'price_scale': 0.04}
        self.ou_params = {'theta': 0.5, 'sigma': 0.5, 'price_scale': 0.04}
        self.uniform_params = {'price_noise': 0.0005, 'time_noise_minutes': 120.0}
        
        # Track MAE% history
        self.mae_history = {
            'pink': [],
            'ou': [],
            'uniform': []
        }
        
        # Track parameter history
        self.param_history = {
            'pink': [],
            'ou': [],
            'uniform': []
        }
        
        # Track exploitability
        self.exploit_history = {
            'pink': [],
            'ou': [],
            'uniform': []
        }
    
    def generate_all_trades(self, iteration: int = 0) -> Dict[str, pd.DataFrame]:
        """
        Generate trades for all 4 policies.
        
        Args:
            iteration: Current iteration number (for varying random seed)
        
        Returns:
            Dictionary mapping policy name to trades DataFrame
        """
        
        print("\n" + "="*80)
        print("GENERATING TRADES FOR ALL POLICIES")
        print("="*80)
        
        # Baseline
        print("\n[1/4] Baseline...")
        baseline_trades = baseline_generate(self.prices)
        print(f"  ✓ Generated {len(baseline_trades)} trades")
        
        # Pink Noise - use iteration to vary seed
        print("\n[2/4] Pink Noise...")
        print(f"  Parameters: alpha={self.pink_params['alpha']}, price_scale={self.pink_params['price_scale']:.4f}")
        pink_policy = PinkNoisePolicy(
            alpha=self.pink_params['alpha'],
            price_scale=self.pink_params['price_scale'],
            seed=self.random_state + iteration * 1000  # Different seed each iteration
        )
        pink_trades = pink_policy.generate_trades(self.prices)
        
        # Calculate actual price deviation
        pink_dev = (pink_trades['ref_price'] - baseline_trades['price']).abs().mean()
        pink_dev_pct = (pink_dev / baseline_trades['price'].mean()) * 100
        print(f"  ✓ Generated {len(pink_trades)} trades")
        print(f"  Avg price deviation: ${pink_dev:.4f} ({pink_dev_pct:.2f}%)")
        
        # OU - use iteration to vary seed
        print("\n[3/4] Ornstein-Uhlenbeck...")
        print(f"  Parameters: theta={self.ou_params['theta']:.4f}, sigma={self.ou_params['sigma']:.4f}, price_scale={self.ou_params['price_scale']:.4f}")
        ou_policy = OUPolicy(
            theta=self.ou_params['theta'],
            sigma=self.ou_params['sigma'],
            price_scale=self.ou_params['price_scale'],
            seed=self.random_state + iteration * 1000  # Different seed each iteration
        )
        ou_trades = ou_policy.generate_trades(self.prices)
        
        ou_dev = (ou_trades['ref_price'] - baseline_trades['price']).abs().mean()
        ou_dev_pct = (ou_dev / baseline_trades['price'].mean()) * 100
        print(f"  ✓ Generated {len(ou_trades)} trades")
        print(f"  Avg price deviation: ${ou_dev:.4f} ({ou_dev_pct:.2f}%)")
        
        # Uniform - uses internal randomization, but update seed via params
        print("\n[4/4] Uniform...")
        print(f"  Parameters: price_noise={self.uniform_params['price_noise']:.4f}, time_noise={self.uniform_params['time_noise_minutes']:.1f}min")
        uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
        uniform_params.update(self.uniform_params)
        uniform_params['seed'] = self.random_state + iteration * 1000  # Different seed each iteration
        uniform_policy = UniformPolicy(params=uniform_params)
        uniform_trades = uniform_policy.generate_trades(self.prices)
        
        uniform_dev = (uniform_trades['ref_price'] - baseline_trades['price']).abs().mean()
        uniform_dev_pct = (uniform_dev / baseline_trades['price'].mean()) * 100
        print(f"  ✓ Generated {len(uniform_trades)} trades")
        print(f"  Avg price deviation: ${uniform_dev:.4f} ({uniform_dev_pct:.2f}%)")
        
        return {
            'baseline': baseline_trades,
            'pink': pink_trades,
            'ou': ou_trades,
            'uniform': uniform_trades
        }
    
    def train_price_predictors(self, trades: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict[str, Dict]:
        """
        Train all three price prediction adversaries.
        
        Args:
            trades: Dictionary of trades for each policy
            verbose: Print detailed results
        
        Returns:
            Dictionary of metrics for each adversary
        """
        
        baseline_trades = trades['baseline']
        pink_trades = trades['pink']
        ou_trades = trades['ou']
        uniform_trades = trades['uniform']
        
        results = {}
        
        # ========== ADVERSARY 1: Pink Price Predictor ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 1: PREDICTING PINK NOISE PRICES")
            print("="*80)
        
        pink_adversary = BaselineToPinkPredictor(random_state=self.random_state)
        train_metrics_pink, test_metrics_pink = train_and_evaluate_price_predictor(
            pink_adversary,
            baseline_trades,
            pink_trades,
            test_size=0.3,
            random_state=self.random_state,
            verbose=verbose
        )
        
        results['pink'] = {
            'train': train_metrics_pink,
            'test': test_metrics_pink,
            'adversary': pink_adversary
        }
        
        # ========== ADVERSARY 2: OU Price Predictor ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 2: PREDICTING OU PRICES")
            print("="*80)
        
        ou_adversary = BaselineToOUPredictor(random_state=self.random_state)
        train_metrics_ou, test_metrics_ou = train_and_evaluate_price_predictor(
            ou_adversary,
            baseline_trades,
            ou_trades,
            test_size=0.3,
            random_state=self.random_state,
            verbose=verbose
        )
        
        results['ou'] = {
            'train': train_metrics_ou,
            'test': test_metrics_ou,
            'adversary': ou_adversary
        }
        
        # ========== ADVERSARY 3: Uniform Price Predictor ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 3: PREDICTING UNIFORM PRICES")
            print("="*80)
        
        uniform_adversary = BaselineToUniformPredictor(random_state=self.random_state)
        train_metrics_uniform, test_metrics_uniform = train_and_evaluate_price_predictor(
            uniform_adversary,
            baseline_trades,
            uniform_trades,
            test_size=0.3,
            random_state=self.random_state,
            verbose=verbose
        )
        
        results['uniform'] = {
            'train': train_metrics_uniform,
            'test': test_metrics_uniform,
            'adversary': uniform_adversary
        }
        
        return results
    
    def adapt_policies(self, results: Dict[str, Dict]) -> Dict[str, bool]:
        """
        Adapt policies if MAE% < threshold (too exploitable).
        
        Args:
            results: Adversary results with test metrics
        
        Returns:
            Dictionary indicating which policies were adapted
        """
        
        adaptations = {}
        
        print("\n" + "="*80)
        print("ADAPTATION DECISIONS (Based on Exploitability)")
        print("="*80)
        print(f"Threshold: {self.mae_threshold}% MAE (typical transaction cost)")
        
        # Pink Noise adaptation
        pink_mae_pct = results['pink']['test']['mae_pct']
        pink_exploit = results['pink']['test']['exploitable_fraction']
        
        print(f"\nPink Noise:")
        print(f"  MAE%: {pink_mae_pct:.4f}%")
        print(f"  Exploitable trades: {pink_exploit*100:.1f}%")
        
        if pink_mae_pct < self.mae_threshold:
            print(f"  ⚠️  MAE% < {self.mae_threshold}% → TOO EXPLOITABLE → ADAPTING")
            self.pink_params['price_scale'] *= 1.2
            print(f"  New price_scale: {self.pink_params['price_scale']:.4f}")
            adaptations['pink'] = True
        else:
            print(f"  ✓ MAE% ≥ {self.mae_threshold}% → SAFE FROM EXPLOITATION")
            adaptations['pink'] = False
        
        # OU adaptation
        ou_mae_pct = results['ou']['test']['mae_pct']
        ou_exploit = results['ou']['test']['exploitable_fraction']
        
        print(f"\nOU Process:")
        print(f"  MAE%: {ou_mae_pct:.4f}%")
        print(f"  Exploitable trades: {ou_exploit*100:.1f}%")
        
        if ou_mae_pct < self.mae_threshold:
            print(f"  ⚠️  MAE% < {self.mae_threshold}% → TOO EXPLOITABLE → ADAPTING")
            self.ou_params['sigma'] *= 1.2
            self.ou_params['theta'] *= 1.2
            self.ou_params['price_scale'] *= 1.2
            print(f"  New sigma: {self.ou_params['sigma']:.4f}")
            print(f"  New theta: {self.ou_params['theta']:.4f}")
            print(f"  New price_scale: {self.ou_params['price_scale']:.4f}")
            adaptations['ou'] = True
        else:
            print(f"  ✓ MAE% ≥ {self.mae_threshold}% → SAFE FROM EXPLOITATION")
            adaptations['ou'] = False
        
        # Uniform adaptation
        uniform_mae_pct = results['uniform']['test']['mae_pct']
        uniform_exploit = results['uniform']['test']['exploitable_fraction']
        
        print(f"\nUniform:")
        print(f"  MAE%: {uniform_mae_pct:.4f}%")
        print(f"  Exploitable trades: {uniform_exploit*100:.1f}%")
        
        if uniform_mae_pct < self.mae_threshold:
            print(f"  ⚠️  MAE% < {self.mae_threshold}% → TOO EXPLOITABLE → ADAPTING")
            self.uniform_params['price_noise'] *= 1.2
            self.uniform_params['time_noise_minutes'] *= 1.2
            print(f"  New price_noise: {self.uniform_params['price_noise']:.4f}")
            print(f"  New time_noise: {self.uniform_params['time_noise_minutes']:.1f}min")
            adaptations['uniform'] = True
        else:
            print(f"  ✓ MAE% ≥ {self.mae_threshold}% → SAFE FROM EXPLOITATION")
            adaptations['uniform'] = False
        
        return adaptations
    
    def run_single_iteration(self, iteration: int = 0) -> Dict[str, Dict]:
        """
        Run a single iteration of the experiment.
        
        Args:
            iteration: Iteration number
        
        Returns:
            Results dictionary
        """
        
        print("\n" + "="*80)
        print(f"ITERATION {iteration}")
        print("="*80)
        
        # Generate trades with current iteration number
        trades = self.generate_all_trades(iteration=iteration)
        
        # Train price predictors
        results = self.train_price_predictors(trades, verbose=True)
        
        # Record history
        self.mae_history['pink'].append(results['pink']['test']['mae_pct'])
        self.mae_history['ou'].append(results['ou']['test']['mae_pct'])
        self.mae_history['uniform'].append(results['uniform']['test']['mae_pct'])
        
        self.exploit_history['pink'].append(results['pink']['test']['exploitable_fraction'])
        self.exploit_history['ou'].append(results['ou']['test']['exploitable_fraction'])
        self.exploit_history['uniform'].append(results['uniform']['test']['exploitable_fraction'])
        
        self.param_history['pink'].append(self.pink_params.copy())
        self.param_history['ou'].append(self.ou_params.copy())
        self.param_history['uniform'].append(self.uniform_params.copy())
        
        # Adapt policies
        adaptations = self.adapt_policies(results)
        
        return results
    
    def run_adaptive_loop(self, n_iterations: int = 10) -> pd.DataFrame:
        """
        Run multiple iterations with adaptation.
        
        Args:
            n_iterations: Number of iterations
        
        Returns:
            DataFrame with MAE% history
        """
        
        print("\n" + "="*80)
        print(f"ADAPTIVE EXPERIMENT (REGRESSION): {n_iterations} ITERATIONS")
        print("="*80)
        print(f"MAE% Threshold: {self.mae_threshold}%")
        print(f"Random State: {self.random_state}")
        
        for i in range(n_iterations):
            self.run_single_iteration(iteration=i)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'iteration': range(n_iterations),
            'pink_mae_pct': self.mae_history['pink'],
            'ou_mae_pct': self.mae_history['ou'],
            'uniform_mae_pct': self.mae_history['uniform'],
            'pink_exploitable': self.exploit_history['pink'],
            'ou_exploitable': self.exploit_history['ou'],
            'uniform_exploitable': self.exploit_history['uniform']
        })
        
        return results_df


def main():
    """Main execution"""
    
    print("="*80)
    print("ADAPTIVE ADVERSARY EXPERIMENT - PRICE PREDICTION (REGRESSION)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load price data
    print("\n[Step 1] Loading price data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ Loaded {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    
    # Initialize experiment with 3% threshold
    print("\n[Step 2] Initializing experiment...")
    experiment = AdaptiveExperimentRegression(
        prices=prices,
        mae_threshold=10.0,  # 3.0% = STRICT threshold for testing adaptation
        random_state=42
    )
    print("  ✓ Experiment initialized")
    print(f"  MAE% threshold: {experiment.mae_threshold}%")
    
    # Run adaptive loop
    print("\n[Step 3] Running adaptive loop...")
    results_df = experiment.run_adaptive_loop(n_iterations=10)
    
    # Print evolution table
    print("\n" + "="*80)
    print("MAE% EVOLUTION OVER ITERATIONS")
    print("="*80)
    print("\n" + results_df[['iteration', 'pink_mae_pct', 'ou_mae_pct', 'uniform_mae_pct']].to_string(index=False))
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nPink Noise:")
    print(f"  Initial MAE%: {results_df['pink_mae_pct'].iloc[0]:.4f}%")
    print(f"  Final MAE%:   {results_df['pink_mae_pct'].iloc[-1]:.4f}%")
    print(f"  Status: {'✅ SAFE' if results_df['pink_mae_pct'].iloc[-1] >= 10.0 else '⚠️ NEEDS MORE ADAPTATION'}")
    
    print(f"\nOU Process:")
    print(f"  Initial MAE%: {results_df['ou_mae_pct'].iloc[0]:.4f}%")
    print(f"  Final MAE%:   {results_df['ou_mae_pct'].iloc[-1]:.4f}%")
    print(f"  Status: {'✅ SAFE' if results_df['ou_mae_pct'].iloc[-1] >= 10.0 else '⚠️ NEEDS MORE ADAPTATION'}")
    
    print(f"\nUniform:")
    print(f"  Initial MAE%: {results_df['uniform_mae_pct'].iloc[0]:.4f}%")
    print(f"  Final MAE%:   {results_df['uniform_mae_pct'].iloc[-1]:.4f}%")
    print(f"  Status: {'✅ SAFE' if results_df['uniform_mae_pct'].iloc[-1] >= 10.0 else '⚠️ NEEDS MORE ADAPTATION'}")
    
    # Save results
    results_df.to_csv('adaptive_regression_results.csv', index=False)
    print(f"\n✓ Results saved to: adaptive_regression_results.csv")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
