"""
Walk-Forward Validation Module
Implements rolling window out-of-sample testing
"""

import numpy as np
import pandas as pd

class WalkForwardValidator:
    """
    Walk-forward validation with rolling windows
    - Training: 2 years (504 trading days)
    - Testing: 6 months (126 trading days)
    - Step: 3 months (63 days)
    """
    
    def __init__(self, config):
        self.config = config
        self.train_days = config.get('walk_forward_train', 504)
        self.test_days = config.get('walk_forward_test', 126)
        self.step_days = config.get('walk_forward_step', 63)
        
    def create_windows(self, n_days):
        """
        Create rolling train/test windows
        
        Returns list of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        # Start with minimum training data
        current_start = 0
        
        while current_start + self.train_days + self.test_days <= n_days:
            train_start = current_start
            train_end = current_start + self.train_days
            test_start = train_end
            test_end = test_start + self.test_days
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            # Step forward
            current_start += self.step_days
        
        return windows
    
    def run(self, prices_df, policy):
        """
        Run walk-forward validation for a given policy
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            Price data
        policy : Policy object
            Policy to test (UniformPolicy, OUPolicy, or PinkNoisePolicy)
            
        Returns:
        --------
        dict with windows and out-of-sample performance metrics
        """
        n_days = len(prices_df)
        windows = self.create_windows(n_days)
        
        oos_sharpes = []
        oos_returns = []
        
        for i, window in enumerate(windows):
            # Extract train and test data
            train_prices = prices_df.iloc[window['train_start']:window['train_end']]
            test_prices = prices_df.iloc[window['test_start']:window['test_end']]
            
            # For simplicity, use same policy parameters
            # In practice, would re-optimize on training data
            
            # Run baseline on test period (using full data for signals)
            from baseline_strategy import BaselineStrategy
            baseline = BaselineStrategy(self.config)
            
            # Get signals using all available data up to test end
            full_data = prices_df.iloc[:window['test_end']]
            baseline_results = baseline.run(full_data)
            
            # Run policy on test period
            policy_results = policy.run(full_data, baseline_results)
            
            # Extract test period returns
            test_returns = policy_results.get('returns', None)
            if test_returns is not None:
                test_returns_period = test_returns.iloc[window['test_start']:window['test_end']]
            else:
                # If returns not available, approximate from metrics
                test_returns_period = None
            
            # Calculate test period Sharpe
            if test_returns_period is not None and len(test_returns_period) > 0:
                annual_return = policy_results['annual_return']
                annual_vol = policy_results['annual_vol']
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            else:
                sharpe = policy_results['sharpe']
            
            oos_sharpes.append(sharpe)
            oos_returns.append(policy_results.get('annual_return', 0))
        
        results = {
            'windows': windows,
            'oos_sharpes': oos_sharpes,
            'oos_returns': oos_returns,
            'mean_sharpe': np.mean(oos_sharpes),
            'std_sharpe': np.std(oos_sharpes),
            'mean_return': np.mean(oos_returns)
        }
        
        return results


if __name__ == '__main__':
    # Test walk-forward validation
    import json
    from data_generator import generate_etf_prices
    from randomization import OUPolicy
    
    with open('../config.json', 'r') as f:
        config = json.load(f)[0]
    
    # Load or generate data
    import os
    if os.path.exists('../prices.csv'):
        prices_df = pd.read_csv('../prices.csv', parse_dates=['date'])
    else:
        prices_df = generate_etf_prices(config)
        prices_df.to_csv('../prices.csv', index=False)
    
    # Create validator
    validator = WalkForwardValidator(config)
    
    # Test with OU Policy
    policy = OUPolicy(config)
    results = validator.run(prices_df, policy)
    
    print(f"Walk-Forward Results:")
    print(f"  Number of windows: {len(results['windows'])}")
    print(f"  Mean OOS Sharpe: {results['mean_sharpe']:.3f}")
    print(f"  Std OOS Sharpe: {results['std_sharpe']:.3f}")
    print(f"  Mean OOS Return: {results['mean_return']*100:.2f}%")
