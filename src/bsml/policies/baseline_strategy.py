"""
Baseline Strategy Module
Implements deterministic time-series momentum strategy
"""

import numpy as np
import pandas as pd

class BaselineStrategy:
    """
    Baseline time-series momentum strategy with volatility targeting
    """
    
    def __init__(self, config):
        self.config = config
        self.universe = config['universe']
        self.lookback_momentum = config['lookback_momentum']
        self.lookback_vol = config['lookback_vol']
        self.target_vol = config['target_vol']
        
    def calculate_signals(self, prices_df):
        """
        Signal = sign(12-month return), lagged by 1 day
        """
        signals_df = pd.DataFrame(index=prices_df.index)
        signals_df['date'] = prices_df['date']
        
        for etf in self.universe:
            # 12-month return
            returns_12m = prices_df[etf].pct_change(self.lookback_momentum)
            
            # Directional signal: +1 (long), -1 (short), 0 (neutral)
            signal = np.where(returns_12m > 0, 1,
                             np.where(returns_12m < 0, -1, 0))
            
            # Lag by 1 day
            signals_df[etf] = pd.Series(signal).shift(1)
        
        return signals_df
    
    def calculate_weights(self, prices_df, signals_df):

        weights_df = pd.DataFrame(index=prices_df.index)
        weights_df['date'] = prices_df['date']
        
        # Calculate returns
        returns_df = prices_df[self.universe].pct_change()
        
        for etf in self.universe:
          
            vol_rolling = returns_df[etf].rolling(self.lookback_vol).std() * np.sqrt(252)
            
          
            raw_weight = signals_df[etf] * (self.target_vol / (vol_rolling + 1e-6))
            
   
            weights_df[etf] = raw_weight.shift(1)
        
        return weights_df
    
    def calculate_returns(self, prices_df, weights_df):
  
        returns_df = prices_df[self.universe].pct_change()
        
       
        portfolio_returns = (weights_df[self.universe].shift(1) * returns_df[self.universe]).sum(axis=1)
        
       
        turnover = weights_df[self.universe].diff().abs().sum(axis=1)
        transaction_costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        
      
        net_returns = portfolio_returns - transaction_costs
        
        return net_returns, transaction_costs
    
    def calculate_metrics(self, returns):
      
       
        returns = returns.dropna()
        
      
        cum_returns = (1 + returns).cumprod()
        
       
        total_return = cum_returns.iloc[-1] - 1
        
      
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
       
        annual_vol = returns.std() * np.sqrt(252)
        
        
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
       
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
    
    def run(self, prices_df):
      
       
        signals_df = self.calculate_signals(prices_df)
        
       
        weights_df = self.calculate_weights(prices_df, signals_df)
        
       
        net_returns, costs = self.calculate_returns(prices_df, weights_df)
        
      
        metrics = self.calculate_metrics(net_returns)
        
      
        results = {
            'signals': signals_df,
            'weights': weights_df,
            'returns': net_returns,
            'costs': costs,
            **metrics
        }
        
        return results


if __name__ == '__main__':
   
    import json
    from data_generator import generate_etf_prices
    
    with open('../config.json', 'r') as f:
        config = json.load(f)[0]
    
   
    import os
    if os.path.exists('../prices.csv'):
        prices_df = pd.read_csv('../prices.csv', parse_dates=['date'])
    else:
        prices_df = generate_etf_prices(config)
    
   
    baseline = BaselineStrategy(config)
    results = baseline.run(prices_df)
    
    print("Baseline Strategy Results:")
    print(f"Sharpe Ratio: {results['sharpe']:.3f}")
    print(f"Annual Return: {results['annual_return']*100:.2f}%")
    print(f"Annual Vol: {results['annual_vol']*100:.2f}%")
    print(f"Max Drawdown: {results['max_dd']*100:.2f}%")
