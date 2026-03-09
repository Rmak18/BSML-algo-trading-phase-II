""" deterministic time-series momentum strategy with proper weight caps
"""

import numpy as np
import pandas as pd

class BaselineStrategy:
    """
    Baseline time-series momentum strategy with volatility targeting
    
    Implements:
    1. 12-month momentum signal calculation (skipping most recent month)
    2. Volatility-targeted position sizing
    3. Per-position weight caps (±25% max)
    4. One-day execution lag for realism
    5. Transaction cost modeling
    """
    
    def __init__(self, config):
        self.config = config
        self.universe = config['universe']
        self.lookback_momentum = config['lookback_momentum']  # 252 days = 12 months
        self.lookback_vol = config['lookback_vol']  # 60 days
        self.target_vol = config['target_vol']  # 40% per position
        self.max_position = config.get('max_position', 0.25)  # ±25% cap
        
    def calculate_signals(self, prices_df):
        
        signals_df = pd.DataFrame(index=prices_df.index)
        signals_df['date'] = prices_df['date']
        
        for etf in self.universe:
           
            returns_12m = prices_df[etf].pct_change(self.lookback_momentum)
            
           
            signal = np.where(returns_12m > 0, 1,
                             np.where(returns_12m < 0, -1, 0))
            
            
            signals_df[etf] = pd.Series(signal).shift(1)
        
        return signals_df
    
    def calculate_weights(self, prices_df, signals_df):
       
        weights_df = pd.DataFrame(index=prices_df.index)
        weights_df['date'] = prices_df['date']
        
       
        returns_df = prices_df[self.universe].pct_change()
        
        for etf in self.universe:
            
            vol_rolling = returns_df[etf].rolling(self.lookback_vol).std() * np.sqrt(252)
            
           
            vol_scaled = np.minimum(self.target_vol / (vol_rolling + 1e-6), 1.0)
            raw_weight = signals_df[etf] * vol_scaled
            
            
            capped_weight = np.clip(raw_weight, -self.max_position, self.max_position)

            # Signal already carries a 1-day lag (shifted in calculate_signals).
            # Do NOT shift again here — that would create a 2-day lag.
            weights_df[etf] = capped_weight
        
        return weights_df
    
    def calculate_returns(self, prices_df, weights_df):
       
        returns_df = prices_df[self.universe].pct_change()
        
       
        # weights_df already reflects the intended position for each day
        # (signal is lagged once in calculate_signals). No additional shift here.
        portfolio_returns = (weights_df[self.universe] * returns_df[self.universe]).sum(axis=1)
        
        
        turnover = weights_df[self.universe].diff().abs().sum(axis=1)
        
        
        transaction_costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        
       
        net_returns = portfolio_returns - transaction_costs
        
        return net_returns, transaction_costs
    
    def calculate_metrics(self, returns):
     
        returns_clean = returns.dropna()
        
      
        cum_returns = (1 + returns_clean).cumprod()
        
       
        total_return = cum_returns.iloc[-1] - 1
        
       
        years = len(returns_clean) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
       
        annual_vol = returns_clean.std() * np.sqrt(252)
        
       
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
    
    with open('config.json', 'r') as f:
        config = json.load(f)[0]
    
    
    import os
    if os.path.exists('prices.csv'):
        prices_df = pd.read_csv('prices.csv', parse_dates=['date'])
    else:
        prices_df = generate_etf_prices(config)
        prices_df.to_csv('prices.csv', index=False)
    
  
    baseline = BaselineStrategy(config)
    results = baseline.run(prices_df)
    
    print("Baseline Strategy Results:")
    print(f"Sharpe Ratio: {results['sharpe']:.3f}")
    print(f"Annual Return: {results['annual_return']*100:.2f}%")
    print(f"Annual Vol: {results['annual_vol']*100:.2f}%")
    print(f"Max Drawdown: {results['max_dd']*100:.2f}%")
    print(f"Total Return: {results['total_return']*100:.2f}%")
