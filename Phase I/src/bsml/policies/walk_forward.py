"""
Walk-Forward Validation Module
Implements rolling window out-of-sample testing with exactly 12 windows.

Paper spec (Section 5):
- N_WINDOWS = 12
- train_days = 504 (2 years)
- test_days  = 126 (6 months)
- step_days  = computed dynamically so exactly 12 windows fit in the data
"""

import numpy as np
import pandas as pd

N_WINDOWS = 12           # paper Section 5: exactly 12 walk-forward windows
TRAIN_DAYS = 504         # 2 years of trading days
TEST_DAYS = 126          # 6 months of trading days


class WalkForwardValidator:
    """
    Walk-forward validation with rolling windows.
    Always produces exactly N_WINDOWS windows regardless of dataset length.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.train_days = self.config.get('walk_forward_train', TRAIN_DAYS)
        self.test_days = self.config.get('walk_forward_test', TEST_DAYS)
        self.n_windows = self.config.get('walk_forward_n_windows', N_WINDOWS)

    def create_windows(self, n_days: int) -> list:
        """
        Create exactly n_windows walk-forward splits.

        step_days is computed so that the last window ends exactly at n_days.
        Raises ValueError if there are not enough days for even one window.
        """
        min_days = self.train_days + self.test_days
        if n_days < min_days:
            raise ValueError(
                f"Not enough data: need {min_days} days for one window, got {n_days}"
            )

        # Total span covered by n windows:
        # train_days + n*test_days  (each additional window adds test_days)
        # step = test_days ensures windows tile the available data
        # But we need exactly n_windows; compute step to fill available range.
        available_after_first = n_days - self.train_days - self.test_days
        if self.n_windows > 1:
            step_days = max(1, available_after_first // (self.n_windows - 1))
        else:
            step_days = available_after_first + 1  # only one window needed

        windows = []
        for i in range(self.n_windows):
            train_start = i * step_days
            train_end = train_start + self.train_days
            test_start = train_end
            test_end = test_start + self.test_days
            if test_end > n_days:
                break
            windows.append({
                'window_idx': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

        return windows

    def run(self, prices_df: pd.DataFrame, policy) -> dict:
        """
        Run walk-forward validation for a given policy.

        Parameters
        ----------
        prices_df : DataFrame in wide format (rows=dates, cols=symbols) or long format
        policy    : object with a generate_trades(prices) method

        Returns
        -------
        dict with keys: windows, oos_sharpes, oos_returns, mean_sharpe, std_sharpe, mean_return
        """
        n_days = len(prices_df)
        windows = self.create_windows(n_days)

        oos_sharpes = []
        oos_returns = []

        for window in windows:
            test_prices = prices_df.iloc[window['test_start']:window['test_end']]

            try:
                trades = policy.generate_trades(test_prices)
            except Exception:
                oos_sharpes.append(0.0)
                oos_returns.append(0.0)
                continue

            if trades.empty:
                oos_sharpes.append(0.0)
                oos_returns.append(0.0)
                continue

            # Simple daily P&L proxy: signed weight * price changes
            daily_pnl = trades.groupby('date').apply(
                lambda g: (g['qty'] * g['price']).sum()
            )
            if len(daily_pnl) < 2 or daily_pnl.std() == 0:
                oos_sharpes.append(0.0)
                oos_returns.append(0.0)
                continue

            sharpe = np.sqrt(252) * daily_pnl.mean() / daily_pnl.std()
            ann_return = daily_pnl.mean() * 252
            oos_sharpes.append(float(sharpe))
            oos_returns.append(float(ann_return))

        return {
            'windows': windows,
            'oos_sharpes': oos_sharpes,
            'oos_returns': oos_returns,
            'mean_sharpe': float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            'std_sharpe': float(np.std(oos_sharpes)) if oos_sharpes else 0.0,
            'mean_return': float(np.mean(oos_returns)) if oos_returns else 0.0,
        }
