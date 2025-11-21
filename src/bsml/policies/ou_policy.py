import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# Default OU params (for bsml.policies.__init__ compatibility)
DEFAULT_OU_PARAMS = {
    "theta": 0.15,
    "sigma": 0.02,
    "price_scale": 1.0,
}

FAST_REVERSION_OU_PARAMS = {
    "theta": 0.5,
    "sigma": 0.02,
    "price_scale": 1.0,
}

SLOW_REVERSION_OU_PARAMS = {
    "theta": 0.05,
    "sigma": 0.02,
    "price_scale": 1.0,
}


class OUPolicy:
    """
    Ornstein–Uhlenbeck mean-reverting noise policy.
    Starts from the baseline schedule and adds correlated noise to ref_price.
    """

    def __init__(self, theta=0.15, sigma=0.02, price_scale=1.0, seed=None):
        self.theta = theta
        self.sigma = sigma
        self.price_scale = price_scale
        self.rng = np.random.default_rng(seed)

    def _ou_noise(self, n: int) -> np.ndarray:
        x = np.zeros(n)
        for t in range(1, n):
            dx = self.theta * (0.0 - x[t - 1]) + self.sigma * self.rng.normal()
            x[t] = x[t - 1] + dx
        # x is mean-reverting around 0
        return x

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        # 🔑 Ensure there is a 'price' column for the cost model
        if "price" not in trades.columns:
            if "ref_price" in trades.columns:
                trades["price"] = trades["ref_price"]
            elif "price" in prices.columns:
                trades["price"] = prices["price"].values
            else:
                raise ValueError(
                    "ou_policy.generate_trades: cannot find 'price' or 'ref_price' "
                    "to build a price column for costs."
                )

        # If ref_price missing for some reason, initialize from price
        if "ref_price" not in trades.columns:
            trades["ref_price"] = trades["price"]

        noise = self._ou_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1.0 + self.price_scale * noise)

        return trades


__all__ = [
    "OUPolicy",
    "DEFAULT_OU_PARAMS",
    "FAST_REVERSION_OU_PARAMS",
    "SLOW_REVERSION_OU_PARAMS",
]
