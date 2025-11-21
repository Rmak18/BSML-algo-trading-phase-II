import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# Parameter presets expected by bsml.policies.__init__
DEFAULT_UNIFORM_PARAMS = {
    "price_noise": 0.03,          # ±3% price noise
    "time_noise_minutes": 30,     # ±30 minutes
}

CONSERVATIVE_UNIFORM_PARAMS = {
    "price_noise": 0.01,
    "time_noise_minutes": 10,
}

AGGRESSIVE_UNIFORM_PARAMS = {
    "price_noise": 0.05,
    "time_noise_minutes": 60,
}

NOCLAMPING_UNIFORM_PARAMS = {
    "price_noise": 0.03,
    "time_noise_minutes": 30,
}


class UniformPolicy:
    """
    Wraps the baseline schedule but applies uniform random perturbations
    to ref_price (and optionally timing).
    """

    def __init__(self, params=None, seed=None):
        if params is None:
            params = DEFAULT_UNIFORM_PARAMS
        self.price_noise = params["price_noise"]
        self.time_noise_minutes = params["time_noise_minutes"]
        self.rng = np.random.default_rng(seed)

    def perturb_price(self, price: float) -> float:
        # price * (1 + U(-α, α))
        eps = self.rng.uniform(-self.price_noise, self.price_noise)
        return price * (1.0 + eps)

    def perturb_time(self, timestamp) -> pd.Timestamp:
        # Shift timestamp by uniform minutes
        delta = self.rng.uniform(-self.time_noise_minutes, self.time_noise_minutes)
        return timestamp + pd.Timedelta(minutes=float(delta))

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        # 🔑 baseline already provides 'price' and 'ref_price'
        # Keep 'price' for cost model, perturb only 'ref_price'
        noise = self.rng.uniform(
            1.0 - self.price_noise, 1.0 + self.price_noise, size=n
        )
        trades["ref_price"] = trades["ref_price"] * noise

        # Optional: also jitter timestamps a bit
        dates = pd.to_datetime(trades["date"])
        trades["date"] = [
            self.perturb_time(ts.to_pydatetime()) for ts in dates
        ]

        return trades


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Module-level entrypoint expected by the runner for policy='uniform_policy'.
    """
    policy = UniformPolicy(seed=42)
    return policy.generate_trades(prices)


__all__ = [
    "UniformPolicy",
    "DEFAULT_UNIFORM_PARAMS",
    "CONSERVATIVE_UNIFORM_PARAMS",
    "AGGRESSIVE_UNIFORM_PARAMS",
    "NOCLAMPING_UNIFORM_PARAMS",
    "generate_trades",
]
