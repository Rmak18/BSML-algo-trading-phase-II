import pandas as pd
from .ou_policy import OUPolicy


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Runner entrypoint for policy='ou'.

    Delegates to OUPolicy, which:
    - Starts from the baseline schedule
    - Adds mean-reverting OU noise to ref_price
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty", "ref_price"])

    # You can tweak these hyperparameters if you want stronger/weaker noise
    # Use paper-default parameters (Section 7): θ=0.5, σ=0.5
    policy = OUPolicy(theta=0.5, sigma=0.5, price_scale=0.04, seed=42)
    return policy.generate_trades(prices)
