import pandas as pd
from .pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Runner entrypoint for policy='pink'.

    Delegates to PinkNoisePolicy, which:
    - Starts from the baseline schedule
    - Applies pink-noise perturbations to ref_price.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty", "ref_price", "price"])

    params = DEFAULT_PINK_PARAMS

    policy = PinkNoisePolicy(
        alpha=params["alpha"],
        price_scale=params["price_scale"],
        seed=42,  # if you want, later you can wire this to the run.yaml seed
    )

    return policy.generate_trades(prices)
