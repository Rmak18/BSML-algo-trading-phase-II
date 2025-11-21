import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# Default pink-noise params (for bsml.policies.__init__ compatibility)
DEFAULT_PINK_PARAMS = {
    "alpha": 1.0,
    "price_scale": 0.04,
}
# Default pink-noise params (for bsml.policies.__init__ compatibility)
DEFAULT_PINK_PARAMS = {
    "alpha": 1.0,
    "price_scale": 0.04,
}

WHITE_NOISE_PARAMS = {
    "alpha": 0.0,      # white noise (flat spectrum)
    "price_scale": 0.04,
}

BROWN_NOISE_PARAMS = {
    "alpha": 2.0,      # brown noise / very low-freq dominated
    "price_scale": 0.04,
}



class PinkPolicy:
    """
    Pink-noise policy: low-frequency noise → persistent drifts.
    """

    def __init__(self, alpha=1.0, price_scale=0.04, seed=None):
        self.alpha = alpha
        self.price_scale = price_scale
        self.rng = np.random.default_rng(seed)

    def generate_pink_noise(self, n: int) -> np.ndarray:
        """1/f noise via FFT method, normalized to mean 0, std 1."""
        freqs = np.fft.rfftfreq(n)
        phases = (self.rng.normal(size=freqs.shape)
                  + 1j * self.rng.normal(size=freqs.shape))

        # Avoid division by zero at freq 0
        amp = np.where(freqs == 0, 0.0, 1.0 / (freqs ** self.alpha))
        spectrum = phases * amp

        noise = np.fft.irfft(spectrum, n)
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        return noise

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Start from baseline schedule
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        # 🔑 Ensure there is a 'price' column for the cost model
        if "price" not in trades.columns:
            if "ref_price" in trades.columns:
                trades["price"] = trades["ref_price"]
            else:
                # Fall back to original prices if present
                if "price" in prices.columns:
                    trades["price"] = prices["price"].values
                else:
                    raise ValueError(
                        "pink_policy.generate_trades: cannot find 'price' or 'ref_price' "
                        "to build a price column for costs."
                    )

        # Apply pink noise to ref_price, keep 'price' for cost calculation
        if "ref_price" not in trades.columns:
            trades["ref_price"] = trades["price"]

        noise = self.generate_pink_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1 + self.price_scale * noise)

        return trades


# Expose the name bsml.policies.__init__ expects
PinkNoisePolicy = PinkPolicy

__all__ = [
    "PinkPolicy",
    "PinkNoisePolicy",
    "DEFAULT_PINK_PARAMS",
    "WHITE_NOISE_PARAMS",
    "BROWN_NOISE_PARAMS",
]
