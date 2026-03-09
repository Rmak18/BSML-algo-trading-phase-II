import pandas as pd
from .pink_policy import generate_trades as _generate_trades  # noqa: F401

# Re-export so the runner can call bsml.policies.pink.generate_trades
generate_trades = _generate_trades
