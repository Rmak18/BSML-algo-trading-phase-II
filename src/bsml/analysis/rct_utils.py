"""
P5 metrics and bootstrap utilities for the RCT pilot (Early vs Late).
"""
from typing import Tuple, Dict, Iterable, Optional
import numpy as np
import pandas as pd

def implementation_shortfall(ref_price: np.ndarray, exec_price: np.ndarray, side: Iterable[str]) -> np.ndarray:
    """
    Vectorized IS calculation.
    For buy: (exec - ref) / ref
    For sell: (ref - exec) / ref
    """
    ref = np.asarray(ref_price, dtype=float)
    exe = np.asarray(exec_price, dtype=float)
    side = np.asarray(list(side))
    buy_mask = np.char.lower(side) == "buy"
    sell_mask = np.char.lower(side) == "sell"
    is_vals = np.empty_like(ref, dtype=float)
    is_vals[buy_mask] = (exe[buy_mask] - ref[buy_mask]) / ref[buy_mask]
    is_vals[sell_mask] = (ref[sell_mask] - exe[sell_mask]) / ref[sell_mask]
    # If any other sides exist, default to buy interpretation
    other_mask = ~(buy_mask | sell_mask)
    is_vals[other_mask] = (exe[other_mask] - ref[other_mask]) / ref[other_mask]
    return is_vals

def delta_is_pairs(df_pairs: pd.DataFrame, early_label: str="early", late_label: str="late") -> pd.Series:
    """
    Given a DataFrame with paired rows per trade_id (early & late arms), compute ΔIS per pair:
        ΔIS = IS_early - IS_late
    Returns a Series indexed by trade_id.
    Expects columns: ['trade_id','arm','ref_price','exec_price','side']
    """
    # Compute IS per row
    df = df_pairs.copy()
    df["IS"] = implementation_shortfall(df["ref_price"].values, df["exec_price"].values, df["side"].values)
    # Pivot to align early/late for each trade_id
    piv = df.pivot_table(index="trade_id", columns="arm", values="IS", aggfunc="mean")
    # Drop incomplete pairs
    piv = piv.dropna(subset=[early_label, late_label], how="any")
    delta = piv[early_label] - piv[late_label]
    return delta

def bootstrap_mean_ci(x: np.ndarray, n_boot: int=2000, ci: float=0.95, seed: Optional[int]=17) -> Dict[str, float]:
    """
    Percentile bootstrap for the mean of x. Returns dict with mean, low, high, se.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "low": np.nan, "high": np.nan, "se": np.nan}
    boots = np.empty(n_boot, dtype=float)
    n = x.size
    for i in range(n_boot):
        bs = rng.choice(x, size=n, replace=True)
        boots[i] = bs.mean()
    alpha = (1 - ci) / 2.0
    low = np.quantile(boots, alpha)
    high = np.quantile(boots, 1 - alpha)
    return {"mean": float(x.mean()), "low": float(low), "high": float(high), "se": float(boots.std(ddof=1))}
