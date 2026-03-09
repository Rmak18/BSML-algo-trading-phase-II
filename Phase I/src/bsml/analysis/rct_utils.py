"""
P5 metrics and bootstrap utilities for the RCT pilot (Early vs Late).
"""
from typing import Tuple, Dict, Iterable, Optional
import numpy as np
import pandas as pd


# ============================================================
# 1) CORE METRIC UTILITIES (your original functions)
# ============================================================

def implementation_shortfall(ref_price: np.ndarray,
                             exec_price: np.ndarray,
                             side: Iterable[str]) -> np.ndarray:
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


def delta_is_pairs(df_pairs: pd.DataFrame,
                   early_label: str = "early",
                   late_label: str = "late") -> pd.Series:
    """
    Given a DataFrame with paired rows per trade_id (early & late arms),
    compute ΔIS per pair:

        ΔIS = IS_early - IS_late

    Returns a Series indexed by trade_id.
    Expects columns: ['trade_id','arm','ref_price','exec_price','side']
    """
    df = df_pairs.copy()
    df["IS"] = implementation_shortfall(
        df["ref_price"].values,
        df["exec_price"].values,
        df["side"].values,
    )

    # Pivot to align early/late for each trade_id
    piv = df.pivot_table(
        index="trade_id",
        columns="arm",
        values="IS",
        aggfunc="mean",
    )

    # Drop incomplete pairs
    piv = piv.dropna(subset=[early_label, late_label], how="any")

    delta = piv[early_label] - piv[late_label]
    return delta


def bootstrap_mean_ci(x: np.ndarray,
                      n_boot: int = 2000,
                      ci: float = 0.95,
                      seed: Optional[int] = 17) -> Dict[str, float]:
    """
    Percentile bootstrap for the mean of x.
    Returns dict with mean, low, high, se.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return {"mean": np.nan, "low": np.nan, "high": np.nan, "se": np.nan}

    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        bs = rng.choice(x, size=n, replace=True)
        boots[i] = bs.mean()

    alpha = (1.0 - ci) / 2.0
    low = np.quantile(boots, alpha)
    high = np.quantile(boots, 1.0 - alpha)

    return {
        "mean": float(x.mean()),
        "low": float(low),
        "high": float(high),
        "se": float(boots.std(ddof=1)),
    }


# ============================================================
# 2) HIGH-LEVEL HELPERS USED BY p5_rct_runner
# ============================================================

def load_trades_for_adversary(path: str) -> pd.DataFrame:
    """
    Load trades_for_adversary.csv and do light cleaning.

    Expected columns (case-insensitive):
        date, symbol, side, ref_price, exec_price, ...
    """
    df = pd.read_csv(path)

    # Normalize column names to lower case
    df.columns = [c.lower() for c in df.columns]

    # Parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Coerce numeric price columns if present
    for col in ["ref_price", "exec_price", "qty", "cost_bps", "cost_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key stuff
    required = [c for c in ["symbol", "side", "ref_price", "exec_price"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    return df


def make_pilot_rct_dataset(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build a *paired* early vs late RCT-style dataset from trades.

    Strategy:
      - sort trades by (symbol, policy, date)
      - within each (symbol, policy) group:
          - take rows in pairs: first = early, second = late
          - assign a shared trade_id for each pair

    This gives us:
        columns: [... original ..., 'arm', 'trade_id', 'IS']
    which can be fed into delta_is_pairs + bootstrap_mean_ci.
    """
    df = trades.copy()

    # Make sure we have consistent column names
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    sort_cols = []
    if "symbol" in df.columns:
        sort_cols.append("symbol")
    if "policy" in df.columns:
        sort_cols.append("policy")
    if "date" in df.columns:
        sort_cols.append("date")

    if not sort_cols:
        # fallback: just use index order
        df = df.copy()
        df["__order"] = np.arange(len(df))
        sort_cols = ["__order"]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Grouping keys for pairing
    group_cols = []
    if "symbol" in df.columns:
        group_cols.append("symbol")
    if "policy" in df.columns:
        group_cols.append("policy")

    if not group_cols:
        # single global group
        df["__group"] = "GLOBAL"
        group_cols = ["__group"]

    pairs_list = []

    for group_key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values(sort_cols).reset_index(drop=False)  # keep original index in "index"
        n = len(g)
        n_pairs = n // 2
        if n_pairs == 0:
            continue

        # Use only first 2 * n_pairs rows to make full pairs
        g_keep = g.iloc[: 2 * n_pairs].copy()
        # pair_idx: 0,0,1,1,2,2,...
        g_keep["pair_idx"] = np.repeat(np.arange(n_pairs), 2)
        # arm: early, late, early, late, ...
        g_keep["arm"] = np.tile(["early", "late"], n_pairs)

        # Build a trade_id string
        if isinstance(group_key, tuple):
            group_id = "_".join(str(x) for x in group_key)
        else:
            group_id = str(group_key)

        g_keep["trade_id"] = g_keep["pair_idx"].apply(
            lambda p: f"{group_id}_pair{int(p)}"
        )

        pairs_list.append(g_keep)

    if not pairs_list:
        # no pairs possible
        return pd.DataFrame(columns=list(df.columns) + ["arm", "trade_id", "IS"])

    rct_df = pd.concat(pairs_list, ignore_index=True)

    # Compute IS per row for convenience
    if all(col in rct_df.columns for col in ["ref_price", "exec_price", "side"]):
        rct_df["IS"] = implementation_shortfall(
            rct_df["ref_price"].values,
            rct_df["exec_price"].values,
            rct_df["side"].values,
        )
    else:
        rct_df["IS"] = np.nan

    return rct_df


def analyze_rct_results(rct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take an RCT-style DataFrame with columns including:
        ['trade_id','arm','ref_price','exec_price','side']
    and compute:
        ΔIS per pair, then bootstrap CI for mean ΔIS.

    Returns a 1-row DataFrame with:
        metric, n_pairs, mean_delta, ci_low, ci_high, se
    """
    if rct_df.empty:
        return pd.DataFrame(
            [{
                "metric": "delta_IS_early_minus_late",
                "n_pairs": 0,
                "mean_delta": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "se": np.nan,
            }]
        )

    delta = delta_is_pairs(rct_df, early_label="early", late_label="late")
    stats = bootstrap_mean_ci(delta.values)

    out = pd.DataFrame(
        [{
            "metric": "delta_IS_early_minus_late",
            "n_pairs": int(delta.notna().sum()),
            "mean_delta": stats["mean"],
            "ci_low": stats["low"],
            "ci_high": stats["high"],
            "se": stats["se"],
        }]
    )
    return out
