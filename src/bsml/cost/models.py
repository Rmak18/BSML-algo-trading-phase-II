from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable

import yaml
import pandas as pd


# ---------------------------
# Public API (P3 ownership)
# ---------------------------

def load_cost_config(path: str) -> Dict[str, Any]:
    """
    Load transaction-cost parameters from a YAML file.

    Parameters
    ----------
    path : str
        File-system path to the YAML file (e.g., 'configs/costs.yaml').

    Returns
    -------
    Dict[str, Any]
        A dictionary with the cost parameters that downstream code can read.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If required keys are missing or malformed.

    Notes
    -----
    - This function does *not* interpret finance formulas. It only loads and
      validates presence of a minimal set of keys so the pipeline is robust.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cost config not found: {p}")

    try:
        cfg = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in cost config: {p}") from e

    _validate_cost_keys(cfg, required_keys=(
        "commission_per_share",
        "spread_factor",
        "temp_impact_bps",
        "perm_impact_bps",
        "slippage_bps",
        "short_borrow_annual",
        "exchange_fee_bps",
    ))
    return cfg


def apply_costs(trades: pd.DataFrame, costs_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Attach execution-related columns to a trades table (placeholder wiring).

    This is an *interface* that standardizes outputs for downstream roles (P1/P2/P5).
    It does NOT implement finance formulas; values are placeholders so the schema is
    stable and the runner can proceed.

    Expected input schema (rows = trade slices)
    -------------------------------------------
    trades columns REQUIRED:
        - 'date'       : trading timestamp or day (any datetime-like)
        - 'symbol'     : instrument identifier (string)
        - 'side'       : literal 'BUY' or 'SELL' (string, upper case)
        - 'qty'        : strictly positive number of shares (float or int)
        - 'ref_price'  : reference/ideal price used for ΔIS (float)

    Output columns added (standardized contract)
    --------------------------------------------
        - 'exec_price' : realized execution price (float, placeholder = ref_price)
        - 'cost_bps'   : total costs in basis points (float, placeholder = 0.0)
        - 'cost_value' : monetary cost per row (float, placeholder = 0.0)

    Parameters
    ----------
    trades : pd.DataFrame
        Table of intended trades as produced by the policy (P2).
    costs_cfg : Dict[str, Any]
        Cost parameters as loaded by `load_cost_config`. They are *not* used here,
        but are accepted to lock the function signature for future updates.

    Returns
    -------
    pd.DataFrame
        A copy of `trades` with the three standardized columns appended.

    Raises
    ------
    ValueError
        If required input columns are missing.
    """
    _require_columns(trades, required=(
        "date", "symbol", "side", "qty", "ref_price"
    ))

    out = trades.copy()

    # Placeholder execution price = reference price.
    # This keeps downstream metrics derivations simple until formulas are added.
    out["exec_price"] = out["ref_price"].astype(float)

    # Placeholders for costs:
    # - cost_bps   : total cost expressed in basis points (1 bp = 0.01%)
    # - cost_value : absolute currency amount of the cost for that trade row
    out["cost_bps"] = 0.0
    out["cost_value"] = 0.0

    # Optional sanity: normalize 'side' to upper-case to avoid downstream surprises.
    if "side" in out.columns:
        out["side"] = out["side"].astype(str).str.upper()

    # Quantities should be positive by contract; enforce numeric dtype.
    out["qty"] = pd.to_numeric(out["qty"], errors="raise")

    return out


# ---------------------------
# Internal helpers (P3)
# ---------------------------

def _validate_cost_keys(cfg: Dict[str, Any], required_keys: Iterable[str]) -> None:
    """Ensure required YAML keys are present; raise clear errors if not."""
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            "Missing keys in cost config: "
            + ", ".join(missing)
            + ". Please update 'configs/costs.yaml'."
        )


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required DataFrame columns are present; raise clear errors if not."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Trades table is missing required columns: "
            + ", ".join(missing)
            + ". Expected at least: "
            + ", ".join(required)
        )

