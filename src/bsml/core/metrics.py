import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, eps: float = 1e-12) -> float:
    """
    Compute the annualized Sharpe ratio.

    Formula:
        Sharpe = mean(returns) / std(returns) * sqrt(252)

    Parameters
    ----------
    returns : pd.Series
        Daily percentage returns (as decimals, e.g. 0.01 = 1%).
    eps : float
        Tiny number added to avoid division by zero.

    Returns
    -------
    float
        The annualized Sharpe ratio.
    """
    mu = returns.mean()
    sd = returns.std(ddof=1)
    return float(mu / (sd + eps) * np.sqrt(252))

def max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown (worst peak-to-trough decline).

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity curve over time.

    Returns
    -------
    float
        The minimum percentage drawdown (negative value).
    """
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

def delta_is(px_ref: pd.Series, px_exec: pd.Series) -> float:
    """
    Compute implementation shortfall ΔIS.

    Formula:
        ΔIS = mean((px_exec - px_ref) / px_ref)

    Parameters
    ----------
    px_ref : pd.Series
        Reference or 'ideal' prices.
    px_exec : pd.Series
        Actual execution prices.

    Returns
    -------
    float
        Average relative shortfall (negative means worse execution).
    """
    rel = (px_exec - px_ref) / px_ref
    return float(rel.mean())

