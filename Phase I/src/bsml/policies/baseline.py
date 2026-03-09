"""
Baseline time-series momentum strategy (Section 5 of paper).

Signal  : sign(price_t / price_{t-252} - 1)  — 12-month TSMOM
Lag     : 1-day execution lag (signal on day t executes on day t+1)
Sizing  : signal * min(0.40 / vol_60d_annualised, 1.0)
Caps    : ±25% per position, net ±5%, gross < 1.5×
"""

import numpy as np
import pandas as pd

# Strategy constants matching Section 5 of the paper
LOOKBACK_MOM: int = 252    # 12 months of trading days
LOOKBACK_VOL: int = 60     # 3 months of trading days
TARGET_VOL: float = 0.40   # 40% annualised volatility target per position
MAX_POSITION: float = 0.25 # ±25% NAV cap per position
MAX_GROSS: float = 1.50    # < 1.5× NAV gross exposure
NET_TOL: float = 0.05      # ±5% net exposure tolerance


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate baseline TSMOM trades from long-format price data.

    Parameters
    ----------
    prices : DataFrame with columns ['date', 'symbol', 'price']

    Returns
    -------
    DataFrame with columns ['date', 'symbol', 'side', 'qty', 'price', 'ref_price']
        qty  : signed portfolio weight (fraction of NAV); positive = long
        price: closing price on execution date (cost model uses this)
        ref_price: intended execution price (policies can override this)
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values(["date", "symbol"])

    # ── Pivot to wide format: rows=dates, cols=symbols ──────────────────────
    wide = df.pivot(index="date", columns="symbol", values="price")
    symbols = wide.columns.tolist()
    daily_returns = wide.pct_change()

    # ── Signal: sign of 12-month momentum ───────────────────────────────────
    mom_12m = wide.pct_change(LOOKBACK_MOM)           # (p_t / p_{t-252}) - 1
    raw_signal = np.sign(mom_12m)                      # +1, 0, or -1

    # ── 1-day execution lag (signal known at close of t, executed on t+1) ───
    signal_lagged = raw_signal.shift(1)

    # ── Volatility-targeted sizing: signal * min(TARGET_VOL / vol_60d, 1.0) ─
    vol_60d = daily_returns.rolling(LOOKBACK_VOL).std() * np.sqrt(252)
    vol_scalar = (TARGET_VOL / vol_60d.clip(lower=1e-8)).clip(upper=1.0)
    raw_weights = signal_lagged * vol_scalar

    # ── Per-position cap: ±25% NAV ──────────────────────────────────────────
    weights = raw_weights.clip(lower=-MAX_POSITION, upper=MAX_POSITION)

    # ── Gross exposure cap: rescale portfolio if total |w| > 1.5× ───────────
    gross_exp = weights.abs().sum(axis=1)
    scale = (MAX_GROSS / gross_exp.clip(lower=1e-8)).clip(upper=1.0)
    weights = weights.multiply(scale, axis=0)

    # ── Net exposure check (log violations; clamp if > NET_TOL) ─────────────
    net_exp = weights.sum(axis=1)
    net_violation = (net_exp.abs() > NET_TOL)
    if net_violation.any():
        # Distribute net exposure equally across positions to neutralise
        n_pos = (weights != 0).sum(axis=1).replace(0, 1)
        adj = net_exp / n_pos
        for sym in symbols:
            weights.loc[net_violation, sym] -= adj.loc[net_violation]
        weights = weights.clip(lower=-MAX_POSITION, upper=MAX_POSITION)

    # ── Convert wide weights to long-format trades ───────────────────────────
    records = []
    for date, row in weights.iterrows():
        for sym in symbols:
            w = row[sym]
            if pd.isna(w) or w == 0.0:
                continue
            px = wide.loc[date, sym]
            if pd.isna(px):
                continue
            records.append(
                {
                    "date": date,
                    "symbol": sym,
                    "side": "BUY" if w > 0 else "SELL",
                    "qty": float(abs(w)),
                    "price": float(px),
                    "ref_price": float(px),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["date", "symbol", "side", "qty", "price", "ref_price"]
        )

    trades = (
        pd.DataFrame(records)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    return trades
