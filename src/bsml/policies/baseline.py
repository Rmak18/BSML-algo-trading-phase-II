import numpy as np
import pandas as pd


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline trading policy.

    - Input: prices with columns ['date', 'symbol', 'price']
    - For each symbol & day:
        * compute daily return
        * if return >= 0 → BUY (qty = +1)
        * if return < 0  → SELL (qty = -1)
    - Output trades DataFrame with:
        ['date', 'symbol', 'side', 'qty', 'price', 'ref_price']
    """

  
    df = prices.copy()
    df = df[["date", "symbol", "price"]].copy()

 
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

   
    df = df.dropna(subset=["price"])

  
    df = df.sort_values(["symbol", "date"])

    df["ret"] = df.groupby("symbol")["price"].pct_change()

   
    ret_filled = df["ret"].fillna(0.0)
    qty = np.where(ret_filled >= 0.0, 1.0, -1.0)

    df["qty"] = qty
    df["side"] = np.where(df["qty"] > 0, "BUY", "SELL")

    trades = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "side": df["side"],
            "qty": df["qty"].astype(float),
            "price": df["price"].astype(float),       # used by cost model
            "ref_price": df["price"].astype(float),   # policies can override this
        }
    )

    return trades
