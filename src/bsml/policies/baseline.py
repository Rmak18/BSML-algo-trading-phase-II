import pandas as pd

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic baseline schedule:
    - Take the price history for one symbol.
    - Execute a fixed total quantity evenly over all timestamps.

    Input `prices` schema (from load_prices):
        - 'date'
        - 'symbol'
        - 'price'

    Output `trades` schema (for cost model):
        - 'date'
        - 'symbol'
        - 'side'
        - 'qty'
        - 'ref_price'
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty", "ref_price"])

    required = {"date", "symbol", "price"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"baseline.generate_trades: missing columns {missing}")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    # simple equal-slice execution over the whole horizon
    n = len(df)
    total_qty = 100_000.0   # arbitrary total quantity
    slice_qty = total_qty / n

    trades = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "side": "BUY",                          # baseline = buy schedule
            "qty": slice_qty,
            "ref_price": df["price"].astype(float), # used by cost model
        }
    )

    return trades
