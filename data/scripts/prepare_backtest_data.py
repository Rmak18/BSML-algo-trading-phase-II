import os
import pandas as pd

INPUT_DIR = os.path.join("data", "etf_1y")
OUTPUT_DIR = os.path.join(INPUT_DIR, "backtest_ready")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean(filepath, symbol):
    df = pd.read_csv(filepath)

    # normalize columns
    df.columns = [c.lower().strip() for c in df.columns]

    # map typical Yahoo columns to "price"
    rename_map = {
        "adj close": "price",
        "adj_close": "price",
        "close": "price",
    }
    df = df.rename(columns=rename_map)

    if "price" not in df.columns:
        raise ValueError(f"❌ Missing price column in {filepath}")

    # 🔴 IMPORTANT: force numeric and drop junk rows (like 'agg')
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    df_out = df[["date", "price"]].copy()
    df_out["symbol"] = symbol.upper()
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out = df_out.sort_values("date")

    return df_out
