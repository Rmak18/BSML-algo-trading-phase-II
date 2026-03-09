import os
import pandas as pd

# Paths relative to repo root
INPUT_DIR = os.path.join("data", "etf_1y")
OUTPUT_DIR = os.path.join(INPUT_DIR, "backtest_ready")
COMBINED_PATH = os.path.join("data", "ALL_backtest.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean(filepath: str, symbol: str) -> pd.DataFrame:
    """
    Load a single ETF CSV and return a backtest-ready DataFrame
    with columns: date, symbol, price
    """
    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename price-like columns to "price"
    rename_map = {
        "adj close": "price",
        "adj_close": "price",
        "close": "price",
        "prices": "price",  # in case some file uses this
        "price": "price",
    }
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError(f"❌ Missing 'date' column in {filepath}. Columns: {df.columns.tolist()}")

    if "price" not in df.columns:
        raise ValueError(f"❌ Missing 'price' column in {filepath}. Columns: {df.columns.tolist()}")

    # Make price numeric, drop bad rows (e.g. strings like 'agg')
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # Build final DataFrame in EXACT order: date, symbol, price
    df_out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "symbol": symbol.upper(),
        "price": df["price"].astype(float),
    })

    # Enforce order again just in case
    df_out = df_out[["date", "symbol", "price"]]

    # Sort by date for that symbol
    df_out = df_out.sort_values("date").reset_index(drop=True)

    return df_out


def main():
    all_dfs = []

    print(f"Looking for ETF CSVs in: {INPUT_DIR}")
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".csv"):
            continue
        if filename.lower() == "all_backtest.csv":
            # skip previously combined files if present
            continue

        filepath = os.path.join(INPUT_DIR, filename)

        # Infer symbol from filename like "SPY_1y.csv" -> "SPY"
        base = os.path.splitext(filename)[0]
        symbol = base.split("_")[0].upper()

        print(f"Processing {filename} → symbol={symbol}")

        df_symbol = load_and_clean(filepath, symbol)

        # Save per-ETF backtest-ready file
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}_backtest.csv")
        df_symbol.to_csv(out_path, index=False)
        print(f"  [OK] Saved {out_path} ({len(df_symbol)} rows)")

        all_dfs.append(df_symbol)

    if not all_dfs:
        print("❌ No CSV files found in input dir. Nothing to do.")
        return

    # Combine all ETFs into one big DataFrame
    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort by symbol + date and enforce column order
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    combined = combined[["date", "symbol", "price"]]

    # Save combined file
    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"\n[OK] Wrote combined backtest CSV to: {COMBINED_PATH}")
    print(f"Rows: {len(combined)}, columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
