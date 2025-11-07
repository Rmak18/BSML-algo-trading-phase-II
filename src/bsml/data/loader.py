from pathlib import Path
import pandas as pd

def load_prices(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file containing daily prices.
    Expected columns: 'date', 'symbol', 'price'.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file on disk (e.g., 'data/toy_prices_baseline.csv').

    Returns
    -------
    pd.DataFrame
        A clean, time-sorted table with the same columns, ready for analysis.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If required columns are missing.
    """
    # Convert the text path into a Path object (safer file handling).
    p = Path(csv_path)

    # Guard 1: the file must exist, otherwise stop early with a clear message.
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    # Read the CSV. parse_dates=['date'] turns 'YYYY-MM-DD' text into real dates.
    df = pd.read_csv(p, parse_dates=["date"])

    # Guard 2: schema check. The loader guarantees these columns exist.
    expected = {"date", "symbol", "price"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV must include {expected}, found {set(df.columns)}")

    # Sort to make results deterministic: same input -> same output order.
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Manual quick test when run in Codespaces or locally:
    df = load_prices("data/toy_prices_baseline.csv")
    print("Rows:", len(df), "| Columns:", list(df.columns))
    print(df.head(3))
