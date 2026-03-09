"""
P5 RCT-style pilot summary over trades_for_adversary.csv

Reads:
    <repo_root>/results/trades_for_adversary.csv

Writes:
    <repo_root>/results/paper/rct_pilot_summary.csv
"""

from pathlib import Path
import pandas as pd

from rct_utils import (
    implementation_shortfall,
    bootstrap_mean_ci,
)


# ---------------------------------------------------------------------
# Paths: repo root / results / paper
# ---------------------------------------------------------------------

# this file:   src/bsml/analysis/rct_pilot.py
# parents[0] = analysis
# parents[1] = bsml
# parents[2] = src
# parents[3] = repo root  ✅
REPO_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = REPO_ROOT / "results" / "trades_for_adversary.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "rct_pilot_summary.csv"


def main() -> None:
    print(f"[INFO] Loading trades from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Basic sanity checks
    required_cols = {"ref_price", "net_price", "side", "policy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trades CSV: {missing}")

    # Implementation shortfall per row
    df["IS"] = implementation_shortfall(
        df["ref_price"].values,
        df["net_price"].values,
        df["side"].values,
    )

    # -----------------------------------------------------------------
    # P5-style summary: bootstrap mean IS per policy
    # -----------------------------------------------------------------
    rows = []
    for pol in sorted(df["policy"].unique()):
        sub = df[df["policy"] == pol]["IS"].values
        stats = bootstrap_mean_ci(sub, n_boot=2000, ci=0.95, seed=17)
        rows.append(
            {
                "policy": pol,
                "n_trades": int(sub.size),
                "is_mean": stats["mean"],
                "is_ci_low": stats["low"],
                "is_ci_high": stats["high"],
                "is_se": stats["se"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Wrote RCT pilot summary to: {OUTPUT_CSV}")
    print(summary)


if __name__ == "__main__":
    main()
