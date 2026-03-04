#!/usr/bin/env python
"""
Run all original backtests (policies × seeds) and build a clean CSV
for the adversary classifier.

Usage:
    python run_all_backtests_and_export.py
"""

from pathlib import Path
import os
import sys
import subprocess
import json

import pandas as pd
import yaml


# -----------------------------
# Config
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "run.yaml"
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "runs" / "runs"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "trades_for_adversary.csv"

# Policies = randomization strategies from your repo
POLICIES = ["baseline", "uniform_policy", "ou", "pink"]

# You can change or extend this
SEEDS = [101, 202, 303]


# -----------------------------
# Helpers
# -----------------------------

def update_run_yaml(policy: str, seed: int):
    """Modify configs/run.yaml in-place for this policy+seed."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["policy"] = policy
    cfg["seed"] = int(seed)
    # keep other fields (data, costs, output_dir, etc.) as is
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_single_backtest():
    """
    Call the original runner: python -m bsml.core.runner
    with PYTHONPATH set to ./src so imports work.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    cmd = [sys.executable, "-m", "bsml.core.runner"]
    print("  -> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def collect_all_trades():
    """
    Walk outputs/runs/*, read trades_costed.csv and run.json,
    and build one big DataFrame with policy / seed / run_id columns added.
    """
    if not OUTPUTS_ROOT.exists():
        raise SystemExit(f"No runs found in {OUTPUTS_ROOT}. Did you run the backtests?")

    all_dfs = []

    for run_dir in OUTPUTS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue

        run_json_path = run_dir / "run.json"
        trades_costed_path = run_dir / "trades_costed.csv"

        if not run_json_path.exists() or not trades_costed_path.exists():
            # skip incomplete runs
            continue

        # Load meta
        meta = json.loads(run_json_path.read_text())
        run_cfg = meta.get("run_yaml", {})
        policy = run_cfg.get("policy")
        seed = run_cfg.get("seed")

        # Load trades
        df = pd.read_csv(trades_costed_path)

        # Add identifiers for P6
        df["policy"] = policy
        df["seed"] = seed
        df["run_id"] = run_dir.name

        all_dfs.append(df)

    if not all_dfs:
        raise SystemExit("No trades_costed.csv files found under outputs/runs/.")

    big_df = pd.concat(all_dfs, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    big_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[OK] Wrote aggregated CSV to: {RESULTS_CSV}")
    print(f"Rows: {len(big_df)}, columns: {len(big_df.columns)}")


# -----------------------------
# Main
# -----------------------------

def main():
    print("=== Running policy × seed matrix ===")
    for pol in POLICIES:
        for seed in SEEDS:
            print(f"\n=== policy={pol}, seed={seed} ===")
            update_run_yaml(pol, seed)
            try:
                run_single_backtest()
            except subprocess.CalledProcessError as e:
                print(f"  -> Run failed for policy={pol}, seed={seed}: {e}")
                # continue so one bad combo doesn't kill everything

    print("\n=== Aggregating trades_costed into one CSV ===")
    collect_all_trades()


if __name__ == "__main__":
    main()
