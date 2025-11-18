from pathlib import Path
from datetime import datetime
import csv
import importlib
import yaml
import pandas as pd

# P3 components you already own
from bsml.data.loader import load_prices
from bsml.utils.logging import run_id_from_cfg, prepare_outdir, snapshot
from bsml.cost.models import load_cost_config, apply_costs

# P2 hook (baseline policy). P2 will implement generate_trades(prices) -> trades
from bsml.policies import baseline as policy


def main():
    """
    P3 runner: orchestrates a run in a reproducible, config-driven way.

    Order of operations (and why):
    1) read config       -> single source of truth for all parameters
    2) prepare out dir   -> where this run's files will live
    3) snapshot configs  -> reproducibility (log exact YAML + timestamp)
    4) load prices       -> validated input table (schema checked by loader)
    5) load costs cfg    -> numbers used later by cost wiring
    6) call P2 policy    -> get intended trades (P2 responsibility)
    7) apply costs       -> attach execution placeholders (P3 wiring)
    8) write CSVs        -> tidy outputs other roles will consume
    """

    # 1) Read main configuration (config-driven pipeline)
    cfg = yaml.safe_load(Path("configs/run.yaml").read_text())

    # 2) Derive a stable run folder from the config
    run_id = run_id_from_cfg(cfg)
    out_dir = prepare_outdir(cfg["output_dir"], run_id)

    # 3) Snapshot configs early (so even preflight runs are logged)
    snapshot(out_dir)

    # 4) Load input prices (schema: date, symbol, price)
    prices = load_prices(cfg["data"]["prices_csv"])

    # 5) Load cost parameters from YAML
    costs_cfg = load_cost_config(cfg["costs"])

    # 6) Ask P2's baseline policy to generate trades.
    #    Until P2 implements, this will raise NotImplementedError.

    policy_name = cfg.get("policy", "baseline")
    try:
        policy = importlib.import_module(f"bsml.policies.{policy_name}")
    except ModuleNotFoundError as e:
        (out_dir / "STATUS.txt").write_text(
            f"Runner preflight complete, but policy module '{policy_name}' was not found.\n"
            f"Error: {e}\n"
        )
        print(f"Policy module '{policy_name}' not found. Runner preflight is complete.")
        return
    try:
        trades = policy.generate_trades(prices)
    except NotImplementedError as e:
        # Preflight success up to the policy boundary: record status and exit gracefully.
        (out_dir / "STATUS.txt").write_text(
            "Runner preflight complete: waiting for P2 baseline implementation.\n"
            f"{e}\n"
        )
        print("Baseline policy not implemented yet (P2). Runner preflight is complete.")
        return

    # If P2 is implemented, continue:

    # 7) Apply cost wiring (placeholders now; schema is stabilized)
    trades_costed = apply_costs(trades, costs_cfg)

    # 8) Write tidy CSVs for downstream roles
    trades.to_csv(out_dir / "trades_raw.csv", index=False)
    trades_costed.to_csv(out_dir / "trades_costed.csv", index=False)

    print(f"Run completed. Outputs in: {out_dir}")
    
# --- Week 2: append to results/seed_sweep.csv *only after a successful run* ---
results_path = Path("results/seed_sweep.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)

header = [
    "policy", "seed", "split",
    "sharpe", "delta_is_bps", "maxdd", "exposure_diff_pct",
    "timestamp"
]

exists = results_path.exists()
with results_path.open("a", newline="") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(header)
    # NOTE: metrics are placeholders; P5 will compute real values.
    # 'split' can be 'all' (single-split) or a label you and P2 decide.
    w.writerow([
        cfg.get("policy"), cfg.get("seed"), "all",
        0.0, 0.0, 0.0, 0.0,
        datetime.utcnow().isoformat()
    ])
# --- end append block ---

if __name__ == "__main__":
    main()

