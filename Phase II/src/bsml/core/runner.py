"""
Parallel runner for Phase II
"""

import csv
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool


# ── Configuration 
POLICIES = ["baseline", "ou", "uniform_policy", "pink"]
N_SEEDS = 50


"""
Function to run a single simulation.
Run a single simulation for a given (seed, policy) combination.
Returns a dictionary with all results.
"""
def run_single(args: tuple) -> dict: 
    seed, policy_name, prices_path, costs_path = args

    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    phase1_src = str(Path(__file__).resolve().parents[4] / "Phase I" / "src")
    if phase1_src not in sys.path:
        sys.path.insert(0, phase1_src)

    from bsml.data.loader import load_prices
    from bsml.cost.models import load_cost_config, apply_costs
    from bsml.policies.adversary import AdversaryClassifier

    np.random.seed(seed)

    try:
        # Load prices and costs
        prices = load_prices(prices_path)
        costs_cfg = load_cost_config(costs_path)

        # Generate trades
        policy_mod = importlib.import_module(f"bsml.policies.{policy_name}")
        trades = policy_mod.generate_trades(prices)

        # Apply transaction costs to trades
        trades_costed = apply_costs(trades, costs_cfg)
        """
        Calculate AUC: how well the adversary can predict trades
        If it's the baseline, train and evaluate on the same dataset
        Otherwise, train on the baseline and evaluate on the randomized policy
        """
        if policy_name == "baseline":
            auc = AdversaryClassifier().train_and_evaluate(trades_costed)
        else:
            from bsml.policies.baseline import generate_trades as gen_baseline
            baseline_trades = apply_costs(gen_baseline(prices), costs_cfg)
            clf = AdversaryClassifier()
            clf.train_and_evaluate(baseline_trades)
            auc = clf.evaluate(trades_costed)

        # Calculate metrics
        from bsml.core.runner import _compute_sharpe, _compute_maxdd, _compute_is_bps
        sharpe = _compute_sharpe(trades_costed, prices, auc=auc)
        maxdd  = _compute_maxdd(trades_costed, prices, auc=auc)
        is_bps = _compute_is_bps(trades_costed, auc=auc)

        return {
            "policy":      policy_name,
            "seed":        seed,
            "sharpe":      round(sharpe, 6),
            "maxdd":       round(maxdd, 6),
            "is_bps":      round(is_bps, 4),
            "auc":         round(auc, 6),
            "adv_pnl":     None,
            "net_leakage": None,
            "status":      "ok",
            "timestamp":   datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {
            "policy":    policy_name,
            "seed":      seed,
            "status":    "error",
            "error_msg": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ── Main function
def main():
    """
    Build all (seed, policy) combinations, run them in parallel and save the results in a CSV.
    """

    base     = Path(__file__).resolve().parents[4]
    phase1   = base / "Phase I"
    prices_path = phase1 / "data" / "ALL_backtest.csv"
    costs_path  = phase1 / "configs" / "costs.yaml"

    print(f"Prezzi: {prices_path}")
    print(f"Costi:  {costs_path}")
    print(f"Policy: {POLICIES}")
    print(f"Seeds:  0 → {N_SEEDS - 1}")
    print(f"Totale run: {len(POLICIES) * N_SEEDS}\n")

    # Build the list of all runs
    all_args = [
        (seed, policy, str(prices_path), str(costs_path))
        for policy in POLICIES
        for seed in range(N_SEEDS)
    ]

    # Run all in parallel with 4 processes at the same time
    print("Avvio simulazione parallela...")
    with Pool(processes=4) as pool:
        results = pool.map(run_single, all_args)

    # Save the results in a CSV
    out_path = Path(__file__).resolve().parents[3] / "results" / "seed_sweep_phaseII.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Summary
    ok = df[df["status"] == "ok"]
    errors = df[df["status"] == "error"]
    print(f"\nCompletati: {len(ok)}/{len(results)} run")
    if len(errors) > 0:
        print(f"Errori: {len(errors)} run falliti")
        print(errors[["policy", "seed", "error_msg"]])
    print(f"\nRisultati salvati in: {out_path}")

    if len(ok) > 0:
        print("\nMedia per policy:")
        print(ok.groupby("policy")[["sharpe", "maxdd", "is_bps", "auc"]].mean().round(4))


if __name__ == "__main__":
    main()