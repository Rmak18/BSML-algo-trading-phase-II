"""
50-seed Monte Carlo sweep across all four execution policies.

Outputs mean ± std for Sharpe, MaxDD, IS, and AUC.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import yaml
from collections import defaultdict

from bsml.data.loader import load_prices
from bsml.cost.models import load_cost_config, apply_costs
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.policies.adversary import AdversaryClassifier
from bsml.policies.uniform_policy import UniformPolicy
from bsml.policies.ou_policy import OUPolicy
from bsml.policies.pink_policy import PinkPolicy
from bsml.core.runner import (
    _compute_sharpe, _compute_maxdd, _compute_is_bps, _ADVERSE_SEL_K
)

N_SEEDS = 50
SEEDS = list(range(N_SEEDS))

cfg = yaml.safe_load(Path("configs/run.yaml").read_text())
costs_cfg = load_cost_config(cfg["costs"])
prices = load_prices("data/ALL_backtest.csv")

# Baseline is deterministic — compute once
baseline_trades = baseline_generate(prices)
baseline_costed = apply_costs(baseline_trades, costs_cfg)

# Train adversary once on baseline (paper spec: train on baseline, evaluate all)
base_clf = AdversaryClassifier()
baseline_auc = base_clf.train_and_evaluate(baseline_costed)

print(f"Baseline AUC (trained): {baseline_auc:.4f}")
print(f"Baseline Sharpe:        {_compute_sharpe(baseline_costed, prices, auc=baseline_auc):.4f}")
print(f"Baseline MaxDD:         {_compute_maxdd(baseline_costed, prices, auc=baseline_auc):.4f}")
print(f"Baseline IS (bps):      {_compute_is_bps(baseline_costed, auc=baseline_auc):.2f}")
print()

results = defaultdict(lambda: defaultdict(list))

for seed in SEEDS:
    if seed % 10 == 0:
        print(f"  seed {seed}/{N_SEEDS}...")

    policies = {
        "uniform": UniformPolicy(seed=seed),
        "ou":      OUPolicy(seed=seed),
        "pink":    PinkPolicy(seed=seed),
    }

    for name, policy in policies.items():
        trades = policy.generate_trades(prices)
        costed = apply_costs(trades, costs_cfg)

        auc = base_clf.evaluate(costed)
        sharpe = _compute_sharpe(costed, prices, auc=auc)
        maxdd  = _compute_maxdd(costed, prices, auc=auc)
        is_bps = _compute_is_bps(costed, auc=auc)

        results[name]["auc"].append(auc)
        results[name]["sharpe"].append(sharpe)
        results[name]["maxdd"].append(maxdd)
        results[name]["is_bps"].append(is_bps)

print()
print(f"{'Policy':<10}  {'Sharpe':>14}  {'MaxDD':>14}  {'IS (bps)':>14}  {'AUC':>14}")
print("-" * 75)

for name in ["uniform", "ou", "pink"]:
    r = results[name]
    sharpe_m, sharpe_s = np.mean(r["sharpe"]), np.std(r["sharpe"])
    maxdd_m,  maxdd_s  = np.mean(r["maxdd"]),  np.std(r["maxdd"])
    is_m,     is_s     = np.mean(r["is_bps"]), np.std(r["is_bps"])
    auc_m,    auc_s    = np.mean(r["auc"]),    np.std(r["auc"])

    print(
        f"{name:<10}  "
        f"{sharpe_m:+.4f}±{sharpe_s:.4f}  "
        f"{maxdd_m:+.4f}±{maxdd_s:.4f}  "
        f"{is_m:6.2f}±{is_s:.2f}    "
        f"{auc_m:.4f}±{auc_s:.4f}"
    )
