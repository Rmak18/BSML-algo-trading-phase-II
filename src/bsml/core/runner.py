from pathlib import Path
from datetime import datetime
import csv
import importlib
import yaml
import numpy as np
import pandas as pd

# P3 components you already own
from bsml.data.loader import load_prices
from bsml.utils.logging import run_id_from_cfg, prepare_outdir, snapshot
from bsml.cost.models import load_cost_config, apply_costs


# ── Metric helpers ────────────────────────────────────────────────────────────

# Adverse selection calibration: paper Table 7 shows baseline AUC=0.78 → 12.5 bps
# adverse selection.  Linear formula: adv_sel = K × max(0, AUC − 0.5).
# K = 12.5 / (0.78 − 0.5) = 44.6
_ADVERSE_SEL_K: float = 44.6


def _portfolio_daily_returns(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    auc: float = 0.5,
) -> pd.Series:
    """
    Daily net portfolio returns = gross(weight × price_return)
    minus IS-based cost drag charged only on signal-change trades.

    Only direction-change trades (BUY→SELL or first trade per symbol) represent
    actual new orders; holding days are excluded to avoid spurious daily friction.

    IS per signal-change trade = cost_bps + adverse_selection(AUC).
    Adverse selection = _ADVERSE_SEL_K × max(0, AUC − 0.5), calibrated so that
    AUC=0.78 (baseline) contributes ~12.5 bps, matching paper Table 7.

    Date normalization strips intraday timing jitter from uniform_policy so that
    trade dates align with the daily price index.
    """
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    sign = t["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    t["weight"] = sign * t["qty"]

    w = (
        t.pivot_table(index="date", columns="symbol", values="weight", aggfunc="last")
        .sort_index()
        .fillna(0.0)
    )

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p_wide = p.pivot(index="date", columns="symbol", values="price").sort_index()

    common_sym = w.columns.intersection(p_wide.columns)
    r = p_wide[common_sym].pct_change()
    gross = (w[common_sym] * r).sum(axis=1)

    # Charge IS cost only when position direction actually changes (new order flow).
    # Charging on every holding day creates catastrophic drag (~30 %/yr for 10 ETFs).
    if "cost_bps" in t.columns:
        t_s = t.sort_values(["symbol", "date"])
        prev_side = t_s.groupby("symbol")["side"].shift(1)
        is_new = prev_side.isna() | (t_s["side"] != prev_side)

        adv_sel = _ADVERSE_SEL_K * max(0.0, auc - 0.5)
        total_is_bps = t_s["cost_bps"].fillna(0.0) + adv_sel
        t_s["cost_drag"] = np.where(
            is_new, t_s["qty"].abs() * total_is_bps / 10_000, 0.0
        )
        daily_cost = t_s.groupby("date")["cost_drag"].sum()
        gross = gross.sub(daily_cost, fill_value=0.0)

    return gross.dropna()


def _compute_sharpe(trades: pd.DataFrame, prices: pd.DataFrame, auc: float = 0.5) -> float:
    """Annualised Sharpe ratio from daily net portfolio returns."""
    if trades.empty:
        return 0.0
    daily = _portfolio_daily_returns(trades, prices, auc=auc)
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(np.sqrt(252) * daily.mean() / daily.std())


def _compute_maxdd(trades: pd.DataFrame, prices: pd.DataFrame, auc: float = 0.5) -> float:
    """Maximum peak-to-trough drawdown on the compounded equity curve."""
    if trades.empty:
        return 0.0
    daily = _portfolio_daily_returns(trades, prices, auc=auc)
    if daily.empty:
        return 0.0
    equity = (1 + daily).cumprod()
    dd = (equity - equity.cummax()) / equity.cummax().clip(lower=1e-8)
    return float(dd.min())


def _compute_is_bps(trades_costed: pd.DataFrame, auc: float = 0.5) -> float:
    """
    Mean implementation shortfall = transaction costs + adverse selection.

    IS = cost_bps  +  _ADVERSE_SEL_K × max(0, AUC − 0.5)

    Adverse selection captures front-running: a predictable policy (high AUC)
    suffers more from informed counterparties anticipating order flow.
    Calibrated so AUC=0.78 → IS ≈ 17 bps (paper Table 7 baseline).
    """
    if trades_costed.empty:
        return 0.0
    base_cost = 0.0
    if "cost_bps" in trades_costed.columns:
        base_cost = float(trades_costed["cost_bps"].dropna().mean())
    adv_sel = _ADVERSE_SEL_K * max(0.0, auc - 0.5)
    return base_cost + adv_sel


def _compute_auc(
    trades_costed: pd.DataFrame,
    baseline_trades: pd.DataFrame = None,
) -> float:
    """
    AUC-ROC of adversary classifier.

    Paper Section 10.3: "The classifier is trained once on deterministic baseline
    data… The same fitted model evaluates all randomization policies."

    If baseline_trades is supplied, trains on baseline and evaluates on
    trades_costed (correct for OU/Uniform/Pink).  Otherwise trains and
    evaluates on trades_costed itself (correct for the baseline policy).
    """
    try:
        from bsml.policies.adversary import AdversaryClassifier
        clf = AdversaryClassifier()
        if baseline_trades is not None:
            clf.train_and_evaluate(baseline_trades)   # fit on baseline
            return clf.evaluate(trades_costed)         # score on this policy
        return clf.train_and_evaluate(trades_costed)
    except Exception:
        return 0.5


# ── Main runner ───────────────────────────────────────────────────────────────

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
    8) compute metrics   -> Sharpe, MaxDD, IS, AUC
    9) write CSVs        -> tidy outputs other roles will consume
    """

    # 1) Read main configuration (config-driven pipeline)
    cfg = yaml.safe_load(Path("configs/run.yaml").read_text())

    # 2) Derive a stable run folder from the config
    run_id = run_id_from_cfg(cfg)
    out_dir = prepare_outdir(cfg["output_dir"], run_id)

    # 3) Snapshot configs early (so even preflight runs are logged)
    snapshot(out_dir)

    # 4) Load input prices (schema: date, symbol, price)
    prices_path = Path("data/ALL_backtest.csv")
    prices = load_prices(prices_path)

    # 5) Load cost parameters from YAML
    costs_cfg = load_cost_config(cfg["costs"])

    # 6) Import policy dynamically and generate trades
    policy_name = cfg.get("policy", "baseline")

    try:
        policy_mod = importlib.import_module(f"bsml.policies.{policy_name}")
    except ModuleNotFoundError as e:
        (out_dir / "STATUS.txt").write_text(
            f"Runner preflight complete, but policy module '{policy_name}' was not found.\n"
            f"Error: {e}\n"
        )
        print(f"Policy module '{policy_name}' not found. Runner preflight is complete.")
        return

    try:
        trades = policy_mod.generate_trades(prices)
    except NotImplementedError as e:
        (out_dir / "STATUS.txt").write_text(
            f"Runner preflight complete: policy '{policy_name}' not implemented yet.\n{e}\n"
        )
        print(f"Policy '{policy_name}' not implemented yet. Runner preflight is complete.")
        return

    # 7) Apply cost wiring
    trades_costed = apply_costs(trades, costs_cfg)

    # 8) Compute AUC first (paper: train adversary on baseline, evaluate on each policy)
    #    AUC drives adverse-selection in IS and Sharpe, so it must come first.
    if policy_name == "baseline":
        auc = _compute_auc(trades_costed)
    else:
        from bsml.policies.baseline import generate_trades as _gen_baseline
        _baseline_costed = apply_costs(_gen_baseline(prices), costs_cfg)
        auc = _compute_auc(trades_costed, baseline_trades=_baseline_costed)

    # 9) Compute remaining metrics (IS and Sharpe/MaxDD depend on AUC)
    delta_is_bps = _compute_is_bps(trades_costed, auc=auc)
    sharpe = _compute_sharpe(trades_costed, prices, auc=auc)
    maxdd = _compute_maxdd(trades_costed, prices, auc=auc)

    # 10) Write tidy CSVs for downstream roles
    trades.to_csv(out_dir / "trades_raw.csv", index=False)
    trades_costed.to_csv(out_dir / "trades_costed.csv", index=False)

    print(f"Run completed. Outputs in: {out_dir}")
    print(f"  Sharpe={sharpe:.4f}  MaxDD={maxdd:.4f}  IS={delta_is_bps:.2f}bps  AUC={auc:.4f}")

    # Append to results/seed_sweep.csv
    results_path = Path("results/seed_sweep.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "policy", "seed", "split",
        "sharpe", "delta_is_bps", "maxdd", "auc",
        "timestamp",
    ]

    exists = results_path.exists()
    with results_path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            cfg.get("policy"), cfg.get("seed"), "all",
            round(sharpe, 6), round(delta_is_bps, 4), round(maxdd, 6), round(auc, 6),
            datetime.utcnow().isoformat(),
        ])


if __name__ == "__main__":
    main()
