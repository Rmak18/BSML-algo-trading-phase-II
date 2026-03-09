"""
CLI to compute ΔIS (Early vs Late) with bootstrap CIs.
"""
import argparse, os, sys, yaml
import pandas as pd
from .rct_utils import delta_is_pairs, bootstrap_mean_ci

def _load_config(p):
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def _ensure_cols(df, schema_map):
    # Rename columns to canonical names if needed
    rename_map = {}
    for want, have in schema_map.items():
        if have in df.columns and have != want:
            rename_map[have] = want
    df = df.rename(columns=rename_map)
    # Check presence
    required = ["trade_id","symbol","arm","side","ref_price","exec_price","qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")
    return df

def summarize_group(delta_series, n_boot, ci, seed):
    s = delta_series.dropna().values
    return bootstrap_mean_ci(s, n_boot=n_boot, ci=ci, seed=seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--input", required=True, help="Path to input CSV with paired early/late rows")
    ap.add_argument("--outdir", required=True, help="Output directory for CSVs and report")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    df = _ensure_cols(df, cfg.get("input_schema", {}))

    early_label = cfg["options"]["arm_early"]
    late_label  = cfg["options"]["arm_late"]
    n_boot = int(cfg["options"]["n_boot"])
    ci     = float(cfg["options"]["ci"])
    seed   = int(cfg["options"]["seed"])

    # Drop incomplete pairs (optional)
    if cfg["options"].get("drop_incomplete_pairs", True):
        # Keep only trade_ids that have both arms present
        counts = df.groupby(["trade_id","arm"]).size().unstack(fill_value=0)
        good_ids = counts[(counts.get(early_label,0)>0) & (counts.get(late_label,0)>0)].index.get_level_values(0).unique()
        df = df[df["trade_id"].isin(good_ids)].copy()

    # Compute ΔIS per pair
    delta_all = delta_is_pairs(df, early_label=early_label, late_label=late_label)

    # Overall
    overall = summarize_group(delta_all, n_boot, ci, seed)
    overall_df = pd.DataFrame([overall])
    overall_path = os.path.join(args.outdir, "delta_is_overall.csv")
    overall_df.to_csv(overall_path, index=False)

    # By symbol
    by_symbol_rows = []
    for sym, grp in df.groupby("symbol"):
        d = delta_is_pairs(grp, early_label=early_label, late_label=late_label)
        stats = summarize_group(d, n_boot, ci, seed)
        stats["symbol"] = sym
        by_symbol_rows.append(stats)
    by_symbol = pd.DataFrame(by_symbol_rows)
    by_symbol_path = os.path.join(args.outdir, "delta_is_by_symbol.csv")
    by_symbol.to_csv(by_symbol_path, index=False)

    # By regime (optional)
    by_regime_path = None
    if "regime" in df.columns:
        by_regime_rows = []
        for reg, grp in df.groupby("regime"):
            d = delta_is_pairs(grp, early_label=early_label, late_label=late_label)
            stats = summarize_group(d, n_boot, ci, seed)
            stats["regime"] = reg
            by_regime_rows.append(stats)
        by_reg = pd.DataFrame(by_regime_rows)
        by_regime_path = os.path.join(args.outdir, "delta_is_by_regime.csv")
        by_reg.to_csv(by_regime_path, index=False)

    # Small human-readable report
    rep_lines = []
    rep_lines.append("# P5 Week 3 — RCT Pilot Results")
    rep_lines.append("## Overall ΔIS (Early - Late)")
    rep_lines.append(overall_df.to_markdown(index=False))
    rep_lines.append("")
    rep_lines.append("## By Symbol")
    rep_lines.append(by_symbol.to_markdown(index=False))
    rep_lines.append("")
    if by_regime_path:
        rep_lines.append("## By Regime")
        rep_lines.append(by_reg.to_markdown(index=False))
        rep_lines.append("")

    with open(os.path.join(args.outdir, "p5_rct_pilot_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep_lines))

    print(f"Wrote:\n- {overall_path}\n- {by_symbol_path}")
    if by_regime_path:
        print(f"- {by_regime_path}")
    print(f"- {os.path.join(args.outdir, 'p5_rct_pilot_report.md')}")

if __name__ == "__main__":
    main()
