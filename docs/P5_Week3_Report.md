# P5 — Week 3 Report  
**Author:** Preslav Georgiev  
**Project:** Randomness in Algorithmic Trading — BSML  
**Date:** November 2025  

---

## Objective
Implement and integrate a **Randomized Controlled Trial (RCT) pilot** for comparing **Early** vs **Late** execution timing.  
Compute the **Implementation Shortfall (IS)** per trade and the **difference in shortfall (ΔIS = ISₑₐᵣly − ISₗₐₜₑ)** across paired trades.  
Provide bootstrap-based confidence intervals and a reproducible CLI for future experiments.

---

## Implementation Overview

### 🧩 Added Modules

| File | Purpose |
|------|----------|
| `src/bsml/analysis/rct_utils.py` | Utility functions: compute IS, ΔIS per trade pair, and bootstrap confidence intervals. |
| `src/bsml/analysis/rct_pilot.py` | CLI script: loads trade data, applies `rct_utils`, and outputs overall/by-symbol/by-regime ΔIS tables and a Markdown report. |
| `src/bsml/configs/p5_rct_pilot.yaml` | YAML config mapping input column names and bootstrap options. |
| `data/sample_trades.csv` | Small synthetic dataset (4 trade pairs) used for validation. |
| `results/p5_week3/` | Automatically created folder for CSV outputs and report. |

All code was integrated into the **existing `analysis/` package**, not a new folder.  
No modifications were made to `core`, `policies`, or `runner` modules.

---

## 🔧 Core Logic

### 1. Implementation Shortfall
```python
def implementation_shortfall(ref_price, exec_price, side):
    # buy:  (exec - ref) / ref
    # sell: (ref - exec) / ref
```
Measures execution efficiency relative to the benchmark.  
Lower IS ⇒ cheaper buys / richer sells.

---

### 2. Pairing & ΔIS
```python
def delta_is_pairs(df_pairs, early_label="early", late_label="late"):
    # compute IS per row, pivot by trade_id, subtract early − late
```
Each `trade_id` corresponds to a matched Early/Late pair.  
Positive ΔIS → Early performs better; Negative ΔIS → Late performs better.

---

### 3. Bootstrap Confidence Intervals
```python
def bootstrap_mean_ci(x, n_boot=2000, ci=0.95, seed=17):
    # percentile bootstrap for mean(ΔIS)
```
Provides mean, lower/upper CI bounds, and standard error.  
Used for overall, per-symbol, and per-regime summaries.

---

### 4. CLI Flow (`rct_pilot.py`)
1. Load YAML config and input CSV.  
2. Validate/rename columns per `input_schema`.  
3. Drop incomplete trade pairs.  
4. Compute ΔIS overall and by groups.  
5. Bootstrap CIs.  
6. Write results:
   - `delta_is_overall.csv`
   - `delta_is_by_symbol.csv`
   - `delta_is_by_regime.csv` (if present)
   - `p5_rct_pilot_report.md` (formatted Markdown summary)

Run command:
```bash
export PYTHONPATH="$PWD/src"
python -m bsml.analysis.rct_pilot   --config src/bsml/configs/p5_rct_pilot.yaml   --input  data/sample_trades.csv   --outdir results/p5_week3
```

---

## ✅ Verification (Synthetic Run)

Using `data/sample_trades.csv` (8 rows, 4 trade pairs):

| File | Key Result |
|------|-------------|
| `delta_is_overall.csv` | ΔIS ≈ −0.00099 (≈ −9.9 bps) → Late outperformed Early. |
| `delta_is_by_symbol.csv` | Both symbols show ΔIS < 0. |
| `delta_is_by_regime.csv` | Both regimes show ΔIS < 0. |
| `p5_rct_pilot_report.md` | Markdown summary generated automatically. |

This confirms the pipeline runs end-to-end and signs behave correctly.

---

## 🔗 Integration with the Main Framework

- **Runner (P3)** will later emit a per-trade CSV with both Early/Late arms using the same schema as `sample_trades.csv`.  
- **P5** (this work) analyzes that output via the RCT pipeline.  
- **P4 Policies** and **P6 Adversary** remain unchanged.  
- All RCT results are written under `results/p5_week3/` for inclusion in the paper.

---

## 🧠 Summary of Contributions
1. Implemented the entire RCT analysis pipeline (ΔIS + bootstrap CIs).  
2. Integrated cleanly into `src/bsml/analysis/`.  
3. Verified functionality on a synthetic dataset.  
4. Documented usage and interpretation (Early > Late ↔ ΔIS > 0).  
5. Prepared for immediate use when real pilot data becomes available.

---

## 📈 Next Steps
- Use real randomized trade logs from the runner once available.  
- Run the same CLI to produce production-grade ΔIS tables and CIs.  
- Add optional plotting of bootstrap distributions and per-symbol bars in Week 4.

---

**Status:** ✅ Implementation complete — awaiting real data for quantitative evaluation.
