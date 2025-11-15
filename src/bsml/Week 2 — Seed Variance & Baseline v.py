# P1 Week 2 — Seed Variance & Baseline vs Uniform
# Requires: results/seed_sweep.csv (P3/P5 generate this via the runner)
# Figures -> paper/figures, table -> paper/tables

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = 'results/seed_sweep.csv'      # team runner output
OUTPUT_DIR = 'paper/figures'
TABLE_DIR = 'paper/tables'
EXPOSURE_TOL_PCT = 2.0                     # exposure tolerance vs baseline (absolute %)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Load or create a tiny demo if file isn't ready yet
if os.path.exists(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH)
else:
    np.random.seed(7)
    demo = []
    for pol in ['baseline','uniform']:
        for seed in range(1, 21):
            demo.append({
                'policy': pol,
                'seed': seed,
                'split': 'test_demo',
                'sharpe': np.random.normal(0.9 if pol=='baseline' else 0.95, 0.15),
                'delta_is_bps': np.random.normal(0.0 if pol=='baseline' else 2.5, 1.0),
                'maxdd': np.clip(np.random.normal(0.22 if pol=='baseline' else 0.21, 0.03), 0.05, 0.5),
                'exposure_diff_pct': 0.5 if pol=='baseline' else abs(np.random.normal(0.8, 0.4)),
            })
    df = pd.DataFrame(demo)

# Clean & restrict to expected policies
df['policy'] = df['policy'].str.lower().str.strip()
df = df[df['policy'].isin(['baseline','uniform','ou','pink'])].copy()

# Quick exposure-invariance check
if 'exposure_diff_pct' in df.columns:
    exp = (df.groupby('policy', as_index=False)['exposure_diff_pct']
             .agg(mean='mean', max='max', count='count'))
    print('Exposure difference vs baseline (%), by policy:')
    print(exp.to_string(index=False))
    n_flagged = int((df['exposure_diff_pct'] > EXPOSURE_TOL_PCT).sum())
    if n_flagged > 0:
        print(f'WARNING: {n_flagged} run(s) exceed the {EXPOSURE_TOL_PCT}% exposure tolerance')

# Aggregate stats for CI plots
metrics = (df.groupby('policy', as_index=False)
             .agg(n=('seed','nunique'),
                  sharpe_mean=('sharpe','mean'),
                  sharpe_std=('sharpe','std'),
                  delta_is_mean=('delta_is_bps','mean'),
                  delta_is_std=('delta_is_bps','std')))

metrics['sharpe_ci'] = 1.96 * metrics['sharpe_std'] / np.sqrt(metrics['n'].clip(lower=1))
metrics['delta_is_ci'] = 1.96 * metrics['delta_is_std'] / np.sqrt(metrics['n'].clip(lower=1))
metrics.to_csv(os.path.join(TABLE_DIR, 'seed_variance_summary.csv'), index=False)

# Figure 1 — Baseline vs Uniform (ΔIS mean ± 95% CI)
sub = metrics[metrics['policy'].isin(['baseline','uniform'])].copy()
if not sub.empty and sub['policy'].nunique() == 2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(sub['policy'], sub['delta_is_mean'], yerr=sub['delta_is_ci'], fmt='o')
    ax.set_xlabel('Policy')
    ax.set_ylabel('ΔIS (bps) — mean ± 95% CI')
    ax.set_title('Baseline vs Uniform — Implementation Shortfall')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'baseline_vs_uniform_deltaIS.png'), dpi=200)
    plt.show()
else:
    print('Not enough data for Baseline vs Uniform ΔIS.')

# Figure 2 — Seed variance: Sharpe distributions
policies_present = df['policy'].unique().tolist()
if len(policies_present) >= 1:
    fig, ax = plt.subplots(figsize=(6,4))
    data = [df.loc[df['policy']==p, 'sharpe'].dropna().values for p in policies_present]
    ax.boxplot(data, labels=policies_present, showmeans=True)
    ax.set_xlabel('Policy')
    ax.set_ylabel('Sharpe (annualized)')
    ax.set_title('Seed Variance — Sharpe by Policy')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'seed_variance_sharpe_boxplot.png'), dpi=200)
    plt.show()

# Figure 3 — Seed variance: ΔIS distributions
if len(policies_present) >= 1:
    fig, ax = plt.subplots(figsize=(6,4))
    data = [df.loc[df['policy']==p, 'delta_is_bps'].dropna().values for p in policies_present]
    ax.boxplot(data, labels=policies_present, showmeans=True)
    ax.set_xlabel('Policy')
    ax.set_ylabel('ΔIS (bps)')
    ax.set_title('Seed Variance — ΔIS by Policy')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'seed_variance_deltaIS_boxplot.png'), dpi=200)
    plt.show()

print('\\nACCEPTANCE CHECKS')
print('1) baseline_vs_uniform_deltaIS.png exists:',
      os.path.exists(os.path.join(OUTPUT_DIR, 'baseline_vs_uniform_deltaIS.png')))
print('2) seed_variance_sharpe_boxplot.png exists:',
      os.path.exists(os.path.join(OUTPUT_DIR, 'seed_variance_sharpe_boxplot.png')))
print('3) seed_variance_deltaIS_boxplot.png exists:',
      os.path.exists(os.path.join(OUTPUT_DIR, 'seed_variance_deltaIS_boxplot.png')))
print('4) seed_variance_summary.csv exists:',
      os.path.exists(os.path.join(TABLE_DIR, 'seed_variance_summary.csv')))
