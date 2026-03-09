# generate_paper_results.py - Generate all tables and figures for paper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bsml.runner import BacktestRunner
from bsml.adversary.adaptive import AdaptiveAdversary, AdversaryConfig

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

# Create output directories
OUTPUT_DIR = Path('paper/figures')
TABLES_DIR = Path('paper/tables')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TABLES_DIR.mkdir(exist_ok=True, parents=True)


def generate_table1_policy_comparison(results_df: pd.DataFrame):
    """
    Table 1: Performance comparison across policies.
    
    Columns: Policy | Avg IS (bps) | Std IS | Sharpe | Avg Cost (bps)
    """
    summary = results_df.groupby('policy').agg({
        'avg_impl_shortfall_bps': ['mean', 'std'],
        'sharpe_ratio': 'mean',
        'avg_cost_bps': 'mean',
        'win_rate': 'mean'
    }).round(2)
    
    # Flatten column names
    summary.columns = [
        'IS Mean (bps)', 'IS Std (bps)', 'Sharpe', 'Cost (bps)', 'Win Rate'
    ]
    
    # Save
    latex = summary.to_latex(
        caption="Performance comparison across randomization policies",
        label="tab:policy_comparison"
    )
    
    with open(TABLES_DIR / 'table1_comparison.tex', 'w') as f:
        f.write(latex)
    
    summary.to_csv(TABLES_DIR / 'table1_comparison.csv')
    
    print("✓ Table 1 generated")
    return summary


def generate_table2_seed_variance(results_df: pd.DataFrame):
    """
    Table 2: Seed variance analysis.
    
    Shows stability of results across different random seeds.
    """
    # Exclude baseline
    randomized = results_df[results_df['policy'] != 'Baseline']
    
    variance_analysis = randomized.groupby('policy').agg({
        'avg_impl_shortfall_bps': ['mean', 'std', 'min', 'max'],
        'sharpe_ratio': ['mean', 'std']
    }).round(3)
    
    variance_analysis.columns = [
        'IS Mean', 'IS Std', 'IS Min', 'IS Max', 'Sharpe Mean', 'Sharpe Std'
    ]
    
    # Save
    latex = variance_analysis.to_latex(
        caption="Seed variance analysis for randomization policies",
        label="tab:seed_variance"
    )
    
    with open(TABLES_DIR / 'table2_seed_variance.tex', 'w') as f:
        f.write(latex)
    
    variance_analysis.to_csv(TABLES_DIR / 'table2_seed_variance.csv')
    
    print("✓ Table 2 generated")
    return variance_analysis


def generate_figure1_is_distribution(trades_df: pd.DataFrame):
    """
    Figure 1: Distribution of implementation shortfall by policy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    policies = ['Baseline', 'Uniform', 'OU', 'Pink']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (policy, color) in enumerate(zip(policies, colors)):
        ax = axes[idx]
        
        data = trades_df[trades_df['policy'] == policy]['impl_shortfall_bps']
        
        ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Implementation Shortfall (bps)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{policy} Policy')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_is_distribution.png')
    plt.close()
    
    print("✓ Figure 1 generated")


def generate_figure2_adaptive_convergence(adversary_history: pd.DataFrame):
    """
    Figure 2: Adaptive adversary AUC convergence over iterations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: AUC over iterations
    ax1.plot(adversary_history['iteration'], adversary_history['auc'], 
             marker='o', linewidth=2, markersize=8)
    ax1.axhline(0.55, color='red', linestyle='--', label='Target AUC')
    ax1.fill_between(adversary_history['iteration'], 0.53, 0.57, 
                     alpha=0.2, color='red', label='Convergence Zone')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Adversarial Classifier AUC Convergence')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Parameter evolution
    if 'multiplier' in adversary_history.columns:
        ax2.bar(adversary_history['iteration'], adversary_history['multiplier'],
                color=['green' if a == 'INCREASE' else 'blue' if a == 'HOLD' else 'orange'
                       for a in adversary_history['action']])
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Multiplier')
        ax2.set_title('Adaptive Parameter Adjustments')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_adaptive_convergence.png')
    plt.close()
    
    print("✓ Figure 2 generated")


def generate_figure3_cost_breakdown(trades_df: pd.DataFrame):
    """
    Figure 3: Cost component breakdown by policy.
    """
    cost_cols = [
        'cost_commission', 'cost_exchange', 'cost_spread',
        'cost_temp_impact', 'cost_perm_impact', 'cost_slippage'
    ]
    
    # Check which columns exist
    available_cols = [col for col in cost_cols if col in trades_df.columns]
    
    if not available_cols:
        print("⚠ Skipping Figure 3: cost breakdown columns not in data")
        return
    
    summary = trades_df.groupby('policy')[available_cols].mean()
    
    summary.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Policy')
    plt.ylabel('Average Cost ($)')
    plt.title('Transaction Cost Breakdown by Policy')
    plt.legend(title='Cost Component', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_cost_breakdown.png')
    plt.close()
    
    print("✓ Figure 3 generated")


def main():
    """Generate all paper results."""
    print("\n" + "="*60)
    print("GENERATING PAPER RESULTS")
    print("="*60 + "\n")
    
    # Check for results files
    results_files = list(Path('results').glob('metrics_*.csv'))
    trades_files = list(Path('results').glob('trades_*.csv'))
    
    if not results_files or not trades_files:
        print("⚠ No results found. Run backtest first:")
        print("  python -m bsml.runner --data data/prices.csv")
        return
    
    # Load most recent results
    results_df = pd.read_csv(sorted(results_files)[-1])
    trades_df = pd.read_csv(sorted(trades_files)[-1])
    
    print(f"Loaded results: {len(results_df)} runs, {len(trades_df)} trades\n")
    
    # Generate tables
    print("Generating tables...")
    generate_table1_policy_comparison(results_df)
    generate_table2_seed_variance(results_df)
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figure1_is_distribution(trades_df)
    generate_figure3_cost_breakdown(trades_df)
    
    # If adaptive adversary results exist, plot them
    adversary_files = list(Path('results').glob('adversary_history_*.csv'))
    if adversary_files:
        adversary_df = pd.read_csv(sorted(adversary_files)[-1])
        generate_figure2_adaptive_convergence(adversary_df)
    
    print("\n" + "="*60)
    print("✓ ALL RESULTS GENERATED")
    print("="*60)
    print(f"\nTables: {TABLES_DIR}/")
    print(f"Figures: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

