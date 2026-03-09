"""
Visualizations for Regression-Based Adaptive Experiment

Plots:
1. MAE% over iterations (lower is more exploitable)
2. Exploitability fraction over iterations
3. Parameter evolution
4. Combined summary

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_mae_over_iterations(
    mae_history: Dict[str, List[float]],
    threshold: float = 0.5,
    output_path: str = None,
    show: bool = True
):
    """
    Plot MAE% over iterations for all three policies.
    
    Lower MAE% = more exploitable (bad)
    Higher MAE% = safe from exploitation (good)
    
    Args:
        mae_history: Dictionary with 'pink', 'ou', 'uniform' keys
        threshold: MAE% threshold (transaction cost)
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(len(mae_history['pink']))
    
    # Plot each policy
    ax.plot(iterations, mae_history['pink'], 'o-', label='Pink Noise', 
            linewidth=2, markersize=8, color='#E91E63')
    ax.plot(iterations, mae_history['ou'], 's-', label='OU Process', 
            linewidth=2, markersize=8, color='#2196F3')
    ax.plot(iterations, mae_history['uniform'], '^-', label='Uniform', 
            linewidth=2, markersize=8, color='#4CAF50')
    
    # Threshold line (safe zone)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Safety Threshold ({threshold}%)', alpha=0.7)
    ax.fill_between(iterations, 0, threshold, alpha=0.1, color='red', 
                     label='Exploitable Zone')
    ax.fill_between(iterations, threshold, ax.get_ylim()[1], alpha=0.1, color='green', 
                     label='Safe Zone')
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE% (Mean Absolute Error %)', fontsize=12, fontweight='bold')
    ax.set_title('Price Prediction Error Over Iterations (Lower = More Exploitable)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for exploitable iterations
    for i, (pink, ou, uniform) in enumerate(zip(
        mae_history['pink'], 
        mae_history['ou'], 
        mae_history['uniform']
    )):
        if pink < threshold:
            ax.annotate('⚠', xy=(i, pink), xytext=(0, -15), 
                       textcoords='offset points', fontsize=14, color='#E91E63')
        if ou < threshold:
            ax.annotate('⚠', xy=(i, ou), xytext=(0, -15), 
                       textcoords='offset points', fontsize=14, color='#2196F3')
        if uniform < threshold:
            ax.annotate('⚠', xy=(i, uniform), xytext=(0, -15), 
                       textcoords='offset points', fontsize=14, color='#4CAF50')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_exploitability_over_iterations(
    exploit_history: Dict[str, List[float]],
    output_path: str = None,
    show: bool = True
):
    """
    Plot fraction of exploitable trades over iterations.
    
    Args:
        exploit_history: Dictionary with exploitability fractions
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(len(exploit_history['pink']))
    
    # Convert to percentages
    pink_pct = [x * 100 for x in exploit_history['pink']]
    ou_pct = [x * 100 for x in exploit_history['ou']]
    uniform_pct = [x * 100 for x in exploit_history['uniform']]
    
    # Plot
    ax.plot(iterations, pink_pct, 'o-', label='Pink Noise', 
            linewidth=2, markersize=8, color='#E91E63')
    ax.plot(iterations, ou_pct, 's-', label='OU Process', 
            linewidth=2, markersize=8, color='#2196F3')
    ax.plot(iterations, uniform_pct, '^-', label='Uniform', 
            linewidth=2, markersize=8, color='#4CAF50')
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exploitable Trades (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fraction of Trades Predictable Within 0.5% (Lower = Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_parameter_evolution_regression(
    param_history: Dict[str, List[Dict]],
    output_path: str = None,
    show: bool = True
):
    """
    Plot parameter evolution for regression experiment.
    
    Args:
        param_history: Dictionary with parameter history for each policy
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    iterations = range(len(param_history['pink']))
    
    # ========== Pink Noise ==========
    ax = axes[0]
    pink_price_scales = [p['price_scale'] for p in param_history['pink']]
    
    ax.plot(iterations, pink_price_scales, 'o-', linewidth=2, markersize=8, 
            color='#E91E63', label='price_scale')
    ax.set_ylabel('Price Scale', fontsize=11, fontweight='bold')
    ax.set_title('Pink Noise: price_scale Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== OU Process ==========
    ax = axes[1]
    ou_price_scales = [p['price_scale'] for p in param_history['ou']]
    
    ax.plot(iterations, ou_price_scales, 'o-', linewidth=2, markersize=8, 
            color='#2196F3', label='price_scale')
    ax.set_ylabel('Price Scale', fontsize=11, fontweight='bold')
    ax.set_title('OU Process: price_scale Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== Uniform ==========
    ax = axes[2]
    uniform_price_noise = [p['price_noise'] for p in param_history['uniform']]
    uniform_time_noise = [p['time_noise_minutes'] / 60.0 for p in param_history['uniform']]
    
    ax.plot(iterations, uniform_price_noise, 'o-', linewidth=2, markersize=8, 
            color='#4CAF50', label='price_noise')
    ax2 = ax.twinx()
    ax2.plot(iterations, uniform_time_noise, 's-', linewidth=2, markersize=8, 
             color='#8BC34A', label='time_noise (hours)')
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price Noise', fontsize=11, fontweight='bold', color='#4CAF50')
    ax2.set_ylabel('Time Noise (hours)', fontsize=11, fontweight='bold', color='#8BC34A')
    ax.set_title('Uniform: Parameter Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_combined_summary_regression(
    mae_history: Dict[str, List[float]],
    exploit_history: Dict[str, List[float]],
    param_history: Dict[str, List[Dict]],
    threshold: float = 0.5,
    output_path: str = None,
    show: bool = True
):
    """
    Create combined summary plot for regression experiment.
    
    Args:
        mae_history: MAE% over iterations
        exploit_history: Exploitability over iterations
        param_history: Parameter values over iterations
        threshold: MAE% threshold
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    iterations = range(len(mae_history['pink']))
    
    # ========== MAE% Plot (top, spans both columns) ==========
    ax_mae = fig.add_subplot(gs[0, :])
    
    ax_mae.plot(iterations, mae_history['pink'], 'o-', label='Pink Noise', 
                linewidth=2, markersize=8, color='#E91E63')
    ax_mae.plot(iterations, mae_history['ou'], 's-', label='OU Process', 
                linewidth=2, markersize=8, color='#2196F3')
    ax_mae.plot(iterations, mae_history['uniform'], '^-', label='Uniform', 
                linewidth=2, markersize=8, color='#4CAF50')
    ax_mae.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Safety Threshold ({threshold}%)', alpha=0.7)
    
    ax_mae.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax_mae.set_ylabel('MAE% (Lower = More Exploitable)', fontsize=12, fontweight='bold')
    ax_mae.set_title('Price Prediction Error', fontsize=14, fontweight='bold')
    ax_mae.legend(loc='best', fontsize=11)
    ax_mae.grid(True, alpha=0.3)
    
    # ========== Exploitability Plot (second row, spans both columns) ==========
    ax_exploit = fig.add_subplot(gs[1, :])
    
    pink_pct = [x * 100 for x in exploit_history['pink']]
    ou_pct = [x * 100 for x in exploit_history['ou']]
    uniform_pct = [x * 100 for x in exploit_history['uniform']]
    
    ax_exploit.plot(iterations, pink_pct, 'o-', label='Pink Noise', 
                    linewidth=2, markersize=8, color='#E91E63')
    ax_exploit.plot(iterations, ou_pct, 's-', label='OU Process', 
                    linewidth=2, markersize=8, color='#2196F3')
    ax_exploit.plot(iterations, uniform_pct, '^-', label='Uniform', 
                    linewidth=2, markersize=8, color='#4CAF50')
    
    ax_exploit.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax_exploit.set_ylabel('Exploitable Trades (%)', fontsize=12, fontweight='bold')
    ax_exploit.set_title('Fraction of Trades Predictable Within 0.5%', fontsize=14, fontweight='bold')
    ax_exploit.legend(loc='best', fontsize=11)
    ax_exploit.grid(True, alpha=0.3)
    ax_exploit.set_ylim([0, 100])
    
    # ========== Parameter Evolution ==========
    
    # Pink
    ax_pink = fig.add_subplot(gs[2, 0])
    pink_scales = [p['price_scale'] for p in param_history['pink']]
    ax_pink.plot(iterations, pink_scales, 'o-', linewidth=2, markersize=8, color='#E91E63')
    ax_pink.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax_pink.set_ylabel('price_scale', fontsize=11, fontweight='bold')
    ax_pink.set_title('Pink Noise Parameters', fontsize=12, fontweight='bold')
    ax_pink.grid(True, alpha=0.3)
    
    # OU
    ax_ou = fig.add_subplot(gs[2, 1])
    ou_scales = [p['price_scale'] for p in param_history['ou']]
    ax_ou.plot(iterations, ou_scales, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax_ou.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax_ou.set_ylabel('price_scale', fontsize=11, fontweight='bold')
    ax_ou.set_title('OU Parameters', fontsize=12, fontweight='bold')
    ax_ou.grid(True, alpha=0.3)
    
    # Uniform
    ax_uniform = fig.add_subplot(gs[3, :])
    uniform_price = [p['price_noise'] for p in param_history['uniform']]
    uniform_time = [p['time_noise_minutes'] for p in param_history['uniform']]
    
    ax_uniform.plot(iterations, uniform_price, 'o-', linewidth=2, markersize=8, 
                    color='#4CAF50', label='price_noise')
    ax_uniform2 = ax_uniform.twinx()
    ax_uniform2.plot(iterations, uniform_time, 's-', linewidth=2, markersize=8, 
                     color='#8BC34A', label='time_noise (min)')
    
    ax_uniform.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax_uniform.set_ylabel('Price Noise', fontsize=11, fontweight='bold', color='#4CAF50')
    ax_uniform2.set_ylabel('Time Noise (min)', fontsize=11, fontweight='bold', color='#8BC34A')
    ax_uniform.set_title('Uniform Parameters', fontsize=12, fontweight='bold')
    ax_uniform.legend(loc='upper left')
    ax_uniform2.legend(loc='upper right')
    ax_uniform.grid(True, alpha=0.3)
    
    plt.suptitle('Adaptive Adversary Experiment - Price Prediction (Regression)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def create_results_table_regression(
    mae_history: Dict[str, List[float]],
    exploit_history: Dict[str, List[float]],
    param_history: Dict[str, List[Dict]],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Create summary table for regression results.
    
    Args:
        mae_history: MAE% scores
        exploit_history: Exploitability fractions
        param_history: Parameters
        threshold: MAE% threshold
    
    Returns:
        DataFrame with summary statistics
    """
    
    summary_data = []
    
    for policy in ['pink', 'ou', 'uniform']:
        maes = mae_history[policy]
        exploits = exploit_history[policy]
        
        row = {
            'Policy': policy.upper(),
            'Initial_MAE%': maes[0],
            'Final_MAE%': maes[-1],
            'Min_MAE%': min(maes),
            'Max_MAE%': max(maes),
            'Mean_MAE%': np.mean(maes),
            'Initial_Exploitable%': exploits[0] * 100,
            'Final_Exploitable%': exploits[-1] * 100,
            'Times_Too_Exploitable': sum(1 for mae in maes if mae < threshold),
            'Adapted': sum(1 for mae in maes if mae < threshold) > 0
        }
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    return df


if __name__ == "__main__":
    """Test visualizations with synthetic data"""
    
    print("="*80)
    print("REGRESSION VISUALIZATION TEST")
    print("="*80)
    
    # Generate synthetic data (showing improvement over iterations)
    n_iterations = 5
    
    mae_history = {
        'pink': [0.35, 0.42, 0.51, 0.68, 0.82],
        'ou': [0.28, 0.38, 0.55, 0.72, 0.89],
        'uniform': [0.42, 0.58, 0.73, 0.91, 1.05]
    }
    
    exploit_history = {
        'pink': [0.85, 0.72, 0.58, 0.35, 0.18],
        'ou': [0.92, 0.81, 0.62, 0.28, 0.12],
        'uniform': [0.78, 0.58, 0.42, 0.22, 0.09]
    }
    
    param_history = {
        'pink': [
            {'alpha': 1.0, 'price_scale': 0.04},
            {'alpha': 1.0, 'price_scale': 0.06},
            {'alpha': 1.0, 'price_scale': 0.09},
            {'alpha': 1.0, 'price_scale': 0.135},
            {'alpha': 1.0, 'price_scale': 0.2025}
        ],
        'ou': [
            {'theta': 0.5, 'sigma': 0.5, 'price_scale': 0.04},
            {'theta': 0.6, 'sigma': 0.6, 'price_scale': 0.048},
            {'theta': 0.72, 'sigma': 0.72, 'price_scale': 0.058},
            {'theta': 0.864, 'sigma': 0.864, 'price_scale': 0.069},
            {'theta': 1.037, 'sigma': 1.037, 'price_scale': 0.083}
        ],
        'uniform': [
            {'price_noise': 0.0005, 'time_noise_minutes': 120},
            {'price_noise': 0.0006, 'time_noise_minutes': 144},
            {'price_noise': 0.00072, 'time_noise_minutes': 173},
            {'price_noise': 0.00086, 'time_noise_minutes': 207},
            {'price_noise': 0.00104, 'time_noise_minutes': 249}
        ]
    }
    
    # Create output directory
    output_dir = Path("/home/claude")
    
    # Test plots
    print("\n1. Testing MAE% plot...")
    plot_mae_over_iterations(
        mae_history,
        threshold=0.5,
        output_path=output_dir / "mae_over_iterations.png",
        show=False
    )
    
    print("\n2. Testing exploitability plot...")
    plot_exploitability_over_iterations(
        exploit_history,
        output_path=output_dir / "exploitability_over_iterations.png",
        show=False
    )
    
    print("\n3. Testing parameter evolution plot...")
    plot_parameter_evolution_regression(
        param_history,
        output_path=output_dir / "parameter_evolution_regression.png",
        show=False
    )
    
    print("\n4. Testing combined summary plot...")
    plot_combined_summary_regression(
        mae_history,
        exploit_history,
        param_history,
        threshold=0.5,
        output_path=output_dir / "combined_summary_regression.png",
        show=False
    )
    
    print("\n5. Testing results table...")
    results_table = create_results_table_regression(mae_history, exploit_history, param_history)
    print("\n" + results_table.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ All visualizations tested successfully!")
    print("="*80)
