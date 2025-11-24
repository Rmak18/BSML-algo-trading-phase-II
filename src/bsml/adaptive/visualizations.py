"""
Visualizations for Adaptive Adversary Experiment

Plots:
1. AUC scores over iterations (with threshold line)
2. Parameter evolution over iterations
3. Feature importance comparisons

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


def plot_auc_over_iterations(
    auc_history: Dict[str, List[float]],
    threshold: float = 0.7,
    output_path: str = None,
    show: bool = True
):
    """
    Plot AUC scores over iterations for all three adversaries.
    
    Args:
        auc_history: Dictionary with 'pink', 'ou', 'uniform' keys
        threshold: AUC threshold line
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(len(auc_history['pink']))
    
    # Plot each adversary
    ax.plot(iterations, auc_history['pink'], 'o-', label='Pink Noise', 
            linewidth=2, markersize=8, color='#E91E63')
    ax.plot(iterations, auc_history['ou'], 's-', label='OU Process', 
            linewidth=2, markersize=8, color='#2196F3')
    ax.plot(iterations, auc_history['uniform'], '^-', label='Uniform', 
            linewidth=2, markersize=8, color='#4CAF50')
    
    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold ({threshold})', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Adversary Detection Performance Over Iterations', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Add annotations for threshold violations
    for i, (pink, ou, uniform) in enumerate(zip(
        auc_history['pink'], 
        auc_history['ou'], 
        auc_history['uniform']
    )):
        if pink > threshold:
            ax.annotate('⚠', xy=(i, pink), xytext=(0, 10), 
                       textcoords='offset points', fontsize=14, color='#E91E63')
        if ou > threshold:
            ax.annotate('⚠', xy=(i, ou), xytext=(0, 10), 
                       textcoords='offset points', fontsize=14, color='#2196F3')
        if uniform > threshold:
            ax.annotate('⚠', xy=(i, uniform), xytext=(0, 10), 
                       textcoords='offset points', fontsize=14, color='#4CAF50')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_parameter_evolution(
    param_history: Dict[str, List[Dict]],
    output_path: str = None,
    show: bool = True
):
    """
    Plot how policy parameters evolve over iterations.
    
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
    ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
    ax.set_title('Pink Noise: price_scale Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== OU Process ==========
    ax = axes[1]
    ou_thetas = [p['theta'] for p in param_history['ou']]
    ou_sigmas = [p['sigma'] for p in param_history['ou']]
    ou_scales = [p['price_scale'] for p in param_history['ou']]
    
    ax.plot(iterations, ou_thetas, 'o-', linewidth=2, markersize=8, 
            color='#2196F3', label='theta')
    ax.plot(iterations, ou_sigmas, 's-', linewidth=2, markersize=8, 
            color='#03A9F4', label='sigma')
    ax.plot(iterations, ou_scales, '^-', linewidth=2, markersize=8, 
            color='#00BCD4', label='price_scale')
    ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
    ax.set_title('OU Process: Parameter Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== Uniform ==========
    ax = axes[2]
    uniform_price_noise = [p['price_noise'] for p in param_history['uniform']]
    uniform_time_noise = [p['time_noise_minutes'] / 60.0 for p in param_history['uniform']]  # Convert to hours
    
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


def plot_combined_summary(
    auc_history: Dict[str, List[float]],
    param_history: Dict[str, List[Dict]],
    threshold: float = 0.7,
    output_path: str = None,
    show: bool = True
):
    """
    Create a combined summary plot with AUC and parameters.
    
    Args:
        auc_history: AUC scores over iterations
        param_history: Parameter values over iterations
        threshold: AUC threshold
        output_path: Path to save figure
        show: Whether to display plot
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========== AUC Plot (spans both columns) ==========
    ax_auc = fig.add_subplot(gs[0, :])
    
    iterations = range(len(auc_history['pink']))
    
    ax_auc.plot(iterations, auc_history['pink'], 'o-', label='Pink Noise', 
                linewidth=2, markersize=8, color='#E91E63')
    ax_auc.plot(iterations, auc_history['ou'], 's-', label='OU Process', 
                linewidth=2, markersize=8, color='#2196F3')
    ax_auc.plot(iterations, auc_history['uniform'], '^-', label='Uniform', 
                linewidth=2, markersize=8, color='#4CAF50')
    ax_auc.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold})', alpha=0.7)
    
    ax_auc.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax_auc.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax_auc.set_title('Adversary Detection Performance', fontsize=14, fontweight='bold')
    ax_auc.legend(loc='best', fontsize=11)
    ax_auc.grid(True, alpha=0.3)
    ax_auc.set_ylim([0.4, 1.0])
    
    # ========== Pink Parameters ==========
    ax_pink = fig.add_subplot(gs[1, 0])
    pink_scales = [p['price_scale'] for p in param_history['pink']]
    ax_pink.plot(iterations, pink_scales, 'o-', linewidth=2, markersize=8, 
                 color='#E91E63')
    ax_pink.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax_pink.set_ylabel('price_scale', fontsize=11, fontweight='bold')
    ax_pink.set_title('Pink Noise Parameters', fontsize=12, fontweight='bold')
    ax_pink.grid(True, alpha=0.3)
    
    # ========== OU Parameters ==========
    ax_ou = fig.add_subplot(gs[1, 1])
    ou_scales = [p['price_scale'] for p in param_history['ou']]
    ax_ou.plot(iterations, ou_scales, 'o-', linewidth=2, markersize=8, 
               color='#2196F3')
    ax_ou.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax_ou.set_ylabel('price_scale', fontsize=11, fontweight='bold')
    ax_ou.set_title('OU Parameters', fontsize=12, fontweight='bold')
    ax_ou.grid(True, alpha=0.3)
    
    # ========== Uniform Parameters ==========
    ax_uniform = fig.add_subplot(gs[2, :])
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
    
    plt.suptitle('Adaptive Adversary Experiment - Complete Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def create_results_table(
    auc_history: Dict[str, List[float]],
    param_history: Dict[str, List[Dict]],
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Create a summary table of results.
    
    Args:
        auc_history: AUC scores
        param_history: Parameters
        threshold: AUC threshold
    
    Returns:
        DataFrame with summary statistics
    """
    
    n_iterations = len(auc_history['pink'])
    
    summary_data = []
    
    for policy in ['pink', 'ou', 'uniform']:
        aucs = auc_history[policy]
        
        row = {
            'Policy': policy.upper(),
            'Initial_AUC': aucs[0],
            'Final_AUC': aucs[-1],
            'Max_AUC': max(aucs),
            'Min_AUC': min(aucs),
            'Mean_AUC': np.mean(aucs),
            'Times_Above_Threshold': sum(1 for auc in aucs if auc > threshold),
            'Adapted': sum(1 for auc in aucs if auc > threshold) > 0
        }
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    return df


if __name__ == "__main__":
    """Test visualizations with synthetic data"""
    
    print("="*80)
    print("VISUALIZATION TEST")
    print("="*80)
    
    # Generate synthetic data
    n_iterations = 5
    
    auc_history = {
        'pink': [0.85, 0.78, 0.72, 0.68, 0.65],
        'ou': [0.92, 0.88, 0.82, 0.75, 0.69],
        'uniform': [0.95, 0.93, 0.89, 0.84, 0.78]
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
            {'theta': 0.15, 'sigma': 0.02, 'price_scale': 1.0},
            {'theta': 0.18, 'sigma': 0.026, 'price_scale': 1.5},
            {'theta': 0.216, 'sigma': 0.034, 'price_scale': 2.25},
            {'theta': 0.259, 'sigma': 0.044, 'price_scale': 3.38},
            {'theta': 0.311, 'sigma': 0.057, 'price_scale': 5.06}
        ],
        'uniform': [
            {'price_noise': 0.03, 'time_noise_minutes': 30},
            {'price_noise': 0.039, 'time_noise_minutes': 39},
            {'price_noise': 0.051, 'time_noise_minutes': 51},
            {'price_noise': 0.066, 'time_noise_minutes': 66},
            {'price_noise': 0.086, 'time_noise_minutes': 86}
        ]
    }
    
    # Create output directory
    output_dir = Path("/home/claude")
    
    # Test AUC plot
    print("\n1. Testing AUC plot...")
    plot_auc_over_iterations(
        auc_history, 
        threshold=0.7,
        output_path=output_dir / "auc_over_iterations.png",
        show=False
    )
    
    # Test parameter evolution plot
    print("\n2. Testing parameter evolution plot...")
    plot_parameter_evolution(
        param_history,
        output_path=output_dir / "parameter_evolution.png",
        show=False
    )
    
    # Test combined summary
    print("\n3. Testing combined summary plot...")
    plot_combined_summary(
        auc_history,
        param_history,
        threshold=0.7,
        output_path=output_dir / "combined_summary.png",
        show=False
    )
    
    # Test results table
    print("\n4. Testing results table...")
    results_table = create_results_table(auc_history, param_history)
    print("\n" + results_table.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ All visualizations tested successfully!")
    print("="*80)
