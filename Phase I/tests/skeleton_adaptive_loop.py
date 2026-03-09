"""
Adaptive Adversary Framework v0.1 - Skeleton Implementation

Project: BSML - Randomized Execution Research
Owner: P7 (Adaptive Adversary Framework)
Date: November 14, 2025
Status: Week 2 Deliverable

This is a skeleton implementation that validates core adaptive logic on toy/mock data.
Full integration with P3 backtests and P6 real adversary happens in v1.0 (Week 3).
"""

from typing import Dict, List, Optional, Callable, Tuple, Any
import copy
import numpy as np
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

AUC_HIGH_THRESHOLD = 0.75      # Too predictable
AUC_LOW_THRESHOLD = 0.55       # Too random
AUC_TARGET_MIN = 0.60          # Lower bound of optimal range
AUC_TARGET_MAX = 0.70          # Upper bound of optimal range
AUC_TARGET_MIDPOINT = 0.65     # Midpoint for nudging

ADJUSTMENT_FACTOR_INCREASE = 1.20  # Multiply by 1.2 to increase stochasticity
ADJUSTMENT_FACTOR_DECREASE = 0.80  # Multiply by 0.8 to decrease stochasticity

DEFAULT_CONVERGENCE_PATIENCE = 5   # Iterations to stay in target for convergence
DEFAULT_MAX_ITERATIONS = 20        # Maximum training iterations


# =============================================================================
# CORE ADJUSTMENT LOGIC
# =============================================================================

def decide_adjustment(auc_score: float) -> Tuple[str, Optional[float]]:
    """
    Decide what adjustment action to take based on AUC score.
    
    Decision rules:
    - AUC > 0.75: INCREASE stochasticity (too predictable)
    - AUC < 0.55: DECREASE stochasticity (too random)
    - 0.60 ≤ AUC ≤ 0.70: HOLD (in optimal range)
    - 0.65 < AUC < 0.75: NUDGE_UP (gently increase)
    - 0.55 < AUC < 0.65: NUDGE_DOWN (gently decrease)
    
    Args:
        auc_score: Adversary AUC on validation fold [0.5, 1.0]
    
    Returns:
        action: One of ['INCREASE', 'DECREASE', 'HOLD', 'NUDGE_UP', 'NUDGE_DOWN']
        multiplier: 1.20, 0.80, or None
    
    Examples:
        >>> decide_adjustment(0.82)
        ('INCREASE', 1.20)
        
        >>> decide_adjustment(0.52)
        ('DECREASE', 0.80)
        
        >>> decide_adjustment(0.65)
        ('HOLD', None)
    """
    # Primary rules
    if auc_score > AUC_HIGH_THRESHOLD:
        return 'INCREASE', ADJUSTMENT_FACTOR_INCREASE
    
    elif auc_score < AUC_LOW_THRESHOLD:
        return 'DECREASE', ADJUSTMENT_FACTOR_DECREASE
    
    elif AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
        return 'HOLD', None
    
    # Secondary rules (between thresholds)
    else:
        if auc_score > AUC_TARGET_MIDPOINT:
            return 'NUDGE_UP', ADJUSTMENT_FACTOR_INCREASE
        else:
            return 'NUDGE_DOWN', ADJUSTMENT_FACTOR_DECREASE


def adaptive_step(
    auc_score: float,
    policy: Any,
    iteration: int,
    adjustment_history: List[Dict]
) -> Dict[str, Any]:
    """
    Execute one adaptive adjustment step.
    
    Workflow:
    1. Decide action based on AUC
    2. Store old parameters
    3. Apply adjustment via P4 API (if not HOLD)
    4. Store new parameters
    5. Check for oscillation
    6. Create adjustment record
    
    Args:
        auc_score: Current adversary AUC on validation fold
        policy: RandomizationPolicy instance from P4
        iteration: Current training iteration number
        adjustment_history: List of prior adjustment records
    
    Returns:
        adjustment_record: Dict with keys:
            - iteration: int
            - auc: float
            - action: str
            - multiplier: float or None
            - rationale: str
            - params_before: dict
            - params_after: dict
            - oscillation_warning: bool
    
    Examples:
        >>> from bsml.randomization import UniformPolicy
        >>> policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
        >>> record = adaptive_step(0.82, policy, 0, [])
        >>> record['action']
        'INCREASE'
        >>> policy.params['timing_range_hours']
        2.4
    """
    # Step 1: Decide action
    action, multiplier = decide_adjustment(auc_score)
    
    # Step 2: Store old parameters
    params_before = copy.deepcopy(policy.params)
    
    # Step 3: Apply adjustment (if not HOLD)
    if action != 'HOLD':
        direction = 'increase' if action in ['INCREASE', 'NUDGE_UP'] else 'decrease'
        policy.adjust_stochasticity(auc_score=auc_score, direction=direction)
    
    # Step 4: Store new parameters
    params_after = copy.deepcopy(policy.params)
    
    # Step 5: Check for oscillation
    oscillation_detected = detect_oscillation(adjustment_history, window=3)
    
    # Step 6: Create record
    record = {
        'iteration': iteration,
        'auc': auc_score,
        'action': action,
        'multiplier': multiplier,
        'rationale': _generate_rationale(auc_score, action),
        'params_before': params_before,
        'params_after': params_after,
        'oscillation_warning': oscillation_detected
    }
    
    return record


def _generate_rationale(auc_score: float, action: str) -> str:
    """Generate human-readable rationale for adjustment action."""
    if action == 'INCREASE':
        return f"AUC {auc_score:.3f} > {AUC_HIGH_THRESHOLD} (too predictable)"
    elif action == 'DECREASE':
        return f"AUC {auc_score:.3f} < {AUC_LOW_THRESHOLD} (too random)"
    elif action == 'HOLD':
        return f"AUC {auc_score:.3f} in target range [{AUC_TARGET_MIN}, {AUC_TARGET_MAX}]"
    elif action == 'NUDGE_UP':
        return f"AUC {auc_score:.3f} above target midpoint {AUC_TARGET_MIDPOINT:.2f}"
    elif action == 'NUDGE_DOWN':
        return f"AUC {auc_score:.3f} below target midpoint {AUC_TARGET_MIDPOINT:.2f}"
    else:
        return f"AUC {auc_score:.3f} → {action}"


# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================

def check_convergence(
    auc_history: List[float],
    patience: int = DEFAULT_CONVERGENCE_PATIENCE
) -> Tuple[bool, Optional[int]]:
    """
    Check if AUC has converged (stable in target range).
    
    Convergence criterion:
    AUC stays within [AUC_TARGET_MIN, AUC_TARGET_MAX] for N consecutive iterations.
    
    Args:
        auc_history: List of AUC scores from all iterations
        patience: Number of consecutive iterations in target required
    
    Returns:
        converged: bool, True if converged
        convergence_iteration: int or None, iteration where convergence started
    
    Examples:
        >>> auc_history = [0.75, 0.72, 0.65, 0.68, 0.62, 0.67, 0.63]
        >>> check_convergence(auc_history, patience=5)
        (True, 2)
    """
    if len(auc_history) < patience:
        return False, None
    
    # Check last N iterations
    last_n = auc_history[-patience:]
    all_in_range = all(
        AUC_TARGET_MIN <= auc <= AUC_TARGET_MAX
        for auc in last_n
    )
    
    if all_in_range:
        convergence_iteration = len(auc_history) - patience
        return True, convergence_iteration
    
    return False, None


# =============================================================================
# OSCILLATION DETECTION
# =============================================================================

def detect_oscillation(
    adjustment_history: List[Dict],
    window: int = 3
) -> bool:
    """
    Detect if adjustments are oscillating.
    
    Oscillation patterns:
    - INCREASE → DECREASE → INCREASE
    - DECREASE → INCREASE → DECREASE
    - NUDGE_UP → NUDGE_DOWN → NUDGE_UP
    - NUDGE_DOWN → NUDGE_UP → NUDGE_DOWN
    
    Args:
        adjustment_history: List of adjustment records
        window: Number of recent adjustments to check
    
    Returns:
        is_oscillating: bool
    
    Examples:
        >>> history = [
        ...     {'action': 'INCREASE'},
        ...     {'action': 'DECREASE'},
        ...     {'action': 'INCREASE'}
        ... ]
        >>> detect_oscillation(history, window=3)
        True
    """
    if len(adjustment_history) < window:
        return False
    
    # Get last N actions
    last_actions = [rec['action'] for rec in adjustment_history[-window:]]
    
    # Define oscillation patterns
    oscillation_patterns = [
        ['INCREASE', 'DECREASE', 'INCREASE'],
        ['DECREASE', 'INCREASE', 'DECREASE'],
        ['NUDGE_UP', 'NUDGE_DOWN', 'NUDGE_UP'],
        ['NUDGE_DOWN', 'NUDGE_UP', 'NUDGE_DOWN']
    ]
    
    # Check if last N actions match any pattern
    for pattern in oscillation_patterns:
        if last_actions == pattern:
            return True
    
    return False


# =============================================================================
# TRAINING LOOP
# =============================================================================

def adaptive_adversary_training_loop(
    policy_initial: Any,
    n_iterations: int = DEFAULT_MAX_ITERATIONS,
    early_stop_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
    mock_auc_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main adaptive training loop (v0.1 skeleton with mock adversary).
    
    This is a skeleton implementation that uses a mock adversary function
    instead of real P6 adversary training. Full integration in v1.0.
    
    Workflow:
    1. Initialize policy and logging
    2. For each iteration:
        a. Get AUC from mock adversary (or real in v1.0)
        b. Execute adaptive adjustment
        c. Check convergence
        d. Check for early stopping
    3. Return results
    
    Args:
        policy_initial: Initial RandomizationPolicy from P4
        n_iterations: Maximum training iterations
        early_stop_patience: Stop if in target range for N iterations
        mock_auc_fn: Optional mock function(policy, iteration) -> auc
                     If None, uses default mock adversary
        verbose: If True, print progress
    
    Returns:
        results: Dict with keys:
            - policy_name: str
            - auc_history: List[float]
            - adjustment_log: List[Dict]
            - final_policy_params: dict
            - converged_iteration: int or None
            - n_iterations_run: int
    
    Examples:
        >>> from bsml.randomization import UniformPolicy
        >>> policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
        >>> results = adaptive_adversary_training_loop(policy, n_iterations=10)
        >>> results['converged_iteration'] is not None
        True
    """
    # Initialize
    policy = copy.deepcopy(policy_initial)
    adjustment_log = []
    auc_history = []
    converged_iteration = None
    
    # Use default mock adversary if none provided
    if mock_auc_fn is None:
        mock_auc_fn = _default_mock_adversary
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ADAPTIVE ADVERSARY TRAINING LOOP v0.1")
        print(f"Policy: {policy.__class__.__name__}")
        print(f"Initial params: {policy.params}")
        print(f"Max iterations: {n_iterations}")
        print(f"{'='*60}\n")
    
    # Training loop
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        
        # Step 1: Get AUC from mock adversary
        auc_score = mock_auc_fn(policy, iteration)
        auc_history.append(auc_score)
        
        if verbose:
            print(f"  Mock AUC: {auc_score:.3f}")
        
        # Step 2: Execute adaptive adjustment
        adjustment = adaptive_step(auc_score, policy, iteration, adjustment_log)
        adjustment_log.append(adjustment)
        
        if verbose:
            print(f"  Action: {adjustment['action']}")
            print(f"  Rationale: {adjustment['rationale']}")
            if adjustment['action'] != 'HOLD':
                print(f"  Updated params: {adjustment['params_after']}")
            if adjustment['oscillation_warning']:
                print(f"  ⚠️  OSCILLATION DETECTED")
        
        # Step 3: Check convergence
        converged, conv_iter = check_convergence(auc_history, early_stop_patience)
        
        if converged:
            converged_iteration = conv_iter
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ CONVERGED at iteration {iteration + 1}")
                print(f"AUC stable in target range for {early_stop_patience} iterations")
                print(f"{'='*60}\n")
            break
    
    # Compile results
    results = {
        'policy_name': policy.__class__.__name__,
        'auc_history': auc_history,
        'adjustment_log': adjustment_log,
        'final_policy_params': copy.deepcopy(policy.params),
        'converged_iteration': converged_iteration,
        'n_iterations_run': iteration + 1
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Iterations run: {results['n_iterations_run']}")
        print(f"Final AUC: {auc_history[-1]:.3f}")
        print(f"Final params: {results['final_policy_params']}")
        if converged_iteration is not None:
            print(f"Converged: Yes (iteration {converged_iteration})")
        else:
            print(f"Converged: No")
        print(f"{'='*60}\n")
    
    return results


def _default_mock_adversary(policy: Any, iteration: int) -> float:
    """
    Default mock adversary for testing.
    
    Simulates adversary AUC based on policy stochasticity:
    - More stochasticity → lower AUC (less predictable)
    - Less stochasticity → higher AUC (more predictable)
    
    Args:
        policy: RandomizationPolicy instance
        iteration: Current iteration number
    
    Returns:
        auc: Simulated AUC score [0.5, 1.0]
    """
    # Extract timing range (main stochasticity parameter)
    if hasattr(policy.params, 'get'):
        timing_range = policy.params.get('timing_range_hours', 2.0)
    else:
        timing_range = 2.0
    
    # Base AUC inversely related to timing_range
    # Initial AUC = 0.85, decreases by 0.05 per hour of range
    base_auc = 0.85 - (timing_range - 2.0) * 0.05
    
    # Add small random noise for realism
    np.random.seed(42 + iteration)  # Deterministic for reproducibility
    noise = np.random.normal(0, 0.02)
    
    auc = base_auc + noise
    
    # Clip to valid range
    return float(max(0.5, min(1.0, auc)))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save adaptive adversary results to CSV files.
    
    Creates two files:
    1. {output_path}_auc_history.csv - AUC trajectory
    2. {output_path}_adjustment_log.csv - Adjustment records
    
    Args:
        results: Results dict from adaptive_adversary_training_loop
        output_path: Base path for output files (without extension)
    
    Examples:
        >>> results = adaptive_adversary_training_loop(policy)
        >>> save_results(results, 'results/adaptive_uniform_seed42')
    """
    import pandas as pd
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save AUC history
    auc_df = pd.DataFrame({
        'iteration': range(len(results['auc_history'])),
        'auc': results['auc_history']
    })
    auc_path = f"{output_path}_auc_history.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"Saved AUC history to: {auc_path}")
    
    # Save adjustment log
    log_records = []
    for rec in results['adjustment_log']:
        flat_rec = {
            'iteration': rec['iteration'],
            'auc': rec['auc'],
            'action': rec['action'],
            'multiplier': rec.get('multiplier'),
            'rationale': rec['rationale'],
            'oscillation_warning': rec.get('oscillation_warning', False)
        }
        
        # Flatten params
        for key, val in rec['params_after'].items():
            flat_rec[f'param_{key}'] = val
        
        log_records.append(flat_rec)
    
    log_df = pd.DataFrame(log_records)
    log_path = f"{output_path}_adjustment_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Saved adjustment log to: {log_path}")


def plot_auc_trajectory(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot AUC trajectory over iterations.
    
    Args:
        results: Results dict from adaptive_adversary_training_loop
        save_path: Optional path to save figure
    
    Examples:
        >>> results = adaptive_adversary_training_loop(policy)
        >>> plot_auc_trajectory(results, 'figures/auc_trajectory.png')
    """
    import matplotlib.pyplot as plt
    
    auc_history = results['auc_history']
    iterations = range(len(auc_history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, auc_history, 'o-', linewidth=2, markersize=6, label='AUC')
    
    # Add threshold lines
    plt.axhline(y=AUC_HIGH_THRESHOLD, color='r', linestyle='--', 
                label=f'High Threshold ({AUC_HIGH_THRESHOLD})')
    plt.axhline(y=AUC_LOW_THRESHOLD, color='b', linestyle='--', 
                label=f'Low Threshold ({AUC_LOW_THRESHOLD})')
    
    # Shade target range
    plt.axhspan(AUC_TARGET_MIN, AUC_TARGET_MAX, alpha=0.2, color='green', 
                label=f'Target Range [{AUC_TARGET_MIN}, {AUC_TARGET_MAX}]')
    
    # Mark convergence if occurred
    if results['converged_iteration'] is not None:
        plt.axvline(x=results['converged_iteration'], color='g', linestyle=':', 
                    linewidth=2, label=f"Converged (iter {results['converged_iteration']})")
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Adversary AUC', fontsize=12)
    plt.title(f"AUC Trajectory - {results['policy_name']}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved AUC trajectory plot to: {save_path}")
    else:
        plt.show()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    """
    Test adaptive_adversary_v0.1 with mock policy.
    
    This demonstrates the skeleton working with toy data.
    Real integration with P4 and P6 happens in v1.0.
    """
    print("Testing adaptive_adversary_v0.1 skeleton...")
    
    # Create a simple mock policy for testing
    class MockPolicy:
        """Mock policy for testing without P4 dependency."""
        
        def __init__(self, seed, params):
            self.seed = seed
            self.params = params.copy()
            self.__class__.__name__ = 'MockUniform'
        
        def adjust_stochasticity(self, auc_score, direction):
            """Mock adjustment that multiplies params."""
            multiplier = 1.2 if direction == 'increase' else 0.8
            
            for key in self.params:
                self.params[key] *= multiplier
                
                # Enforce bounds
                if key == 'timing_range_hours':
                    self.params[key] = max(0.5, min(6.0, self.params[key]))
                elif key == 'threshold_pct':
                    self.params[key] = max(0.05, min(0.25, self.params[key]))
    
    # Initialize mock policy
    policy = MockPolicy(seed=42, params={
        'timing_range_hours': 2.0,
        'threshold_pct': 0.10
    })
    
    # Run adaptive loop
    results = adaptive_adversary_training_loop(
        policy_initial=policy,
        n_iterations=20,
        early_stop_patience=5,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Initial params: timing_range=2.0, threshold_pct=0.10")
    print(f"Final params: {results['final_policy_params']}")
    print(f"AUC trajectory: {results['auc_history'][0]:.3f} → {results['auc_history'][-1]:.3f}")
    print(f"Converged: {results['converged_iteration'] is not None}")
    print(f"Total iterations: {results['n_iterations_run']}")
    print("="*60)
    
    # Test individual functions
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL FUNCTIONS")
    print("="*60)
    
    # Test decide_adjustment
    print("\n1. Testing decide_adjustment:")
    test_aucs = [0.82, 0.52, 0.65, 0.72, 0.57]
    for auc in test_aucs:
        action, mult = decide_adjustment(auc)
        print(f"   AUC={auc:.2f} → {action} (×{mult})")
    
    # Test convergence detection
    print("\n2. Testing check_convergence:")
    test_history = [0.75, 0.72, 0.65, 0.68, 0.62, 0.67, 0.63]
    converged, iter_conv = check_convergence(test_history, patience=5)
    print(f"   History: {[f'{x:.2f}' for x in test_history]}")
    print(f"   Converged: {converged}, at iteration: {iter_conv}")
    
    # Test oscillation detection
    print("\n3. Testing detect_oscillation:")
    test_adjustments = [
        {'action': 'INCREASE'},
        {'action': 'DECREASE'},
        {'action': 'INCREASE'}
    ]
    oscillating = detect_oscillation(test_adjustments)
    print(f"   Actions: {[r['action'] for r in test_adjustments]}")
    print(f"   Oscillating: {oscillating}")
    
    print("\n✓ All tests complete!")
