"""
Ornstein-Uhlenbeck (OU) process randomization policy.

Adds autocorrelated noise to timing and thresholds. OU processes exhibit
mean reversion and temporal correlation, making perturbations more realistic
than IID uniform noise.

Mathematical Formulation:
    dx(t) = θ(μ - x(t))dt + σdW(t)
    
    Discrete-time exact solution:
    x(t+Δt) = μ + (x(t) - μ)exp(-θΔt) + σ√((1-exp(-2θΔt))/(2θ)) * ε
    where ε ~ N(0,1)

Critical Design Decisions:
1. Use exact analytical solution (not Euler-Maruyama) for stability
2. Track actual elapsed time between perturbations
3. Monitor state drift and reset if needed
4. Independent OU processes for timing and threshold

Owner: P4
Week: 3
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import warnings

from .base_policy import RandomizationPolicy
from .utils import clamp_to_market_hours, check_market_hours


class OUPolicy(RandomizationPolicy):
    """
    Ornstein-Uhlenbeck process randomization policy.
    
    Provides autocorrelated perturbations with mean reversion. Unlike Uniform
    policy, perturbations are temporally correlated: if the last perturbation
    was positive, the next one is more likely to be positive (but decaying
    toward zero over time).
    
    Parameters:
        theta_timing (float): Mean reversion rate for timing (higher = faster reversion)
                              Half-life = ln(2)/θ
        theta_threshold (float): Mean reversion rate for threshold
        sigma_timing (float): Volatility for timing (hours)
        sigma_threshold (float): Volatility for threshold (fractional)
        mu (float): Long-term mean (typically 0)
        state_reset_threshold (float): Reset state if |x| > threshold * σ/√(2θ)
        track_elapsed_time (bool): Use actual elapsed time for Δt (True recommended)
        dt_fixed (float): Fixed time step in hours (only if track_elapsed_time=False)
    
    Statistical Properties:
        - Mean: E[x(∞)] = μ
        - Variance: Var[x(∞)] = σ²/(2θ)
        - Autocorrelation: ρ(τ) = exp(-θτ)
        - Half-life: t_1/2 = ln(2)/θ
    
    Example:
        >>> policy = OUPolicy(
        ...     seed=42,
        ...     params={
        ...         'theta_timing': 0.5,      # Half-life ≈ 1.4 hours
        ...         'sigma_timing': 1.0,
        ...         'theta_threshold': 0.5,
        ...         'sigma_threshold': 0.05,
        ...     }
        ... )
        >>> 
        >>> # Perturb timing (autocorrelated)
        >>> ts1 = datetime(2025, 7, 15, 10, 0)
        >>> perturbed1 = policy.perturb_timing(ts1)
        >>> 
        >>> ts2 = datetime(2025, 7, 15, 11, 0)  # 1 hour later
        >>> perturbed2 = policy.perturb_timing(ts2)
        >>> # perturbed2 shift will be correlated with perturbed1 shift
    """
    
    # Parameter bounds
    PARAMETER_BOUNDS = {
        'theta_timing': {'min': 0.01, 'max': 10.0, 'default': 0.5},
        'theta_threshold': {'min': 0.01, 'max': 10.0, 'default': 0.5},
        'sigma_timing': {'min': 0.1, 'max': 5.0, 'default': 1.0},
        'sigma_threshold': {'min': 0.01, 'max': 0.2, 'default': 0.05},
        'mu': {'min': -2.0, 'max': 2.0, 'default': 0.0},
        'state_reset_threshold': {'min': 5.0, 'max': 20.0, 'default': 10.0},
        'dt_fixed': {'min': 0.1, 'max': 24.0, 'default': 1.0},
    }
    
    def __init__(self, seed: int, params: Dict[str, Any]):
        """
        Initialize OU policy.
        
        Args:
            seed: Master seed for reproducibility
            params: Must contain 'theta_timing', 'sigma_timing', etc.
        
        Raises:
            ValueError: If required parameters are missing
        """
        # Validate required parameters
        required = ['theta_timing', 'sigma_timing', 'theta_threshold', 'sigma_threshold']
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"OUPolicy requires parameters: {', '.join(missing)}. "
                f"Got: {list(params.keys())}"
            )
        
        # Set defaults
        defaults = {
            'mu': 0.0,
            'state_reset_threshold': 10.0,
            'track_elapsed_time': True,
            'dt_fixed': 1.0,
            'respect_market_hours': True,
        }
        for key, default in defaults.items():
            if key not in params:
                params[key] = default
        
        # Initialize base class
        super().__init__(seed, params)
        
        # OU state variables
        self._ou_timing_state = 0.0
        self._ou_threshold_state = 0.0
        self._last_timestamp: Optional[datetime] = None
        
        # State history for diagnostics
        self._ou_timing_history: List[float] = []
        self._ou_threshold_history: List[float] = []
        self._reset_events: List[Dict[str, Any]] = []
        
        # Validate parameters
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate OU parameters."""
        # Check positivity constraints
        assert self.params['theta_timing'] > 0, "theta_timing must be positive"
        assert self.params['theta_threshold'] > 0, "theta_threshold must be positive"
        assert self.params['sigma_timing'] > 0, "sigma_timing must be positive"
        assert self.params['sigma_threshold'] > 0, "sigma_threshold must be positive"
        assert self.params['state_reset_threshold'] > 0, "state_reset_threshold must be positive"
        assert self.params['dt_fixed'] > 0, "dt_fixed must be positive"
    
    def _ou_update(
        self, 
        x_current: float, 
        theta: float, 
        sigma: float, 
        mu: float, 
        dt: float
    ) -> float:
        """
        Exact OU process update using analytical solution.
        
        This is numerically stable for all values of θ and dt.
        
        Formula:
            x_next = μ + (x - μ)exp(-θΔt) + σ√((1-exp(-2θΔt))/(2θ)) * ε
        
        Args:
            x_current: Current state value
            theta: Mean reversion rate
            sigma: Volatility
            mu: Long-term mean
            dt: Time step
        
        Returns:
            Updated state value
        """
        # Mean reversion term
        mean_term = mu + (x_current - mu) * np.exp(-theta * dt)
        
        # Variance term (exact for OU process)
        variance = (sigma**2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        std = np.sqrt(max(variance, 0))  # Ensure non-negative
        
        # Random shock
        noise = self.rng.randn() * std
        
        return mean_term + noise
    
    def _check_and_reset_state(
        self, 
        state: float, 
        theta: float, 
        sigma: float, 
        state_name: str
    ) -> float:
        """
        Monitor OU state and reset if it drifts too far.
        
        This prevents state explosion due to random walk component.
        
        Threshold: |x| > state_reset_threshold * σ/√(2θ)
        
        Args:
            state: Current state value
            theta: Mean reversion rate
            sigma: Volatility
            state_name: 'timing' or 'threshold' (for logging)
        
        Returns:
            State value (reset to 0 if threshold exceeded, otherwise unchanged)
        """
        stationary_std = sigma / np.sqrt(2 * theta)
        threshold = self.params['state_reset_threshold'] * stationary_std
        
        if abs(state) > threshold:
            # Log reset event
            self._reset_events.append({
                'state_name': state_name,
                'old_value': state,
                'threshold': threshold,
                'timestamp': self._last_timestamp,
            })
            
            warnings.warn(
                f"OU state '{state_name}' exceeded threshold "
                f"({abs(state):.3f} > {threshold:.3f}). Resetting to 0."
            )
            return 0.0
        
        return state
    
    def perturb_timing(
        self, 
        timestamp: datetime, 
        signal_strength: float = 1.0
    ) -> datetime:
        """
        Apply OU perturbation to timing.
        
        Critical Logic:
        1. Calculate Δt since last call
        2. Update OU state using exact solution
        3. Check for state drift and reset if needed
        4. Convert OU state to time shift
        5. Clamp to market hours
        
        Args:
            timestamp: Original signal timestamp
            signal_strength: Optional scaling factor
        
        Returns:
            Perturbed timestamp
        """
        # Determine time step
        if self.params['track_elapsed_time'] and self._last_timestamp is not None:
            dt = (timestamp - self._last_timestamp).total_seconds() / 3600  # Hours
        else:
            dt = self.params['dt_fixed']
        
        # Update OU state
        self._ou_timing_state = self._ou_update(
            self._ou_timing_state,
            self.params['theta_timing'],
            self.params['sigma_timing'],
            self.params['mu'],
            dt
        )
        
        # Check for drift and reset if needed
        self._ou_timing_state = self._check_and_reset_state(
            self._ou_timing_state,
            self.params['theta_timing'],
            self.params['sigma_timing'],
            'timing'
        )
        
        # Convert state to time shift (in hours)
        shift_hours = self._ou_timing_state * signal_strength
        perturbed_time = timestamp + timedelta(hours=shift_hours)
        
        # Clamp to market hours if requested
        if self.params.get('respect_market_hours', True):
            from datetime import time
            perturbed_time = clamp_to_market_hours(
                perturbed_time,
                time(9, 30),
                time(16, 0)
            )
        
        # Update tracking
        self._last_timestamp = timestamp
        self._ou_timing_history.append(self._ou_timing_state)
        
        return perturbed_time
    
    def perturb_threshold(
        self, 
        base_threshold: float, 
        current_price: Optional[float] = None
    ) -> float:
        """
        Apply OU perturbation to threshold.
        
        Uses independent OU process for threshold (not correlated with timing).
        
        Args:
            base_threshold: Original threshold value
            current_price: Not used (kept for interface compatibility)
        
        Returns:
            Perturbed threshold (always positive)
        """
        # Use same dt as timing
        if self.params['track_elapsed_time'] and self._last_timestamp is not None and len(self._ou_timing_history) > 0:
            # We've already updated timing, use its dt
            dt = self.params['dt_fixed']  # Approximate
        else:
            dt = self.params['dt_fixed']
        
        # Update OU state
        self._ou_threshold_state = self._ou_update(
            self._ou_threshold_state,
            self.params['theta_threshold'],
            self.params['sigma_threshold'],
            self.params['mu'],
            dt
        )
        
        # Check for drift
        self._ou_threshold_state = self._check_and_reset_state(
            self._ou_threshold_state,
            self.params['theta_threshold'],
            self.params['sigma_threshold'],
            'threshold'
        )
        
        # Apply perturbation as multiplicative factor
        perturbed = base_threshold * (1.0 + self._ou_threshold_state)
        
        # Ensure positive
        perturbed = max(perturbed, 0.01)
        
        # Update history
        self._ou_threshold_history.append(self._ou_threshold_state)
        
        return perturbed
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return OU policy diagnostics.
        
        Includes:
        - Standard diagnostics (policy, seed, params)
        - OU state histories
        - Autocorrelation functions
        - Half-life calculations
        - State reset events
        
        Returns:
            Dictionary with comprehensive diagnostics
        """
        diagnostics = {
            'policy': 'OU',
            'seed': self.seed,
            'params': self.params.copy(),
            'n_perturbations': len(self._ou_timing_history),
        }
        
        # OU-specific statistics
        if self._ou_timing_history:
            timing_array = np.array(self._ou_timing_history)
            
            # Compute autocorrelation
            timing_autocorr = self._compute_autocorrelation(
                self._ou_timing_history, 
                max_lag=min(20, len(self._ou_timing_history) - 1)
            )
            
            # Theoretical properties
            half_life_timing = np.log(2) / self.params['theta_timing']
            stationary_std_timing = self.params['sigma_timing'] / np.sqrt(
                2 * self.params['theta_timing']
            )
            
            diagnostics['timing'] = {
                'mean_state': float(np.mean(timing_array)),
                'std_state': float(np.std(timing_array)),
                'min_state': float(np.min(timing_array)),
                'max_state': float(np.max(timing_array)),
                'current_state': self._ou_timing_state,
                'autocorrelation': timing_autocorr,
                'half_life_hours': half_life_timing,
                'stationary_std_theory': stationary_std_timing,
                'n_samples': len(timing_array),
            }
        
        if self._ou_threshold_history:
            threshold_array = np.array(self._ou_threshold_history)
            
            threshold_autocorr = self._compute_autocorrelation(
                self._ou_threshold_history,
                max_lag=min(20, len(self._ou_threshold_history) - 1)
            )
            
            half_life_threshold = np.log(2) / self.params['theta_threshold']
            stationary_std_threshold = self.params['sigma_threshold'] / np.sqrt(
                2 * self.params['theta_threshold']
            )
            
            diagnostics['threshold'] = {
                'mean_state': float(np.mean(threshold_array)),
                'std_state': float(np.std(threshold_array)),
                'min_state': float(np.min(threshold_array)),
                'max_state': float(np.max(threshold_array)),
                'current_state': self._ou_threshold_state,
                'autocorrelation': threshold_autocorr,
                'half_life': half_life_threshold,
                'stationary_std_theory': stationary_std_threshold,
                'n_samples': len(threshold_array),
            }
        
        # State reset events
        diagnostics['resets'] = {
            'n_resets': len(self._reset_events),
            'events': self._reset_events,
        }
        
        # Exposure checks
        exposure_log = self.get_exposure_log()
        if exposure_log:
            violations = [log for log in exposure_log if not log['valid']]
            diagnostics['exposure'] = {
                'n_checks': len(exposure_log),
                'n_violations': len(violations),
                'violation_rate': len(violations) / len(exposure_log),
            }
        
        # Adjustments
        adjustment_log = self.get_adjustment_log()
        if adjustment_log:
            diagnostics['adjustments'] = {
                'n_adjustments': len(adjustment_log),
                'last_auc': adjustment_log[-1]['auc_score'],
                'last_direction': adjustment_log[-1]['direction'],
            }
        
        return diagnostics
    
    @staticmethod
    def _compute_autocorrelation(series: List[float], max_lag: int) -> List[float]:
        """
        Compute sample autocorrelation function.
        
        Args:
            series: Time series data
            max_lag: Maximum lag to compute
        
        Returns:
            List of autocorrelation values [ρ(0), ρ(1), ..., ρ(max_lag)]
        """
        if len(series) < max_lag + 1:
            return []
        
        x = np.array(series)
        x = x - np.mean(x)  # De-mean
        
        autocorr = []
        c0 = np.dot(x, x) / len(x)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr.append(1.0)
            else:
                c_lag = np.dot(x[:-lag], x[lag:]) / len(x)
                autocorr.append(c_lag / c0 if c0 > 0 else 0.0)
        
        return autocorr
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"OUPolicy(seed={self.seed}, "
            f"theta_timing={self.params['theta_timing']:.2f}, "
            f"sigma_timing={self.params['sigma_timing']:.2f})"
        )


# ============================================================================
# Predefined Parameter Sets
# ============================================================================

DEFAULT_OU_PARAMS = {
    'theta_timing': 0.5,          # Half-life ≈ 1.4 hours
    'sigma_timing': 1.0,
    'theta_threshold': 0.5,
    'sigma_threshold': 0.05,
    'mu': 0.0,
    'state_reset_threshold': 10.0,
    'track_elapsed_time': True,
    'dt_fixed': 1.0,
    'respect_market_hours': True,
}

FAST_REVERSION_OU_PARAMS = {
    'theta_timing': 1.0,          # Half-life ≈ 0.7 hours (fast)
    'sigma_timing': 1.0,
    'theta_threshold': 1.0,
    'sigma_threshold': 0.05,
    'mu': 0.0,
    'state_reset_threshold': 10.0,
    'track_elapsed_time': True,
    'dt_fixed': 1.0,
    'respect_market_hours': True,
}

SLOW_REVERSION_OU_PARAMS = {
    'theta_timing': 0.1,          # Half-life ≈ 7 hours (slow)
    'sigma_timing': 1.0,
    'theta_threshold': 0.1,
    'sigma_threshold': 0.05,
    'mu': 0.0,
    'state_reset_threshold': 10.0,
    'track_elapsed_time': True,
    'dt_fixed': 1.0,
    'respect_market_hours': True,
}