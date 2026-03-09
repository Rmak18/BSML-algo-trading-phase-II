"""
Abstract base class for all randomization policies.

This defines the interface that Uniform, OU, and Pink policies must implement.

Critical Design Decisions:
1. Abstract methods force concrete implementations
2. Shared methods (exposure checks, adjustment) are provided
3. Logging is built-in for diagnostics

Owner: P4
Week: 2
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any, List
import numpy as np

from .utils import (
    calculate_net_exposure,
    is_within_exposure_tolerance,
    validate_parameter_bounds
)


class RandomizationPolicy(ABC):
    """
    Abstract base class for all randomization policies.
    
    All concrete policies (Uniform, OU, Pink) must inherit from this
    and implement the abstract methods.
    
    Responsibilities:
    - Perturb trade timing and thresholds
    - Maintain exposure invariance (net exposure ≈ 0 ± 5%)
    - Support reproducibility via seeds
    - Provide diagnostics for validation
    - Support dynamic adjustment by P7's adaptive adversary
    
    Attributes:
        seed (int): Master seed for reproducibility
        params (dict): Policy-specific parameters
        rng (np.random.RandomState): Random number generator
    """
    
    def __init__(self, seed: int, params: Dict[str, Any]):
        """
        Initialize the randomization policy.
        
        Args:
            seed: Master seed for reproducibility
            params: Policy-specific parameters (validated by concrete class)
        
        Example:
            >>> # Don't instantiate directly - use concrete classes
            >>> from bsml.policies import UniformPolicy
            >>> policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0, ...})
        """
        self.seed = seed
        self.params = params.copy()  # Copy to avoid external mutations
        
        # Initialize RNG with seed for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Logging structures
        self._perturbation_log: List[Dict[str, Any]] = []
        self._exposure_log: List[Dict[str, Any]] = []
        self._adjustment_log: List[Dict[str, Any]] = []
    
    # ========================================================================
    # Abstract Methods (MUST be implemented by concrete classes)
    # ========================================================================
    
    @abstractmethod
    def perturb_timing(
        self, 
        timestamp: datetime, 
        signal_strength: float = 1.0
    ) -> datetime:
        """
        Add random noise to trade execution timing.
        
        Critical: This is the core randomization logic. Each policy
        implements this differently (uniform, OU, pink noise).
        
        Args:
            timestamp: Original signal timestamp
            signal_strength: Signal magnitude [0, 1] for optional scaling
            
        Returns:
            Perturbed timestamp
        
        Example:
            >>> original = datetime(2025, 7, 15, 10, 30)
            >>> perturbed = policy.perturb_timing(original)
            >>> # Uniform: might return datetime(2025, 7, 15, 11, 47)
            >>> # OU: autocorrelated shift
            >>> # Pink: long-memory shift
        """
        pass
    
    @abstractmethod
    def perturb_threshold(
        self, 
        base_threshold: float, 
        current_price: Optional[float] = None
    ) -> float:
        """
        Add random noise to decision thresholds.
        
        Args:
            base_threshold: Original threshold (e.g., price trigger)
            current_price: Current market price for optional scaling
            
        Returns:
            Perturbed threshold
        
        Example:
            >>> base = 150.00
            >>> perturbed = policy.perturb_threshold(base)
            >>> # Uniform: might return 151.50 (+1%)
            >>> # OU: autocorrelated shift
            >>> # Pink: long-memory shift
        """
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return policy-specific diagnostic metrics.
        
        Must include at minimum:
        - 'policy': str (policy name)
        - 'seed': int (seed used)
        - 'n_perturbations': int (number of perturbations applied)
        
        Returns:
            Dictionary of diagnostics
        
        Example:
            >>> diag = policy.get_diagnostics()
            >>> print(diag['policy'])
            'Uniform'
            >>> print(diag['n_perturbations'])
            150
        """
        pass
    
    # ========================================================================
    # Shared Methods (Provided by base class)
    # ========================================================================
    
    def check_exposure_invariance(
        self, 
        positions_before: Dict[str, float], 
        positions_after: Dict[str, float],
        tolerance: float = 5.0
    ) -> bool:
        """
        Validate that net exposure remains within tolerance.
        
        Critical Design: Log violations but don't block (research mode).
        P1/P5 can analyze violations post-hoc.
        
        Constraint: |net_after - net_before| ≤ tolerance
        
        Args:
            positions_before: {symbol: shares} before perturbation
            positions_after: {symbol: shares} after perturbation
            tolerance: Maximum allowed absolute change (default 5.0 shares)
            
        Returns:
            True if constraint satisfied, False otherwise
        
        Example:
            >>> before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
            >>> after = {'AAPL': 102, 'MSFT': -98}    # Net = 4
            >>> policy.check_exposure_invariance(before, after, tolerance=5.0)
            True
            
            >>> # Violation
            >>> after = {'AAPL': 120, 'MSFT': -80}    # Net = 40
            >>> policy.check_exposure_invariance(before, after, tolerance=5.0)
            False
        """
        net_before = calculate_net_exposure(positions_before)
        net_after = calculate_net_exposure(positions_after)
        
        delta = net_after - net_before
        is_valid = is_within_exposure_tolerance(
            positions_before, 
            positions_after, 
            tolerance
        )
        
        # Log for diagnostics
        self._exposure_log.append({
            'net_before': net_before,
            'net_after': net_after,
            'delta': delta,
            'abs_delta': abs(delta),
            'tolerance': tolerance,
            'valid': is_valid
        })
        
        return is_valid
    
    def adjust_stochasticity(
        self, 
        auc_score: float, 
        direction: str
    ) -> None:
        """
        Adjust randomness level based on adversary feedback.
        
        Called by P7's adaptive adversary to dynamically tune randomness.
        
        Critical Design: 20% adjustment (1.2x or 0.8x) with parameter bounds.
        
        Strategy:
        - If direction='increase': multiply stochastic params by 1.2
        - If direction='decrease': multiply stochastic params by 0.8
        - Clamp to parameter bounds
        
        Args:
            auc_score: Current adversary AUC [0.5, 1.0]
                       0.5 = unpredictable, 1.0 = very predictable
            direction: 'increase' or 'decrease'
        
        Raises:
            ValueError: If direction is not 'increase' or 'decrease'
        
        Example:
            >>> policy = UniformPolicy(seed=42, params={
            ...     'timing_range_hours': 2.0,
            ...     'threshold_pct': 0.10
            ... })
            >>> 
            >>> # P7 detects high predictability
            >>> policy.adjust_stochasticity(auc_score=0.85, direction='increase')
            >>> 
            >>> # Parameters are now:
            >>> # timing_range_hours: 2.4 (was 2.0)
            >>> # threshold_pct: 0.12 (was 0.10)
        """
        if direction not in ['increase', 'decrease']:
            raise ValueError(
                f"Direction must be 'increase' or 'decrease', got '{direction}'"
            )
        
        # Adjustment factor
        factor = 1.2 if direction == 'increase' else 0.8
        
        # Store old params for logging
        old_params = self.params.copy()
        
        # Apply to all stochastic parameters
        # Look for common naming patterns: range, pct, sigma, scale
        stochastic_keywords = ['range', 'pct', 'sigma', 'scale', 'volatility']
        
        for key in self.params:
            if any(keyword in key.lower() for keyword in stochastic_keywords):
                old_value = self.params[key]
                new_value = old_value * factor
                
                # Apply bounds if defined
                # (Concrete policies can override with tighter bounds)
                if 'range' in key.lower():
                    new_value = validate_parameter_bounds(
                        key, new_value, 
                        min_value=0.01, 
                        max_value=10.0
                    )
                elif 'pct' in key.lower():
                    new_value = validate_parameter_bounds(
                        key, new_value,
                        min_value=0.001,
                        max_value=0.5
                    )
                
                self.params[key] = new_value
        
        # Log adjustment
        self._adjustment_log.append({
            'auc_score': auc_score,
            'direction': direction,
            'factor': factor,
            'params_before': old_params,
            'params_after': self.params.copy()
        })
        
        print(f"[{self.__class__.__name__}] Adjusted {direction}: "
              f"factor={factor:.2f}, new params={self.params}")
    
    def get_exposure_log(self) -> List[Dict[str, Any]]:
        """
        Get the full log of exposure checks.
        
        Returns:
            List of dictionaries with exposure check history
        
        Example:
            >>> log = policy.get_exposure_log()
            >>> print(f"Total checks: {len(log)}")
            >>> print(f"Violations: {sum(1 for x in log if not x['valid'])}")
        """
        return self._exposure_log.copy()
    
    def get_adjustment_log(self) -> List[Dict[str, Any]]:
        """
        Get the full log of stochasticity adjustments.
        
        Returns:
            List of dictionaries with adjustment history
        
        Example:
            >>> log = policy.get_adjustment_log()
            >>> for adj in log:
            ...     print(f"AUC={adj['auc_score']:.2f}, direction={adj['direction']}")
        """
        return self._adjustment_log.copy()
    
    def reset_logs(self) -> None:
        """
        Clear all logs (perturbation, exposure, adjustment).
        
        Useful for starting fresh between experiments.
        """
        self._perturbation_log.clear()
        self._exposure_log.clear()
        self._adjustment_log.clear()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}(seed={self.seed}, "
            f"params={self.params})"
        )
