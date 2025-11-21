"""
Pink (1/f) noise randomization policy.

Adds noise with long-range temporal correlations. Pink noise exhibits
memory over many time scales, making perturbations more realistic for
modeling slow-changing market conditions.

Mathematical Formulation:
    Power Spectral Density: S(f) = 1/f^α
    where α ≈ 1 for pink noise

Generation Method: FFT (Frequency Domain)
1. Create frequency array
2. Compute amplitudes A(f) = 1/f^α
3. Add random phases φ ~ Uniform(0, 2π)
4. Construct complex spectrum
5. IFFT to time domain
6. Normalize and scale

Critical Design Decisions:
1. Pre-generate buffer using FFT for precision
2. De-mean to remove DC drift
3. Regenerate when exhausted (with warning)
4. Independent sequences for timing and threshold

Owner: P4
Week: 3
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import warnings

from .base_policy import RandomizationPolicy
from .utils import clamp_to_market_hours


class PinkNoisePolicy(RandomizationPolicy):
    """
    Pink (1/f) noise randomization policy.
    
    Provides perturbations with long-range correlations across multiple time
    scales. Unlike Uniform (no correlation) or OU (exponential decay), pink
    noise maintains correlations over long periods.
    
    Parameters:
        alpha (float): Spectral exponent (1.0 = pink, 0.0 = white, 2.0 = brown)
        scale_timing (float): Target std dev for timing perturbations (hours)
        scale_threshold (float): Target std dev for threshold perturbations (fractional)
        buffer_size (int): Number of samples to pre-generate (default 10000)
    
    Statistical Properties:
        - Power Spectral Density: S(f) ∝ 1/f^α
        - Long-range dependence (slow autocorrelation decay)
        - Zero mean (after de-meaning)
        - Hurst exponent: H ≈ (α+1)/2
    
    Example:
        >>> policy = PinkNoisePolicy(
        ...     seed=42,
        ...     params={
        ...         'alpha': 1.0,              # Pink noise
        ...         'scale_timing': 1.5,       # ±1.5 hours std
        ...         'scale_threshold': 0.075,  # ±7.5% std
        ...         'buffer_size': 10000,
        ...     }
        ... )
        >>> 
        >>> # Perturb timing (long-memory correlated)
        >>> ts1 = datetime(2025, 7, 15, 10, 0)
        >>> perturbed1 = policy.perturb_timing(ts1)
        >>> 
        >>> # Many perturbations later, still correlated
        >>> for i in range(100):
        ...     policy.perturb_timing(datetime(2025, 7, 15, 10+i, 0))
    """
    
    # Parameter bounds
    PARAMETER_BOUNDS = {
        'alpha': {'min': 0.0, 'max': 2.0, 'default': 1.0},
        'scale_timing': {'min': 0.1, 'max': 5.0, 'default': 1.5},
        'scale_threshold': {'min': 0.01, 'max': 0.2, 'default': 0.075},
        'buffer_size': {'min': 100, 'max': 100000, 'default': 10000},
    }
    
    def __init__(self, seed: int, params: Dict[str, Any]):
        """
        Initialize Pink noise policy.
        
        Args:
            seed: Master seed for reproducibility
            params: Must contain 'alpha', 'scale_timing', etc.
        
        Raises:
            ValueError: If required parameters are missing
        """
        # Validate required parameters
        required = ['alpha', 'scale_timing', 'scale_threshold']
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"PinkNoisePolicy requires parameters: {', '.join(missing)}. "
                f"Got: {list(params.keys())}"
            )
        
        # Set defaults
        if 'buffer_size' not in params:
            params['buffer_size'] = 10000
        if 'respect_market_hours' not in params:
            params['respect_market_hours'] = True
        
        # Initialize base class
        super().__init__(seed, params)
        
        # Buffer state
        self._timing_buffer: np.ndarray = np.array([])
        self._threshold_buffer: np.ndarray = np.array([])
        self._timing_index = 0
        self._threshold_index = 0
        self._buffer_regenerations = 0
        
        # Validate parameters
        self._validate_params()
        
        # Generate initial buffers
        self._regenerate_buffers()
    
    def _validate_params(self) -> None:
        """Validate pink noise parameters."""
        assert 0.0 <= self.params['alpha'] <= 2.0, "alpha must be in [0, 2]"
        assert self.params['scale_timing'] > 0, "scale_timing must be positive"
        assert self.params['scale_threshold'] > 0, "scale_threshold must be positive"
        assert self.params['buffer_size'] >= 100, "buffer_size must be >= 100"
    
    def _generate_pink_noise_fft(self, n: int, alpha: float) -> np.ndarray:
        """
        Generate pink noise using FFT method.
        
        This is the standard frequency-domain method for generating
        1/f^α noise with precise spectral properties.
        
        Args:
            n: Number of samples
            alpha: Spectral exponent (1.0 = pink, 0.0 = white)
        
        Returns:
            Real-valued pink noise sequence (de-meaned)
        """
        # Frequency array (exclude DC component at f=0)
        freqs = np.fft.rfftfreq(n)[1:]  # Start from index 1
        
        # Amplitude spectrum: A(f) = 1/f^α
        amplitudes = 1.0 / (freqs ** alpha)
        
        # Random phases using policy's RNG
        phases = self.rng.uniform(0, 2 * np.pi, len(amplitudes))
        
        # Complex spectrum
        spectrum = amplitudes * np.exp(1j * phases)
        
        # Add zero at DC (f=0)
        spectrum_full = np.concatenate([[0], spectrum])
        
        # IFFT to time domain
        pink_noise = np.fft.irfft(spectrum_full, n=n)
        
        # De-mean to remove DC drift
        pink_noise = pink_noise - np.mean(pink_noise)
        
        return pink_noise
    
    def _regenerate_buffers(self) -> None:
        """
        Generate fresh pink noise buffers.
        
        Normalizes to target standard deviation and de-means.
        """
        n = self.params['buffer_size']
        alpha = self.params['alpha']
        
        # Generate raw pink noise
        timing_raw = self._generate_pink_noise_fft(n, alpha)
        threshold_raw = self._generate_pink_noise_fft(n, alpha)
        
        # Normalize to unit variance, then scale to target
        timing_std = np.std(timing_raw)
        threshold_std = np.std(threshold_raw)
        
        if timing_std > 0:
            self._timing_buffer = (timing_raw / timing_std) * self.params['scale_timing']
        else:
            self._timing_buffer = timing_raw
        
        if threshold_std > 0:
            self._threshold_buffer = (threshold_raw / threshold_std) * self.params['scale_threshold']
        else:
            self._threshold_buffer = threshold_raw
        
        # Reset indices
        self._timing_index = 0
        self._threshold_index = 0
        
        self._buffer_regenerations += 1
        
        if self._buffer_regenerations > 1:
            warnings.warn(
                f"Pink noise buffer exhausted and regenerated "
                f"(event #{self._buffer_regenerations}). "
                f"Consider increasing buffer_size if this happens frequently."
            )
    
    def perturb_timing(
        self, 
        timestamp: datetime, 
        signal_strength: float = 1.0
    ) -> datetime:
        """
        Apply pink noise perturbation to timing.
        
        Critical Logic:
        1. Check if buffer exhausted → regenerate if needed
        2. Get next pink noise sample from buffer
        3. Apply shift to timestamp
        4. Clamp to market hours
        
        Args:
            timestamp: Original signal timestamp
            signal_strength: Optional scaling factor
        
        Returns:
            Perturbed timestamp
        """
        # Check if buffer exhausted
        if self._timing_index >= len(self._timing_buffer):
            self._regenerate_buffers()
        
        # Get next pink noise sample
        shift_hours = self._timing_buffer[self._timing_index] * signal_strength
        self._timing_index += 1
        
        # Apply shift
        perturbed_time = timestamp + timedelta(hours=shift_hours)
        
        # Clamp to market hours if requested
        if self.params.get('respect_market_hours', True):
            from datetime import time
            perturbed_time = clamp_to_market_hours(
                perturbed_time,
                time(9, 30),
                time(16, 0)
            )
        
        return perturbed_time
    
    def perturb_threshold(
        self, 
        base_threshold: float, 
        current_price: Optional[float] = None
    ) -> float:
        """
        Apply pink noise perturbation to threshold.
        
        Args:
            base_threshold: Original threshold value
            current_price: Not used (kept for interface compatibility)
        
        Returns:
            Perturbed threshold (always positive)
        """
        # Check if buffer exhausted
        if self._threshold_index >= len(self._threshold_buffer):
            self._regenerate_buffers()
        
        # Get next pink noise sample
        perturbation = self._threshold_buffer[self._threshold_index]
        self._threshold_index += 1
        
        # Apply as multiplicative factor
        perturbed = base_threshold * (1.0 + perturbation)
        
        # Ensure positive
        perturbed = max(perturbed, 0.01)
        
        return perturbed
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return Pink noise policy diagnostics.
        
        Includes:
        - Standard diagnostics (policy, seed, params)
        - Buffer usage statistics
        - Spectral properties (if scipy available)
        - Regeneration events
        
        Returns:
            Dictionary with comprehensive diagnostics
        """
        diagnostics = {
            'policy': 'Pink',
            'seed': self.seed,
            'params': self.params.copy(),
            'n_perturbations': self._timing_index + self._threshold_index,
        }
        
        # Buffer statistics
        diagnostics['buffer'] = {
            'size': self.params['buffer_size'],
            'timing_usage': self._timing_index / len(self._timing_buffer) if len(self._timing_buffer) > 0 else 0,
            'threshold_usage': self._threshold_index / len(self._threshold_buffer) if len(self._threshold_buffer) > 0 else 0,
            'regenerations': self._buffer_regenerations,
        }
        
        # Buffer statistics
        if len(self._timing_buffer) > 0:
            diagnostics['timing'] = {
                'buffer_mean': float(np.mean(self._timing_buffer)),
                'buffer_std': float(np.std(self._timing_buffer)),
                'buffer_min': float(np.min(self._timing_buffer)),
                'buffer_max': float(np.max(self._timing_buffer)),
                'current_index': self._timing_index,
            }
        
        if len(self._threshold_buffer) > 0:
            diagnostics['threshold'] = {
                'buffer_mean': float(np.mean(self._threshold_buffer)),
                'buffer_std': float(np.std(self._threshold_buffer)),
                'buffer_min': float(np.min(self._threshold_buffer)),
                'buffer_max': float(np.max(self._threshold_buffer)),
                'current_index': self._threshold_index,
            }
        
        # Try to compute power spectral density
        try:
            from scipy import signal as scipy_signal
            
            # Compute PSD on first 1000 samples
            n_samples = min(1000, len(self._timing_buffer))
            series = self._timing_buffer[:n_samples]
            
            frequencies, power = scipy_signal.welch(
                series, 
                nperseg=min(256, n_samples//4)
            )
            
            # Fit log-log line to estimate spectral slope
            valid = (frequencies > 0.01) & (frequencies < 0.5)
            if np.sum(valid) > 10:
                log_f = np.log10(frequencies[valid])
                log_p = np.log10(power[valid])
                slope, _ = np.polyfit(log_f, log_p, 1)
                
                diagnostics['spectral'] = {
                    'measured_slope': float(slope),
                    'expected_slope': float(-self.params['alpha']),
                    'slope_error': float(abs(slope + self.params['alpha'])),
                }
        except ImportError:
            # scipy not available, skip spectral analysis
            pass
        
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
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PinkNoisePolicy(seed={self.seed}, "
            f"alpha={self.params['alpha']:.1f}, "
            f"scale_timing={self.params['scale_timing']:.2f})"
        )


# ============================================================================
# Predefined Parameter Sets
# ============================================================================

DEFAULT_PINK_PARAMS = {
    'alpha': 1.0,                # True pink noise
    'scale_timing': 1.5,
    'scale_threshold': 0.075,
    'buffer_size': 10000,
    'respect_market_hours': True,
}

WHITE_NOISE_PARAMS = {
    'alpha': 0.0,                # White noise (no correlation)
    'scale_timing': 1.5,
    'scale_threshold': 0.075,
    'buffer_size': 10000,
    'respect_market_hours': True,
}

BROWN_NOISE_PARAMS = {
    'alpha': 2.0,                # Brown noise (very strong correlation)
    'scale_timing': 1.5,
    'scale_threshold': 0.075,
    'buffer_size': 10000,
    'respect_market_hours': True,
}