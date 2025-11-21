"""
Test suite for Pink Noise policy.

Tests:
- Parameter validation
- Buffer management
- Spectral properties
- Zero mean
- Adaptive adjustment
- Diagnostic output

Owner: P4
Week: 3
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

# FIXED: Correct import path
from bsml.policies import PinkNoisePolicy


# (rest of the file stays the same as before)
class TestPinkPolicyBasics:
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test pink noise policy initializes correctly."""
        policy = PinkNoisePolicy(
            seed=42,
            params={
                'alpha': 1.0,
                'scale_timing': 1.5,
                'scale_threshold': 0.075,
                'buffer_size': 1000,
            }
        )
        
        assert policy.seed == 42
        assert len(policy._timing_buffer) == 1000
        assert len(policy._threshold_buffer) == 1000
        assert policy._timing_index == 0
        assert policy._buffer_regenerations == 1  # Generated once at init
    
    def test_missing_required_params(self):
        """Test that missing required params raises error."""
        with pytest.raises(ValueError, match="requires parameters"):
            PinkNoisePolicy(seed=42, params={})
    
    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(AssertionError):
            PinkNoisePolicy(seed=42, params={
                'alpha': -1.0,
                'scale_timing': 1.5,
                'scale_threshold': 0.075,
            })
        
        with pytest.raises(AssertionError):
            PinkNoisePolicy(seed=42, params={
                'alpha': 3.0,
                'scale_timing': 1.5,
                'scale_threshold': 0.075,
            })
    
    def test_buffer_too_small(self):
        """Test that buffer too small raises error."""
        with pytest.raises(AssertionError):
            PinkNoisePolicy(seed=42, params={
                'alpha': 1.0,
                'scale_timing': 1.5,
                'scale_threshold': 0.075,
                'buffer_size': 50,
            })


class TestPinkPerturbations:
    """Test perturbation mechanics."""
    
    def test_perturb_timing_basic(self):
        """Test basic timing perturbation."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        timestamp = datetime(2025, 7, 15, 10, 30)
        perturbed = policy.perturb_timing(timestamp)
        
        # Should return a datetime
        assert isinstance(perturbed, datetime)
        
        # Should be within market hours
        assert 9 <= perturbed.hour <= 16
        
        # Index should have advanced
        assert policy._timing_index == 1
    
    def test_perturb_timing_market_hours_clamping(self):
        """Test that timing is clamped to market hours."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 5.0,  # Large scale to increase chance of clamping
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        # Test many perturbations
        for _ in range(100):
            timestamp = datetime(2025, 7, 15, 10, 0)
            perturbed = policy.perturb_timing(timestamp)
            
            # Should always be within market hours
            assert 9 <= perturbed.hour <= 16
    
    def test_perturb_threshold_basic(self):
        """Test basic threshold perturbation."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        base_threshold = 150.0
        perturbed = policy.perturb_threshold(base_threshold)
        
        # Should return a positive float
        assert isinstance(perturbed, float)
        assert perturbed > 0
        
        # Index should have advanced
        assert policy._threshold_index == 1


class TestPinkBufferManagement:
    """Test buffer management."""
    
    def test_buffer_exhaustion_and_regeneration(self):
        """Test buffer regeneration when exhausted."""
        # Use tiny buffer to trigger exhaustion
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 100,
        })
        
        # Use entire buffer and more
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(150):  # More than buffer size
            with pytest.warns(UserWarning, match="buffer exhausted"):
                policy.perturb_timing(timestamp)
                timestamp += timedelta(minutes=30)
        
        # Should have regenerated at least once
        assert policy._buffer_regenerations > 1


class TestPinkStatisticalProperties:
    """Test pink noise statistical properties."""
    
    def test_zero_mean(self):
        """Test that pink noise has zero mean (after de-meaning)."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        buffer_mean = np.mean(policy._timing_buffer)
        buffer_std = np.std(policy._timing_buffer)
        
        # Mean should be very close to zero (after de-meaning)
        assert abs(buffer_mean) < 0.1 * buffer_std, \
            f"Buffer mean {buffer_mean:.4f} not near zero (std={buffer_std:.4f})"
    
    def test_spectral_slope(self):
        """Test pink noise spectral slope (requires scipy)."""
        try:
            from scipy import signal as scipy_signal
        except ImportError:
            pytest.skip("scipy not available")
        
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 2000,
        })
        
        # Use buffer for PSD
        series = policy._timing_buffer[:1000]
        frequencies, power = scipy_signal.welch(series, nperseg=256)
        
        # Exclude very low frequencies
        valid = (frequencies > 0.01) & (frequencies < 0.5)
        freqs = frequencies[valid]
        pwr = power[valid]
        
        # Log-log fit
        log_f = np.log10(freqs)
        log_p = np.log10(pwr)
        slope, _ = np.polyfit(log_f, log_p, 1)
        
        expected_slope = -policy.params['alpha']
        
        # Allow 50% tolerance (spectral estimation is noisy)
        assert abs(slope - expected_slope) < 0.5, \
            f"Spectral slope {slope:.2f} doesn't match α={expected_slope:.2f}"


class TestPinkReproducibility:
    """Test reproducibility with seeds."""
    
    def test_same_seed_same_sequence(self):
        """Test same seed gives same sequence."""
        policy1 = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        policy2 = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        # Buffers should be identical
        assert np.allclose(policy1._timing_buffer, policy2._timing_buffer), \
            "Same seed didn't produce same buffer"
    
    def test_different_seed_different_sequence(self):
        """Test different seeds give different sequences."""
        policy1 = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        policy2 = PinkNoisePolicy(seed=43, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        # Buffers should be different
        assert not np.allclose(policy1._timing_buffer, policy2._timing_buffer), \
            "Different seeds produced same buffer"


class TestPinkAdaptiveAdjustment:
    """Test adaptive parameter adjustment."""
    
    def test_adjust_stochasticity_regenerates_buffer(self):
        """Test that adjustment regenerates buffer."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        initial_regens = policy._buffer_regenerations
        
        # Adjust should trigger regeneration
        policy.adjust_stochasticity(auc_score=0.85, direction='increase')
        
        # Should have regenerated
        assert policy._buffer_regenerations > initial_regens


class TestPinkDiagnostics:
    """Test diagnostic output."""
    
    def test_get_diagnostics(self):
        """Test diagnostic data collection."""
        policy = PinkNoisePolicy(seed=42, params={
            'alpha': 1.0,
            'scale_timing': 1.5,
            'scale_threshold': 0.075,
            'buffer_size': 1000,
        })
        
        # Run perturbations
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(100):
            policy.perturb_timing(timestamp)
            policy.perturb_threshold(150.0)
            timestamp += timedelta(minutes=30)
        
        # Get diagnostics
        diag = policy.get_diagnostics()
        
        # Check structure
        assert diag['policy'] == 'Pink'
        assert diag['seed'] == 42
        assert 'buffer' in diag
        assert 'timing' in diag
        assert 'threshold' in diag