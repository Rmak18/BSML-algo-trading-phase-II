"""
Test suite for OU (Ornstein-Uhlenbeck) policy.

Tests:
- Parameter validation
- Mean reversion
- Stationary variance
- Autocorrelation structure
- State drift monitoring
- Adaptive adjustment
- Diagnostic output

Owner: P4
Week: 3
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

# FIXED: Correct import path
from bsml.policies import OUPolicy


class TestOUPolicyBasics:
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test OU policy initializes correctly."""
        policy = OUPolicy(
            seed=42,
            params={
                'theta_timing': 0.5,
                'sigma_timing': 1.0,
                'theta_threshold': 0.5,
                'sigma_threshold': 0.05,
            }
        )
        
        assert policy.seed == 42
        assert policy._ou_timing_state == 0.0
        assert policy._ou_threshold_state == 0.0
        assert len(policy._ou_timing_history) == 0
    
    def test_missing_required_params(self):
        """Test that missing required params raises error."""
        with pytest.raises(ValueError, match="requires parameters"):
            OUPolicy(seed=42, params={})
    
    def test_invalid_theta(self):
        """Test that negative theta raises error."""
        with pytest.raises(AssertionError):
            OUPolicy(seed=42, params={
                'theta_timing': -0.5,
                'sigma_timing': 1.0,
                'theta_threshold': 0.5,
                'sigma_threshold': 0.05,
            })
    
    def test_invalid_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(AssertionError):
            OUPolicy(seed=42, params={
                'theta_timing': 0.5,
                'sigma_timing': -1.0,
                'theta_threshold': 0.5,
                'sigma_threshold': 0.05,
            })


class TestOUPerturbations:
    """Test perturbation mechanics."""
    
    def test_perturb_timing_basic(self):
        """Test basic timing perturbation."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        timestamp = datetime(2025, 7, 15, 10, 30)
        perturbed = policy.perturb_timing(timestamp)
        
        # Should return a datetime
        assert isinstance(perturbed, datetime)
        
        # Should be within market hours
        assert 9 <= perturbed.hour <= 16
        
        # State should have been updated
        assert len(policy._ou_timing_history) == 1
    
    def test_perturb_timing_market_hours_clamping(self):
        """Test that timing is clamped to market hours."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 5.0,  # Large sigma to increase chance of clamping
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Test many perturbations
        for _ in range(100):
            timestamp = datetime(2025, 7, 15, 10, 0)
            perturbed = policy.perturb_timing(timestamp)
            
            # Should always be within market hours
            assert 9 <= perturbed.hour <= 16
            if perturbed.hour == 9:
                assert perturbed.minute >= 30
            if perturbed.hour == 16:
                assert perturbed.minute == 0 and perturbed.second == 0
    
    def test_perturb_threshold_basic(self):
        """Test basic threshold perturbation."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Need to perturb timing first to set up state
        policy.perturb_timing(datetime(2025, 7, 15, 10, 0))
        
        base_threshold = 150.0
        perturbed = policy.perturb_threshold(base_threshold)
        
        # Should return a positive float
        assert isinstance(perturbed, float)
        assert perturbed > 0
        
        # State should have been updated
        assert len(policy._ou_threshold_history) == 1


class TestOUStatisticalProperties:
    """Test OU process statistical properties."""
    
    def test_mean_reversion(self):
        """Test that OU process reverts to mean."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
            'mu': 0.0,
        })
        
        # Run many perturbations
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(1000):
            policy.perturb_timing(timestamp)
            timestamp += timedelta(hours=1)
        
        # Mean should be near μ (0.0)
        states = np.array(policy._ou_timing_history)
        mean_state = np.mean(states)
        
        # Allow 20% tolerance
        assert abs(mean_state - policy.params['mu']) < 0.2, \
            f"Mean state {mean_state:.3f} far from μ={policy.params['mu']}"
    
    def test_stationary_variance(self):
        """Test OU stationary variance matches theory."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Run to reach stationarity
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(1000):
            policy.perturb_timing(timestamp)
            timestamp += timedelta(hours=1)
        
        # Variance should match σ²/(2θ)
        states = np.array(policy._ou_timing_history)
        measured_std = np.std(states)
        
        theta = policy.params['theta_timing']
        sigma = policy.params['sigma_timing']
        expected_std = sigma / np.sqrt(2 * theta)
        
        relative_error = abs(measured_std - expected_std) / expected_std
        
        # Allow 20% tolerance
        assert relative_error < 0.2, \
            f"Std {measured_std:.3f} doesn't match theory {expected_std:.3f}"
    
    def test_autocorrelation_structure(self):
        """Test OU autocorrelation matches theory."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Generate sequence
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(500):
            policy.perturb_timing(timestamp)
            timestamp += timedelta(hours=1)
        
        # Compute autocorrelation
        acf = OUPolicy._compute_autocorrelation(
            policy._ou_timing_history, 
            max_lag=10
        )
        
        # Check lag-1 autocorrelation
        if len(acf) > 1:
            theta = policy.params['theta_timing']
            expected_acf1 = np.exp(-theta * 1.0)
            
            # Allow 20% tolerance
            assert abs(acf[1] - expected_acf1) < 0.2, \
                f"ACF(1) = {acf[1]:.3f} doesn't match theory {expected_acf1:.3f}"


class TestOUStateDriftMonitoring:
    """Test state drift monitoring and reset."""
    
    def test_state_reset_on_drift(self):
        """Test state reset when drift exceeds threshold."""
        # Use aggressive parameters to trigger reset
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.01,     # Very slow reversion
            'sigma_timing': 5.0,      # High volatility
            'theta_threshold': 0.01,
            'sigma_threshold': 0.05,
            'state_reset_threshold': 2.0,  # Low threshold
        })
        
        # Run many perturbations
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(500):
            policy.perturb_timing(timestamp)
            timestamp += timedelta(hours=1)
        
        # Should have had at least one reset
        assert len(policy._reset_events) > 0, "Expected at least one state reset"


class TestOUAdaptiveAdjustment:
    """Test adaptive parameter adjustment."""
    
    def test_adjust_stochasticity_increase(self):
        """Test increasing stochasticity."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        initial_sigma = policy.params['sigma_timing']
        
        # Increase stochasticity
        policy.adjust_stochasticity(auc_score=0.85, direction='increase')
        
        # Sigma should have increased
        assert policy.params['sigma_timing'] > initial_sigma
    
    def test_adjust_stochasticity_decrease(self):
        """Test decreasing stochasticity."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        initial_sigma = policy.params['sigma_timing']
        
        # Decrease stochasticity
        policy.adjust_stochasticity(auc_score=0.55, direction='decrease')
        
        # Sigma should have decreased
        assert policy.params['sigma_timing'] < initial_sigma


class TestOUDiagnostics:
    """Test diagnostic output."""
    
    def test_get_diagnostics(self):
        """Test diagnostic data collection."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Run perturbations
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(100):
            policy.perturb_timing(timestamp)
            policy.perturb_threshold(150.0)
            timestamp += timedelta(hours=1)
        
        # Get diagnostics
        diag = policy.get_diagnostics()
        
        # Check structure
        assert diag['policy'] == 'OU'
        assert diag['seed'] == 42
        assert 'timing' in diag
        assert 'threshold' in diag
        assert 'autocorrelation' in diag['timing']
        assert 'half_life_hours' in diag['timing']
    
    def test_diagnostics_autocorrelation(self):
        """Test that diagnostics include autocorrelation."""
        policy = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Run perturbations
        timestamp = datetime(2025, 7, 15, 10, 0)
        for _ in range(100):
            policy.perturb_timing(timestamp)
            timestamp += timedelta(hours=1)
        
        diag = policy.get_diagnostics()
        
        # Should have autocorrelation data
        assert 'autocorrelation' in diag['timing']
        acf = diag['timing']['autocorrelation']
        assert len(acf) > 0
        assert acf[0] == 1.0  # ACF(0) = 1


class TestOUReproducibility:
    """Test reproducibility with seeds."""
    
    def test_same_seed_same_sequence(self):
        """Test same seed gives same sequence."""
        policy1 = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        policy2 = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Generate sequences
        timestamp = datetime(2025, 7, 15, 10, 0)
        seq1 = []
        seq2 = []
        
        for _ in range(100):
            policy1.perturb_timing(timestamp)
            policy2.perturb_timing(timestamp)
            
            seq1.append(policy1._ou_timing_state)
            seq2.append(policy2._ou_timing_state)
            
            timestamp += timedelta(hours=1)
        
        # Sequences should be identical
        assert np.allclose(seq1, seq2), "Same seed didn't produce same sequence"
    
    def test_different_seed_different_sequence(self):
        """Test different seeds give different sequences."""
        policy1 = OUPolicy(seed=42, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        policy2 = OUPolicy(seed=43, params={
            'theta_timing': 0.5,
            'sigma_timing': 1.0,
            'theta_threshold': 0.5,
            'sigma_threshold': 0.05,
        })
        
        # Generate sequences
        timestamp = datetime(2025, 7, 15, 10, 0)
        seq1 = []
        seq2 = []
        
        for _ in range(100):
            policy1.perturb_timing(timestamp)
            policy2.perturb_timing(timestamp)
            
            seq1.append(policy1._ou_timing_state)
            seq2.append(policy2._ou_timing_state)
            
            timestamp += timedelta(hours=1)
        
        # Sequences should be different
        assert not np.allclose(seq1, seq2), "Different seeds produced same sequence"