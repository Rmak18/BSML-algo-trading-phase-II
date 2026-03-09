# Randomization Policy API Specification
**Project:** BSML - Randomized Execution Research  
**Owner:** Neel (Randomization Modules)  
**Version:** 1.0  
**Date:** November 10, 2025  
**Status:** Week 1 Deliverable

---

## Purpose

This document defines the interface for randomization policies that perturb the baseline trading strategy while preserving portfolio exposure constraints. It serves as the contract between P4's randomization modules and the rest of the system.

---

## System Architecture Overview

```
┌─────────────────┐
│ Baseline (P2)   │ generates signals
│ Strategy        │ (time, threshold, size)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Randomization   │ ← YOU ARE HERE (P4)
│ Policy          │ perturbs signals while keeping exposure ~0
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Backtesting     │ executes perturbed orders
│ Harness (P3)    │ with realistic costs
└─────────────────┘
```

---

## Core Interface

### Base Class: `RandomizationPolicy`

All randomization policies (Uniform, OU, Pink) inherit from this abstract base class.

```python
from abc import ABC, abstractmethod
import numpy as np

class RandomizationPolicy(ABC):
    """
    Abstract base class for all randomization policies.
    
    Responsibilities:
    - Perturb trade timing and thresholds
    - Maintain exposure invariance (net exposure ≈ 0 ± 5%)
    - Support reproducibility via seeds
    - Provide diagnostics for validation
    """
    
    def __init__(self, seed: int, params: dict):
        """
        Initialize the randomization policy.
        
        Args:
            seed (int): Master seed for reproducibility
            params (dict): Policy-specific parameters
            
        Example:
            policy = UniformPolicy(
                seed=42,
                params={
                    'timing_range_hours': 2.0,
                    'threshold_pct': 0.10
                }
            )
        """
        self.seed = seed
        self.params = params
        self.rng = np.random.RandomState(seed)
        self._perturbation_log = []
    
    @abstractmethod
    def perturb_timing(self, timestamp, signal_strength=1.0):
        """
        Add random noise to trade execution timing.
        
        Args:
            timestamp (datetime): Original signal timestamp
            signal_strength (float): Signal magnitude [0, 1] for scaling
            
        Returns:
            datetime: Perturbed timestamp
            
        Example:
            original = datetime(2025, 7, 15, 10, 30)  # 10:30 AM
            perturbed = policy.perturb_timing(original)
            # Returns: datetime(2025, 7, 15, 11, 17)  # 11:17 AM
        """
        pass
    
    @abstractmethod
    def perturb_threshold(self, base_threshold, current_price=None):
        """
        Add random noise to decision thresholds.
        
        Args:
            base_threshold (float): Original threshold (e.g., price trigger)
            current_price (float, optional): Current market price for scaling
            
        Returns:
            float: Perturbed threshold
            
        Example:
            base = 150.00  # Original buy trigger
            perturbed = policy.perturb_threshold(base)
            # Returns: 151.50  # New buy trigger (+1%)
        """
        pass
    
    def check_exposure_invariance(self, positions_before, positions_after):
        """
        Validate that net exposure remains within tolerance.
        
        Constraint: |net_after - net_before| ≤ 5% of NAV
        
        Args:
            positions_before (dict): {symbol: shares} before perturbation
            positions_after (dict): {symbol: shares} after perturbation
            
        Returns:
            bool: True if constraint satisfied, False otherwise
            
        Example:
            before = {'AAPL': 100, 'MSFT': -50, 'GOOGL': -50}  # net = 0
            after = {'AAPL': 110, 'MSFT': -55, 'GOOGL': -50}   # net = 5
            
            is_valid = policy.check_exposure_invariance(before, after)
            # Returns: True (within ±5 share tolerance)
        """
        net_before = sum(positions_before.values())
        net_after = sum(positions_after.values())
        tolerance = 0.05  # 5% of NAV
        
        is_valid = abs(net_after - net_before) <= tolerance
        
        # Log for debugging
        self._perturbation_log.append({
            'net_before': net_before,
            'net_after': net_after,
            'valid': is_valid
        })
        
        return is_valid
    
    @abstractmethod
    def get_diagnostics(self):
        """
        Return policy-specific diagnostic metrics.
        
        Returns:
            dict: Diagnostics for validation
            
        Example for Uniform:
            {
                'mean_timing_shift': 0.02,  # hours
                'std_timing_shift': 1.15,
                'mean_threshold_shift': 0.001,  # percentage
                'exposure_violations': 0  # count
            }
        """
        pass
    
    def adjust_stochasticity(self, auc_score, direction):
        """
        Adjust randomness level based on adversary feedback.
        
        Called by P7's adaptive adversary to dynamically tune randomness.
        
        Args:
            auc_score (float): Current adversary AUC [0.5, 1.0]
                             0.5 = unpredictable, 1.0 = very predictable
            direction (str): 'increase' or 'decrease'
            
        Returns:
            None (modifies self.params in-place)
            
        Example:
            # Adversary is predicting too well (AUC = 0.85)
            policy.adjust_stochasticity(auc_score=0.85, direction='increase')
            # Internally: increases timing_range and threshold_pct by 20%
        """
        adjustment_factor = 1.2 if direction == 'increase' else 0.8
        
        # Apply to all stochastic parameters
        for key in self.params:
            if 'range' in key or 'pct' in key or 'sigma' in key:
                self.params[key] *= adjustment_factor
        
        print(f"Adjusted stochasticity {direction}: new params = {self.params}")
```

---

## Policy Implementations

### 1. Uniform Policy (Week 2)

**Purpose:** Add independent, identically distributed (IID) uniform noise.

**Use Case:** Baseline randomization with no temporal correlation.

```python
class UniformPolicy(RandomizationPolicy):
    """
    Adds uniform random noise to timing and thresholds.
    
    Parameters:
        timing_range_hours (float): Max shift in hours (e.g., 2.0 = ±2 hours)
        threshold_pct (float): Max threshold shift as % (e.g., 0.10 = ±10%)
    """
    
    def perturb_timing(self, timestamp, signal_strength=1.0):
        # Shift by uniform random hours in [-range, +range]
        shift_hours = self.rng.uniform(
            -self.params['timing_range_hours'],
            self.params['timing_range_hours']
        )
        return timestamp + timedelta(hours=shift_hours)
    
    def perturb_threshold(self, base_threshold, current_price=None):
        # Shift by uniform random percentage in [-pct, +pct]
        shift_pct = self.rng.uniform(
            -self.params['threshold_pct'],
            self.params['threshold_pct']
        )
        return base_threshold * (1 + shift_pct)
    
    def get_diagnostics(self):
        return {
            'policy': 'Uniform',
            'timing_range': self.params['timing_range_hours'],
            'threshold_range': self.params['threshold_pct'],
            'exposure_violations': sum(
                1 for log in self._perturbation_log if not log['valid']
            )
        }
```

**Default Parameters:**
```python
uniform_params = {
    'timing_range_hours': 2.0,   # ±2 hours
    'threshold_pct': 0.10         # ±10%
}
```

---

### 2. OU (Ornstein-Uhlenbeck) Policy (Week 3)

**Purpose:** Add autocorrelated, mean-reverting noise.

**Use Case:** Model realistic market microstructure with temporal dependence.

```python
class OUPolicy(RandomizationPolicy):
    """
    Adds Ornstein-Uhlenbeck process noise (mean-reverting).
    
    Parameters:
        theta (float): Mean reversion speed (higher = faster reversion)
        sigma (float): Volatility of the noise process
        mu (float): Long-term mean (typically 0)
    """
    
    def __init__(self, seed, params):
        super().__init__(seed, params)
        self.x = 0.0  # Current OU state
    
    def _update_ou_state(self, dt=1.0):
        # dx = theta * (mu - x) * dt + sigma * dW
        dx = (
            self.params['theta'] * (self.params['mu'] - self.x) * dt +
            self.params['sigma'] * np.sqrt(dt) * self.rng.randn()
        )
        self.x += dx
        return self.x
    
    def perturb_timing(self, timestamp, signal_strength=1.0):
        noise = self._update_ou_state()
        shift_hours = noise * 2.0  # Scale to reasonable hour range
        return timestamp + timedelta(hours=shift_hours)
    
    def get_diagnostics(self):
        return {
            'policy': 'OU',
            'theta': self.params['theta'],
            'sigma': self.params['sigma'],
            'autocorr_lag1': '(computed post-hoc)'
        }
```

**Default Parameters:**
```python
ou_params = {
    'theta': 0.15,   # Mean reversion speed
    'sigma': 0.05,   # Volatility
    'mu': 0.0        # Long-term mean
}
```

---

### 3. Pink Noise Policy (Week 3)

**Purpose:** Add 1/f noise with long-range dependence.

**Use Case:** Model persistent market effects and regime changes.

```python
class PinkNoisePolicy(RandomizationPolicy):
    """
    Adds pink noise (1/f^alpha) for long memory effects.
    
    Parameters:
        alpha (float): Spectral exponent (1.0 for pink noise)
        scale (float): Amplitude scaling factor
    """
    
    def __init__(self, seed, params):
        super().__init__(seed, params)
        # Pre-generate pink noise sequence using FFT method
        self.noise_buffer = self._generate_pink_noise(length=10000)
        self.noise_index = 0
    
    def _generate_pink_noise(self, length):
        # Generate via spectral method (1/f^alpha in frequency domain)
        freqs = np.fft.rfftfreq(length)[1:]  # Exclude DC
        spectrum = 1.0 / (freqs ** self.params['alpha'])
        
        phases = self.rng.uniform(0, 2*np.pi, len(spectrum))
        spectrum_complex = spectrum * np.exp(1j * phases)
        
        noise = np.fft.irfft(spectrum_complex, n=length)
        return noise * self.params['scale']
    
    def perturb_timing(self, timestamp, signal_strength=1.0):
        noise = self.noise_buffer[self.noise_index % len(self.noise_buffer)]
        self.noise_index += 1
        
        shift_hours = noise
        return timestamp + timedelta(hours=shift_hours)
    
    def get_diagnostics(self):
        return {
            'policy': 'Pink',
            'alpha': self.params['alpha'],
            'scale': self.params['scale'],
            'hurst_exponent': '(computed post-hoc)'
        }
```

**Default Parameters:**
```python
pink_params = {
    'alpha': 1.0,   # Spectral exponent (1.0 = pink)
    'scale': 0.08   # Amplitude scaling
}
```

---

## Integration Points

### With P2 (Baseline Strategy)

**P2 provides:**
- Signal timestamp
- Signal threshold (entry/exit price)
- Current positions

**P4 returns:**
- Perturbed timestamp
- Perturbed threshold
- Exposure validation status

**Example usage in P2's code:**
```python
# P2's baseline generates a signal
signal = {
    'timestamp': datetime(2025, 7, 15, 10, 30),
    'symbol': 'AAPL',
    'threshold': 150.00,
    'size': 100
}

# P2 calls P4's randomization
policy = UniformPolicy(seed=42, params=uniform_params)
perturbed_timestamp = policy.perturb_timing(signal['timestamp'])
perturbed_threshold = policy.perturb_threshold(signal['threshold'])

# P2 submits perturbed signal to execution
```

---

### With P3 (Backtesting Harness)

**P3 provides:**
- Config file specifying which policy to use
- Master seed for the run
- Current portfolio state

**P4 returns:**
- Perturbed signals
- Exposure compliance logs

**Example config (YAML):**
```yaml
randomization:
  policy: "Uniform"
  seed: 42
  params:
    timing_range_hours: 2.0
    threshold_pct: 0.10
```

**P3's runner code:**
```python
# P3 loads config and instantiates policy
config = load_config('config.yaml')
policy_class = {
    'Uniform': UniformPolicy,
    'OU': OUPolicy,
    'Pink': PinkNoisePolicy
}[config['randomization']['policy']]

policy = policy_class(
    seed=config['randomization']['seed'],
    params=config['randomization']['params']
)

# P3 runs backtest with policy
results = run_backtest(baseline_strategy, policy)
```

---

### With P7 (Adaptive Adversary)

**P7 provides:**
- Current adversary AUC score
- Direction to adjust ('increase' or 'decrease')

**P4 returns:**
- Updated policy parameters

**Example adaptive loop:**
```python
# P7's adaptive adversary training loop
for epoch in range(num_epochs):
    # Train adversary to predict trades
    auc_score = train_adversary(policy, data)
    
    # Adjust policy based on predictability
    if auc_score > 0.75:  # Too predictable
        policy.adjust_stochasticity(auc_score, direction='increase')
    elif auc_score < 0.55:  # Too random (hurting returns)
        policy.adjust_stochasticity(auc_score, direction='decrease')
    
    # Retrain with adjusted policy
```

---

## Seed Control & Reproducibility

### Seed Hierarchy

To enable reproducibility and seed variance analysis:

```python
def generate_policy_seed(master_seed, policy_name, date=None):
    """
    Generate deterministic seeds for different contexts.
    
    Args:
        master_seed (int): Top-level seed (e.g., 42)
        policy_name (str): 'Uniform', 'OU', or 'Pink'
        date (datetime, optional): For daily seeds
    
    Returns:
        int: Derived seed
    """
    seed_string = f"{master_seed}_{policy_name}"
    if date:
        seed_string += f"_{date.strftime('%Y%m%d')}"
    
    return hash(seed_string) % (2**31)  # Keep within int32 range
```

**Usage:**
```python
# Master seed for entire run
master_seed = 42

# Policy-specific seed
policy_seed = generate_policy_seed(master_seed, 'Uniform')
policy = UniformPolicy(seed=policy_seed, params=uniform_params)

# Daily seed for intraday randomness
daily_seed = generate_policy_seed(master_seed, 'Uniform', date=datetime(2025, 7, 15))
```

---

## Validation & Testing

### Unit Tests Required

```python
# tests/test_randomization.py

def test_seed_reproducibility():
    """Same seed produces identical perturbations."""
    policy1 = UniformPolicy(seed=42, params=uniform_params)
    policy2 = UniformPolicy(seed=42, params=uniform_params)
    
    ts = datetime(2025, 7, 15, 10, 30)
    assert policy1.perturb_timing(ts) == policy2.perturb_timing(ts)

def test_exposure_invariance():
    """Exposure stays within ±5% tolerance."""
    policy = UniformPolicy(seed=42, params=uniform_params)
    
    before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
    after = {'AAPL': 103, 'MSFT': -98}    # Net = 5
    
    assert policy.check_exposure_invariance(before, after) == True
    
def test_adjustment_api():
    """Adaptive adjustment modifies parameters."""
    policy = UniformPolicy(seed=42, params={'timing_range_hours': 2.0})
    
    original_range = policy.params['timing_range_hours']
    policy.adjust_stochasticity(auc_score=0.85, direction='increase')
    
    assert policy.params['timing_range_hours'] > original_range
```

---

## Deliverables Checklist

- [ ] `base_policy.py` - Abstract `RandomizationPolicy` class
- [ ] `uniform_policy.py` - Uniform implementation (Week 2)
- [ ] `ou_policy.py` - OU implementation (Week 3)
- [ ] `pink_policy.py` - Pink noise implementation (Week 3)
- [ ] `utils.py` - Seed generation, exposure checks
- [ ] `tests/test_randomization.py` - Unit tests
- [ ] `examples/demo_uniform.ipynb` - Usage example
- [ ] `README.md` - Installation and quick start
- [ ] This API specification document

---

## Open Questions for Team Review

1. **Timing constraints:** Should perturbations respect market hours (9:30 AM - 4:00 PM)?
2. **Position limits:** How to handle perturbations that violate single-name cap (≤1% NAV)?
3. **Adjustment strategy:** Should P7's adaptive adjustments be linear (×1.2) or exponential?
4. **Diagnostic frequency:** Log every perturbation or aggregate by day?

---

## Contact & Feedback

**Owner:** P4  
**Review by:** P2 (baseline integration), P3 (config/logging), P7 (adaptive API)  
**Due date:** End of Week 1 (November 9, 2025)  
**Status:** ✅ Completed

Please provide feedback via GitHub issues or in the weekly sync meeting.