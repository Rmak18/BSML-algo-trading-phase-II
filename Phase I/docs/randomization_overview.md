# P4 - Randomization Modules

**Owner:** Neel  
**Project:** BSML - Randomized Execution Research  
**Status:** Week 1 Complete | Week 2 In Progress

---

## What This Module Does

Adds controlled randomness to trade execution timing and thresholds while keeping portfolio exposure neutral (~0% net exposure).

**Three randomization policies:**
1. **Uniform** - IID uniform noise (Week 2)
2. **OU** - Mean-reverting autocorrelated noise (Week 3)
3. **Pink** - 1/f noise with long memory (Week 3)

---

## Quick Start

```python
from randomization import UniformPolicy

# Create policy with seed for reproducibility
policy = UniformPolicy(
    seed=42,
    params={
        'timing_range_hours': 2.0,  # ±2 hours
        'threshold_pct': 0.10        # ±10%
    }
)

# Perturb a trade signal
from datetime import datetime

original_time = datetime(2025, 7, 15, 10, 30)
original_threshold = 150.00

perturbed_time = policy.perturb_timing(original_time)
perturbed_threshold = policy.perturb_threshold(original_threshold)

print(f"Original: {original_time}, ${original_threshold}")
print(f"Perturbed: {perturbed_time}, ${perturbed_threshold:.2f}")
```

**Output:**
```
Original: 2025-07-15 10:30:00, $150.00
Perturbed: 2025-07-15 11:47:00, $151.50
```

---

## Files in This Module

```
randomization/
├── README.md                    # This file
├── API_SPEC.md                  # Full technical specification
├── base_policy.py               # Abstract base class
├── uniform_policy.py            # Week 2 (IN PROGRESS)
├── ou_policy.py                 # Week 3 (TODO)
├── pink_policy.py               # Week 3 (TODO)
├── utils.py                     # Seed generation, exposure checks
├── tests/
│   └── test_randomization.py   # Unit tests
└── examples/
    └── demo_uniform.ipynb       # Interactive demo
```

---

## Installation

```bash
# Clone the repo
git clone <repo_url>
cd randomization

# Install dependencies
pip install numpy pandas matplotlib pytest

# Run tests
pytest tests/
```

---

## Usage by Other Team Members

### For P2 (Baseline Strategy)
```python
# In your baseline strategy code:
signal = generate_signal()  # Your existing code

# Add randomization
policy = UniformPolicy(seed=42, params=uniform_params)
perturbed_signal = {
    'time': policy.perturb_timing(signal['time']),
    'threshold': policy.perturb_threshold(signal['threshold']),
    'size': signal['size']  # Not randomized
}
```

### For P3 (Backtesting Harness)
```yaml
# In your config.yaml:
randomization:
  policy: "Uniform"
  seed: 42
  params:
    timing_range_hours: 2.0
    threshold_pct: 0.10
```

### For P7 (Adaptive Adversary)
```python
# In your adaptive training loop:
if adversary_auc > 0.75:
    policy.adjust_stochasticity(adversary_auc, direction='increase')
```

---

## Development Timeline

- ✅ **Week 1 (Nov 3-9):** API specification
- ⏳ **Week 2 (Nov 10-16):** Uniform policy + exposure checks
- 📅 **Week 3 (Nov 17-23):** OU and Pink policies
- 📅 **Week 4 (Nov 24-30):** Ablations and final validation

---

## Key Constraints

1. **Exposure Invariance:** Net exposure must stay within ±5% of target
2. **Reproducibility:** Same seed → identical perturbations
3. **Market Hours:** (TBD) Should timing respect 9:30 AM - 4:00 PM?
4. **Position Limits:** Single-name cap ≤ 1% NAV

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_randomization.py::test_seed_reproducibility

# Run with coverage
pytest --cov=randomization tests/
```

---

## Documentation

- **API Specification:** See `API_SPEC.md` for full technical details
- **Examples:** See `examples/demo_uniform.ipynb` for interactive walkthrough
- **Team Coordination:** See main project timeline PDF for integration points

---

## Questions or Issues?

**Owner:** P4  
**Integration help:** P3 (infrastructure)  
**Finance questions:** P1, P2  
**Adaptive adversary:** P7

File issues on GitHub or bring to weekly sync meetings.