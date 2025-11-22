"""
P7 Adaptive Adversary Module V2
Task: Policy Distinguishability - Can adversary distinguish Baseline vs Uniform?

V2 Improvements:
- Focused prediction task (policy classification, not next-day prediction)
- Better feature engineering (execution characteristics, timing patterns)
- Inverted feedback logic (lower AUC = better randomization)
- Simplified classifier (single strong GB model)
- Feature importance tracking

Owner: P7
Week: 3
"""

# Import V1 (legacy - for backward compatibility)
from .adaptive_loop_v1 import main as main_v1
from .adaptive_loop_v1 import adaptive_training_loop as adaptive_training_loop_v1

# Import V2 (current - recommended)
try:
    from .adaptive_loop_v2 import main as main_v2
    from .adaptive_loop_v2 import adaptive_training_loop_v2
    from .bridge_v2 import prepare_adversary_data_v2, time_split_data
    from .adversary_classifier_v2 import P7AdversaryV2
    
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    main_v2 = None
    adaptive_training_loop_v2 = None

# Default to V2 if available
if HAS_V2:
    main = main_v2
    adaptive_training_loop = adaptive_training_loop_v2
else:
    main = main_v1
    adaptive_training_loop = adaptive_training_loop_v1

__all__ = [
    # V2 (recommended)
    'main',
    'adaptive_training_loop',
    'main_v2',
    'adaptive_training_loop_v2',
    'P7AdversaryV2',
    'prepare_adversary_data_v2',
    
    # V1 (legacy)
    'main_v1',
    'adaptive_training_loop_v1',
]
