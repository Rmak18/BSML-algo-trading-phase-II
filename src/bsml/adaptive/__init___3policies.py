"""
P7 Adaptive Adversary Module - 3-Policy Comparison

Task: Compare predictability across Uniform, Pink Noise, and OU randomization.
Goal: Identify which randomization strategy best evades adversarial detection.

Owner: P7
Week: 3
"""

from .adaptive_loop_3policies import (
    main,
    adaptive_three_policy_loop,
    compare_three_policies,
    ThreePolicyConfig
)
from .bridge_3policies import (
    prepare_three_policy_data,
    time_split_data
)
from .next_trade_predictor import NextTradePredictor

__all__ = [
    # Main functions
    'main',
    'adaptive_three_policy_loop',
    'compare_three_policies',
    
    # Configuration
    'ThreePolicyConfig',
    
    # Data preparation
    'prepare_three_policy_data',
    'time_split_data',
    
    # Predictor
    'NextTradePredictor',
]

__version__ = '3.1.0'
__author__ = 'P7'
__description__ = '3-Policy Comparison: Finding the best randomization strategy'
