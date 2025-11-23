"""
P7 Adaptive Adversary Module - Next-Trade Prediction

Task: Given recent trade history, predict if there will be a trade tomorrow.
Goal: Compare predictability between Baseline vs Uniform policies.
Lower AUC = Less predictable = Better randomization

Owner: P7
Week: 3
"""

from .adaptive_loop_next_trade import (
    main,
    adaptive_next_trade_loop,
    run_next_trade_experiment,
    NextTradePredictionConfig
)
from .bridge_next_trade import (
    prepare_next_trade_data,
    create_next_trade_dataset,
    time_split_data
)
from .next_trade_predictor import NextTradePredictor

__all__ = [
    # Main functions
    'main',
    'adaptive_next_trade_loop',
    'run_next_trade_experiment',
    
    # Configuration
    'NextTradePredictionConfig',
    
    # Data preparation
    'prepare_next_trade_data',
    'create_next_trade_dataset',
    'time_split_data',
    
    # Predictor
    'NextTradePredictor',
]

__version__ = '3.0.0'
__author__ = 'P7'
__description__ = 'Next-Trade Prediction: Measuring trading pattern predictability'
