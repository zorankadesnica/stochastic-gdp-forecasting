"""
Stochastic Optimization for Economic Forecasting

A Python package for robust GDP nowcasting using stochastic optimization,
Bayesian inference, and Monte Carlo simulation.
"""

from .data_loader import (
    SORSDataLoader,
    NBSDataLoader, 
    FREDDataLoader,
    WorldBankDataLoader,
    load_serbian_indicators
)
from .composite_index import CompositeIndex, SerbianCompositeIndex
from .stochastic_opt import SAAOptimizer, RobustOptimizer, AdaptiveSGDOptimizer
from .monte_carlo import MonteCarloSimulator

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'SORSDataLoader',
    'NBSDataLoader',
    'FREDDataLoader', 
    'WorldBankDataLoader',
    'load_serbian_indicators',
    'CompositeIndex',
    'SerbianCompositeIndex',
    'SAAOptimizer',
    'RobustOptimizer',
    'AdaptiveSGDOptimizer',
    'MonteCarloSimulator',
]
