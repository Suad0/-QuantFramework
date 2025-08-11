"""
Strategy framework infrastructure components.

This module provides the infrastructure layer implementations for the strategy framework,
including strategy registry, signal aggregation, parameter optimization, and composition.
"""

from .registry import StrategyRegistry
from .aggregator import SignalAggregator
from .optimizer import StrategyOptimizer
from .validator import SignalValidator
from .composer import StrategyComposer

__all__ = [
    'StrategyRegistry',
    'SignalAggregator', 
    'StrategyOptimizer',
    'SignalValidator',
    'StrategyComposer'
]