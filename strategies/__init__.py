"""
Strategies Package

Contains trading strategy implementations for the adaptive bot.
"""

from .consolidation_strategy import ConsolidationStrategy, ConsolidationSignal, SignalType
from .trend_strategy import TrendStrategy, TrendSignal, TrendSignalType

__all__ = [
    'ConsolidationStrategy', 
    'ConsolidationSignal', 
    'SignalType',
    'TrendStrategy', 
    'TrendSignal', 
    'TrendSignalType'
]