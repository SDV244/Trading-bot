"""Strategy implementations and registry."""

from packages.core.strategies.base import CandleInput, Signal, Strategy, StrategyContext, registry
from packages.core.strategies.breakout import BreakoutStrategy
from packages.core.strategies.mean_reversion import MeanReversionStrategy
from packages.core.strategies.trend import EMATrendStrategy

__all__ = [
    "BreakoutStrategy",
    "CandleInput",
    "EMATrendStrategy",
    "MeanReversionStrategy",
    "Signal",
    "Strategy",
    "StrategyContext",
    "registry",
]
