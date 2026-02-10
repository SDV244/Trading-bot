"""Base strategy interfaces and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

SignalSide = Literal["BUY", "SELL", "HOLD"]


@dataclass(slots=True, frozen=True)
class Signal:
    """Standard strategy output schema."""

    side: SignalSide
    confidence: float
    reason: str
    indicators: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CandleInput:
    """Minimal candle shape required by strategies."""

    open_time: datetime
    close: Decimal
    high: Decimal
    low: Decimal
    volume: Decimal


@dataclass(slots=True, frozen=True)
class StrategyContext:
    """Market context passed to strategy implementations."""

    symbol: str
    candles_1h: list[CandleInput]
    candles_4h: list[CandleInput]
    now: datetime


class Strategy(ABC):
    """Strategy interface."""

    name: str

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal:
        """Generate a signal from the provided market context."""

    def data_requirements(self) -> dict[str, int]:
        """Minimum candles required per timeframe."""
        return {}


class StrategyRegistry:
    """Registry for strategy plugins."""

    def __init__(self) -> None:
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, name: str, strategy_class: type[Strategy]) -> None:
        """Register a strategy class by name."""
        self._strategies[name] = strategy_class

    def create(self, name: str, **kwargs: object) -> Strategy:
        """Create a strategy instance by name."""
        strategy_class = self._strategies.get(name)
        if strategy_class is None:
            available = ", ".join(sorted(self._strategies.keys()))
            msg = f"Unknown strategy '{name}'. Available: {available}"
            raise ValueError(msg)
        return strategy_class(**kwargs)

    def list_names(self) -> list[str]:
        """List registered strategy names."""
        return sorted(self._strategies.keys())


registry = StrategyRegistry()
