"""Enhanced smart-grid strategy profile with stronger regime adaptation."""

from dataclasses import dataclass

from packages.core.strategies.base import registry
from packages.core.strategies.smart_grid import SmartAdaptiveGridStrategy


@dataclass(slots=True)
class EnhancedSmartGridStrategy(SmartAdaptiveGridStrategy):
    """
    Tuned smart-grid profile for adaptive production paper/live testing.

    Uses the same logic as SmartAdaptiveGridStrategy but with defaults that
    emphasize regime adaptation and execution safety.
    """

    name: str = "enhanced_smart_grid"
    lookback_1h: int = 160
    atr_period_1h: int = 21
    grid_levels: int = 7
    min_spacing_bps: int = 30
    max_spacing_bps: int = 300
    trend_tilt: float = 1.4
    volatility_blend: float = 0.9
    take_profit_buffer: float = 0.03
    stop_loss_buffer: float = 0.06
    regime_adaptation_enabled: bool = True


registry.register("enhanced_smart_grid", EnhancedSmartGridStrategy)

