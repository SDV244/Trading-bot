"""Breakout strategy using 4h range and 1h confirmation."""

from dataclasses import dataclass
from decimal import Decimal

from packages.core.strategies.base import Signal, Strategy, StrategyContext, registry


@dataclass(slots=True)
class BreakoutStrategy(Strategy):
    """4h range breakout confirmed on 1h closes."""

    name: str = "breakout_4h_1h"
    lookback_4h: int = 20
    confirmation_bars_1h: int = 2
    breakout_buffer_bps: int = 10

    def data_requirements(self) -> dict[str, int]:
        return {"1h": self.confirmation_bars_1h, "4h": self.lookback_4h}

    def generate_signal(self, context: StrategyContext) -> Signal:
        if len(context.candles_4h) < self.lookback_4h or len(context.candles_1h) < self.confirmation_bars_1h:
            return Signal(side="HOLD", confidence=0.0, reason="insufficient_data", indicators={})

        recent_4h = context.candles_4h[-self.lookback_4h :]
        range_high = max(c.high for c in recent_4h)
        range_low = min(c.low for c in recent_4h)
        buffer = Decimal(self.breakout_buffer_bps) / Decimal("10000")

        confirm_closes = [c.close for c in context.candles_1h[-self.confirmation_bars_1h :]]
        high_trigger = range_high * (Decimal("1") + buffer)
        low_trigger = range_low * (Decimal("1") - buffer)

        indicators = {
            "range_high": float(range_high),
            "range_low": float(range_low),
            "high_trigger": float(high_trigger),
            "low_trigger": float(low_trigger),
            "confirm_bars": float(self.confirmation_bars_1h),
        }

        if all(price >= high_trigger for price in confirm_closes):
            return Signal(side="BUY", confidence=0.75, reason="range_breakout_up", indicators=indicators)
        if all(price <= low_trigger for price in confirm_closes):
            return Signal(side="SELL", confidence=0.75, reason="range_breakout_down", indicators=indicators)
        return Signal(side="HOLD", confidence=0.2, reason="inside_range", indicators=indicators)


registry.register("breakout_4h_1h", BreakoutStrategy)
