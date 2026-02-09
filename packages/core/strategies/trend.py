"""EMA trend strategy (4h regime + 1h timing)."""

from dataclasses import dataclass
from decimal import Decimal

from packages.core.strategies.base import Signal, Strategy, StrategyContext, registry


def _ema(values: list[Decimal], period: int) -> Decimal:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        raise ValueError("not enough values to compute EMA")

    multiplier = Decimal("2") / Decimal(period + 1)
    ema_value = sum(values[:period]) / Decimal(period)
    for value in values[period:]:
        ema_value = (value - ema_value) * multiplier + ema_value
    return ema_value


@dataclass(slots=True)
class EMATrendStrategy(Strategy):
    """Trend following strategy with multi-timeframe confirmation."""

    name: str = "trend_ema"
    regime_fast_period_4h: int = 20
    regime_slow_period_4h: int = 50
    timing_period_1h: int = 20
    pullback_threshold_bps: int = 30
    min_regime_strength_bps: int = 0

    def data_requirements(self) -> dict[str, int]:
        return {
            "1h": self.timing_period_1h,
            "4h": max(self.regime_fast_period_4h, self.regime_slow_period_4h),
        }

    def generate_signal(self, context: StrategyContext) -> Signal:
        closes_4h = [c.close for c in context.candles_4h]
        closes_1h = [c.close for c in context.candles_1h]

        min_4h = max(self.regime_fast_period_4h, self.regime_slow_period_4h)
        if len(closes_4h) < min_4h or len(closes_1h) < self.timing_period_1h:
            return Signal(
                side="HOLD",
                confidence=0.0,
                reason="insufficient_data",
                indicators={
                    "required_4h": float(min_4h),
                    "required_1h": float(self.timing_period_1h),
                    "available_4h": float(len(closes_4h)),
                    "available_1h": float(len(closes_1h)),
                },
            )

        fast_ema_4h = _ema(closes_4h, self.regime_fast_period_4h)
        slow_ema_4h = _ema(closes_4h, self.regime_slow_period_4h)
        timing_ema_1h = _ema(closes_1h, self.timing_period_1h)
        last_close_1h = closes_1h[-1]
        threshold = Decimal(self.pullback_threshold_bps) / Decimal("10000")

        regime_strength = abs((fast_ema_4h - slow_ema_4h) / slow_ema_4h)
        confidence = float(min(Decimal("1"), regime_strength * Decimal("20")))

        indicators = {
            "ema_fast_4h": float(fast_ema_4h),
            "ema_slow_4h": float(slow_ema_4h),
            "ema_timing_1h": float(timing_ema_1h),
            "last_close_1h": float(last_close_1h),
            "regime_strength": float(regime_strength),
            "regime_strength_bps": float(regime_strength * Decimal("10000")),
        }

        if regime_strength * Decimal("10000") < Decimal(self.min_regime_strength_bps):
            return Signal("HOLD", confidence=confidence, reason="weak_regime", indicators=indicators)

        if fast_ema_4h > slow_ema_4h:
            # Bullish regime: only buy when 1h price is near/above timing EMA.
            lower_bound = timing_ema_1h * (Decimal("1") - threshold)
            if last_close_1h >= lower_bound:
                return Signal("BUY", confidence=max(confidence, 0.55), reason="bullish_regime_entry", indicators=indicators)
            return Signal("HOLD", confidence=confidence, reason="bullish_wait_pullback", indicators=indicators)

        if fast_ema_4h < slow_ema_4h:
            # Bearish regime: only sell when 1h price is near/below timing EMA.
            upper_bound = timing_ema_1h * (Decimal("1") + threshold)
            if last_close_1h <= upper_bound:
                return Signal("SELL", confidence=max(confidence, 0.55), reason="bearish_regime_entry", indicators=indicators)
            return Signal("HOLD", confidence=confidence, reason="bearish_wait_retest", indicators=indicators)

        return Signal("HOLD", confidence=0.0, reason="neutral_regime", indicators=indicators)


@dataclass(slots=True)
class EMATrendFastStrategy(EMATrendStrategy):
    """Faster trend profile for BTCUSDT with weak-regime filter."""

    name: str = "trend_ema_fast"
    regime_fast_period_4h: int = 13
    regime_slow_period_4h: int = 34
    timing_period_1h: int = 13
    pullback_threshold_bps: int = 30
    min_regime_strength_bps: int = 15


registry.register("trend_ema", EMATrendStrategy)
registry.register("trend_ema_fast", EMATrendFastStrategy)
