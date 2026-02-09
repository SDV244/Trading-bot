"""Smart adaptive grid strategy (AI-guided by market regime + volatility feedback)."""

from dataclasses import dataclass
from decimal import Decimal
from statistics import pstdev

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


def _atr(highs: list[Decimal], lows: list[Decimal], closes: list[Decimal], period: int) -> Decimal:
    if len(closes) < period + 1:
        raise ValueError("not enough values to compute ATR")
    true_ranges: list[Decimal] = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / Decimal(period)


@dataclass(slots=True)
class SmartAdaptiveGridStrategy(Strategy):
    """Adaptive grid strategy with trend-tilt and volatility-aware spacing."""

    name: str = "smart_grid_ai"
    lookback_1h: int = 120
    atr_period_1h: int = 14
    grid_levels: int = 6
    spacing_mode: str = "geometric"
    min_spacing_bps: int = 25
    max_spacing_bps: int = 220
    trend_tilt: float = 1.25
    volatility_blend: float = 0.7
    take_profit_buffer: float = 0.02
    stop_loss_buffer: float = 0.05
    regime_fast_period_4h: int = 21
    regime_slow_period_4h: int = 55

    def data_requirements(self) -> dict[str, int]:
        return {
            "1h": max(self.lookback_1h, self.atr_period_1h + 2),
            "4h": max(self.regime_fast_period_4h, self.regime_slow_period_4h),
        }

    def generate_signal(self, context: StrategyContext) -> Signal:
        spacing_mode = self.spacing_mode.strip().lower()
        if spacing_mode not in {"geometric", "arithmetic"}:
            return Signal(
                side="HOLD",
                confidence=0.0,
                reason="invalid_grid_spacing_mode",
                indicators={},
            )

        closes_1h = [c.close for c in context.candles_1h]
        highs_1h = [c.high for c in context.candles_1h]
        lows_1h = [c.low for c in context.candles_1h]
        closes_4h = [c.close for c in context.candles_4h]

        req = self.data_requirements()
        if len(closes_1h) < req["1h"] or len(closes_4h) < req["4h"]:
            return Signal(
                side="HOLD",
                confidence=0.0,
                reason="insufficient_data",
                indicators={
                    "required_1h": float(req["1h"]),
                    "required_4h": float(req["4h"]),
                    "available_1h": float(len(closes_1h)),
                    "available_4h": float(len(closes_4h)),
                },
            )

        last_close = closes_1h[-1]
        prev_close = closes_1h[-2]
        atr_value = _atr(highs_1h, lows_1h, closes_1h, self.atr_period_1h)
        atr_bps = (atr_value / last_close) * Decimal("10000")

        returns = []
        for i in range(1, len(closes_1h)):
            prev = closes_1h[i - 1]
            curr = closes_1h[i]
            returns.append(float((curr - prev) / prev) if prev != 0 else 0.0)
        vol_window = max(self.atr_period_1h * 3, 30)
        vol = pstdev(returns[-vol_window:]) if len(returns) >= 2 else 0.0
        vol_bps = Decimal(str(vol * 10000))

        raw_spacing = (atr_bps * Decimal(str(self.volatility_blend))) + (vol_bps * Decimal("0.50"))
        spacing_bps = int(
            max(
                Decimal(self.min_spacing_bps),
                min(Decimal(self.max_spacing_bps), raw_spacing),
            )
        )
        spacing_bps = max(spacing_bps, 1)

        center_period = min(55, max(20, self.lookback_1h // 2))
        center_ema = _ema(closes_1h[-self.lookback_1h :], center_period)
        regime_fast = _ema(closes_4h, self.regime_fast_period_4h)
        regime_slow = _ema(closes_4h, self.regime_slow_period_4h)
        regime_strength = (regime_fast - regime_slow) / regime_slow

        tilt_multiplier = Decimal("1") + (regime_strength * Decimal(str(self.trend_tilt)))
        tilted_center = center_ema * tilt_multiplier

        min_step = tilted_center * Decimal(self.min_spacing_bps) / Decimal("10000")
        max_step = tilted_center * Decimal(self.max_spacing_bps) / Decimal("10000")
        geometric_step = tilted_center * Decimal(spacing_bps) / Decimal("10000")
        vol_abs = tilted_center * vol_bps / Decimal("10000")
        arithmetic_raw_step = (atr_value * Decimal(str(self.volatility_blend))) + (vol_abs * Decimal("0.50"))
        arithmetic_step = max(min_step, min(max_step, arithmetic_raw_step))
        step = geometric_step if spacing_mode == "geometric" else arithmetic_step
        if step <= 0:
            return Signal(side="HOLD", confidence=0.0, reason="invalid_grid_step", indicators={})

        half_levels = Decimal(str(max(1, self.grid_levels // 2)))
        upper_band = tilted_center + (step * half_levels)
        lower_band = tilted_center - (step * half_levels)
        buy_trigger = tilted_center - step
        sell_trigger = tilted_center + step

        distance_to_center = abs(last_close - tilted_center)
        max_distance = max(step * half_levels, Decimal("0.00000001"))
        confidence = float(min(Decimal("0.95"), distance_to_center / max_distance))

        indicators = {
            "grid_center": float(tilted_center),
            "grid_upper": float(upper_band),
            "grid_lower": float(lower_band),
            "grid_step": float(step),
            "grid_levels": float(self.grid_levels),
            "spacing_bps": float(spacing_bps),
            "spacing_mode_geometric": 1.0 if spacing_mode == "geometric" else 0.0,
            "atr_bps": float(atr_bps),
            "vol_bps": float(vol_bps),
            "regime_strength": float(regime_strength),
            "last_close": float(last_close),
            "prev_close": float(prev_close),
            "buy_trigger": float(buy_trigger),
            "sell_trigger": float(sell_trigger),
        }

        if self.take_profit_buffer > 0:
            take_profit_trigger = upper_band * (Decimal("1") + Decimal(str(self.take_profit_buffer)))
            indicators["take_profit_trigger"] = float(take_profit_trigger)
            if last_close >= take_profit_trigger:
                return Signal(
                    side="SELL",
                    confidence=max(confidence, 0.7),
                    reason="grid_take_profit_buffer_hit",
                    indicators=indicators,
                )
        if self.stop_loss_buffer > 0:
            stop_loss_trigger = lower_band * (Decimal("1") - Decimal(str(self.stop_loss_buffer)))
            indicators["stop_loss_trigger"] = float(stop_loss_trigger)
            if last_close <= stop_loss_trigger:
                return Signal(
                    side="SELL",
                    confidence=max(confidence, 0.8),
                    reason="grid_stop_loss_buffer_hit",
                    indicators=indicators,
                )

        # Grid rebalance entry/exit: react when price crosses next level.
        if prev_close > buy_trigger and last_close <= buy_trigger:
            return Signal(
                side="BUY",
                confidence=max(confidence, 0.55),
                reason="grid_buy_rebalance",
                indicators=indicators,
            )
        if prev_close < sell_trigger and last_close >= sell_trigger:
            return Signal(
                side="SELL",
                confidence=max(confidence, 0.55),
                reason="grid_sell_rebalance",
                indicators=indicators,
            )

        if last_close < lower_band or last_close > upper_band:
            return Signal(side="HOLD", confidence=confidence, reason="grid_recenter_wait", indicators=indicators)

        return Signal(side="HOLD", confidence=0.25, reason="grid_wait_inside_band", indicators=indicators)


registry.register("smart_grid_ai", SmartAdaptiveGridStrategy)
