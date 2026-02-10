"""Smart adaptive grid strategy (AI-guided by market regime + volatility feedback)."""

from dataclasses import dataclass
from decimal import Decimal
from statistics import pstdev
from typing import Any

from packages.core.market_regime import MarketRegimeDetector
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
    recenter_mode: str = "aggressive"
    regime_adaptation_enabled: bool = False

    def __post_init__(self) -> None:
        self._regime_detector = MarketRegimeDetector()

    def _calculate_dynamic_spacing(self, volatility_percentile: float) -> tuple[int, int]:
        """Adapt spacing guardrails using volatility percentile regime."""
        if volatility_percentile < 0.30:
            return max(10, self.min_spacing_bps - 10), max(self.min_spacing_bps + 20, self.max_spacing_bps - 70)
        if volatility_percentile < 0.70:
            return self.min_spacing_bps, self.max_spacing_bps
        return max(20, self.min_spacing_bps + 15), min(800, self.max_spacing_bps + 130)

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
        recenter_mode = self.recenter_mode.strip().lower()
        if recenter_mode not in {"conservative", "aggressive"}:
            recenter_mode = "aggressive"

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

        regime_analysis = self._regime_detector.detect_regime(
            closes=closes_1h,
            volumes=[c.volume for c in context.candles_1h],
            highs=highs_1h,
            lows=lows_1h,
        )
        dynamic_min_spacing, dynamic_max_spacing = self._calculate_dynamic_spacing(
            regime_analysis.volatility_percentile
        )
        adaptation = self._regime_detector.recommended_adaptation(
            analysis=regime_analysis,
            base_grid_levels=self.grid_levels,
            base_take_profit_buffer=self.take_profit_buffer,
            base_stop_loss_buffer=self.stop_loss_buffer,
        )
        effective_min_spacing = dynamic_min_spacing
        effective_max_spacing = dynamic_max_spacing
        if self.regime_adaptation_enabled:
            effective_min_spacing = int(max(10, effective_min_spacing * adaptation.spacing_multiplier))
            effective_max_spacing = int(max(effective_min_spacing + 10, effective_max_spacing * adaptation.spacing_multiplier))

        raw_spacing = (atr_bps * Decimal(str(self.volatility_blend))) + (vol_bps * Decimal("0.50"))
        spacing_bps = int(
            max(
                Decimal(effective_min_spacing),
                min(Decimal(effective_max_spacing), raw_spacing),
            )
        )
        spacing_bps = max(spacing_bps, 1)

        center_period = min(55, max(20, self.lookback_1h // 2))
        center_ema = _ema(closes_1h[-self.lookback_1h :], center_period)
        regime_fast = _ema(closes_4h, self.regime_fast_period_4h)
        regime_slow = _ema(closes_4h, self.regime_slow_period_4h)
        regime_strength = (regime_fast - regime_slow) / regime_slow

        tilt_multiplier = Decimal("1") + (regime_strength * Decimal(str(self.trend_tilt)))
        if self.regime_adaptation_enabled:
            tilt_multiplier *= Decimal(str(adaptation.tilt_multiplier))
        tilted_center = center_ema * tilt_multiplier

        min_step = tilted_center * Decimal(effective_min_spacing) / Decimal("10000")
        max_step = tilted_center * Decimal(effective_max_spacing) / Decimal("10000")
        geometric_step = tilted_center * Decimal(spacing_bps) / Decimal("10000")
        vol_abs = tilted_center * vol_bps / Decimal("10000")
        arithmetic_raw_step = (atr_value * Decimal(str(self.volatility_blend))) + (vol_abs * Decimal("0.50"))
        arithmetic_step = max(min_step, min(max_step, arithmetic_raw_step))
        step = geometric_step if spacing_mode == "geometric" else arithmetic_step
        if step <= 0:
            return Signal(side="HOLD", confidence=0.0, reason="invalid_grid_step", indicators={})

        effective_grid_levels = adaptation.grid_levels if self.regime_adaptation_enabled else self.grid_levels
        levels_each_side = max(1, effective_grid_levels // 2)
        half_levels = Decimal(str(levels_each_side))
        upper_band = tilted_center + (step * half_levels)
        lower_band = tilted_center - (step * half_levels)
        buy_trigger = tilted_center - step
        sell_trigger = tilted_center + step

        distance_to_center = abs(last_close - tilted_center)
        max_distance = max(step * half_levels, Decimal("0.00000001"))
        confidence = float(min(Decimal("0.95"), distance_to_center / max_distance))

        indicators: dict[str, Any] = {
            "grid_center": float(tilted_center),
            "grid_upper": float(upper_band),
            "grid_lower": float(lower_band),
            "grid_step": float(step),
            "grid_levels": float(effective_grid_levels),
            "spacing_bps": float(spacing_bps),
            "spacing_mode_geometric": 1.0 if spacing_mode == "geometric" else 0.0,
            "atr_bps": float(atr_bps),
            "vol_bps": float(vol_bps),
            "volatility_percentile": float(regime_analysis.volatility_percentile),
            "regime_code": float(regime_analysis.regime.code),
            "regime_confidence": float(regime_analysis.confidence),
            "regime_trend_strength": float(regime_analysis.trend_strength),
            "regime_breakout_probability": float(regime_analysis.breakout_probability),
            "position_size_multiplier": float(adaptation.position_size_multiplier),
            "effective_min_spacing_bps": float(effective_min_spacing),
            "effective_max_spacing_bps": float(effective_max_spacing),
            "regime_strength": float(regime_strength),
            "last_close": float(last_close),
            "prev_close": float(prev_close),
            "buy_trigger": float(buy_trigger),
            "sell_trigger": float(sell_trigger),
            "recenter_mode_aggressive": 1.0 if recenter_mode == "aggressive" else 0.0,
        }
        for level in range(1, levels_each_side + 1):
            indicators[f"grid_buy_level_{level}"] = float(tilted_center - (step * Decimal(level)))
            indicators[f"grid_sell_level_{level}"] = float(tilted_center + (step * Decimal(level)))

        effective_take_profit_buffer = (
            adaptation.take_profit_buffer if self.regime_adaptation_enabled else self.take_profit_buffer
        )
        effective_stop_loss_buffer = (
            adaptation.stop_loss_buffer if self.regime_adaptation_enabled else self.stop_loss_buffer
        )
        if effective_take_profit_buffer > 0:
            take_profit_trigger = upper_band * (Decimal("1") + Decimal(str(effective_take_profit_buffer)))
            indicators["take_profit_trigger"] = float(take_profit_trigger)
            if last_close >= take_profit_trigger:
                return Signal(
                    side="SELL",
                    confidence=max(confidence, 0.7),
                    reason="grid_take_profit_buffer_hit",
                    indicators=indicators,
                )
        if effective_stop_loss_buffer > 0:
            stop_loss_trigger = lower_band * (Decimal("1") - Decimal(str(effective_stop_loss_buffer)))
            indicators["stop_loss_trigger"] = float(stop_loss_trigger)
            if last_close <= stop_loss_trigger:
                return Signal(
                    side="SELL",
                    confidence=max(confidence, 0.8),
                    reason="grid_stop_loss_buffer_hit",
                    indicators=indicators,
                )

        if last_close < lower_band or last_close > upper_band:
            # Auto-recenter: when price escapes the active band, anchor a new band to
            # current price so the next cycles can continue operating without deadlock.
            recentered_center = last_close
            recentered_upper = recentered_center + (step * half_levels)
            recentered_lower = recentered_center - (step * half_levels)
            recentered_buy_trigger = recentered_center - step
            recentered_sell_trigger = recentered_center + step
            indicators.update(
                {
                    "recentered": 1.0,
                    "recenter_from_center": float(tilted_center),
                    "recenter_from_upper": float(upper_band),
                    "recenter_from_lower": float(lower_band),
                    "grid_center": float(recentered_center),
                    "grid_upper": float(recentered_upper),
                    "grid_lower": float(recentered_lower),
                    "buy_trigger": float(recentered_buy_trigger),
                    "sell_trigger": float(recentered_sell_trigger),
                }
            )
            if recenter_mode == "aggressive":
                broke_out_up = prev_close <= upper_band and last_close > upper_band
                broke_out_down = prev_close >= lower_band and last_close < lower_band
                if broke_out_up:
                    return Signal(
                        side="BUY",
                        confidence=max(confidence, 0.65),
                        reason="grid_recentered_auto_breakout_buy",
                        indicators=indicators,
                    )
                if broke_out_down:
                    return Signal(
                        side="SELL",
                        confidence=max(confidence, 0.65),
                        reason="grid_recentered_auto_breakdown_sell",
                        indicators=indicators,
                    )
            return Signal(side="HOLD", confidence=confidence, reason="grid_recentered_auto", indicators=indicators)

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

        return Signal(side="HOLD", confidence=0.25, reason="grid_wait_inside_band", indicators=indicators)


registry.register("smart_grid_ai", SmartAdaptiveGridStrategy)
