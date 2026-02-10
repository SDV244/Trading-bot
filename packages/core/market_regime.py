"""Market regime detection utilities for adaptive strategy/risk behavior."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from statistics import pstdev

import numpy as np


class MarketRegime(str, Enum):
    """Supported market regimes for strategy/risk adaptation."""

    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_TIGHT = "ranging_tight"
    RANGING_WIDE = "ranging_wide"
    BREAKOUT_PENDING = "breakout_pending"
    VOLATILE_CHAOS = "volatile_chaos"
    UNKNOWN = "unknown"

    @property
    def code(self) -> int:
        """Stable numeric code for serialization in numeric indicator payloads."""
        return {
            MarketRegime.UNKNOWN: 0,
            MarketRegime.TRENDING_BULLISH: 1,
            MarketRegime.TRENDING_BEARISH: 2,
            MarketRegime.RANGING_TIGHT: 3,
            MarketRegime.RANGING_WIDE: 4,
            MarketRegime.BREAKOUT_PENDING: 5,
            MarketRegime.VOLATILE_CHAOS: 6,
        }[self]

    @classmethod
    def from_code(cls, code: int | float | None) -> MarketRegime:
        """Map numeric indicator code back to enum."""
        normalized = int(code or 0)
        mapping = {
            1: cls.TRENDING_BULLISH,
            2: cls.TRENDING_BEARISH,
            3: cls.RANGING_TIGHT,
            4: cls.RANGING_WIDE,
            5: cls.BREAKOUT_PENDING,
            6: cls.VOLATILE_CHAOS,
        }
        return mapping.get(normalized, cls.UNKNOWN)


@dataclass(slots=True, frozen=True)
class RegimeAnalysis:
    """Market regime analysis output."""

    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_percentile: float
    mean_reversion_factor: float
    breakout_probability: float
    indicators: dict[str, float]


@dataclass(slots=True, frozen=True)
class RegimeAdaptation:
    """Recommended strategy/risk adaptation derived from the current regime."""

    grid_levels: int
    spacing_multiplier: float
    tilt_multiplier: float
    position_size_multiplier: float
    take_profit_buffer: float
    stop_loss_buffer: float


class MarketRegimeDetector:
    """
    Lightweight regime detector using trend + volatility + breakout compression.

    The implementation avoids heavy optional dependencies and is resilient when
    historical depth is limited.
    """

    def __init__(self, *, volatility_history_size: int = 252) -> None:
        self.volatility_history_size = max(30, volatility_history_size)
        self._volatility_history: list[float] = []

    def detect_regime(
        self,
        *,
        closes: Iterable[Decimal],
        volumes: Iterable[Decimal],
        highs: Iterable[Decimal],
        lows: Iterable[Decimal],
    ) -> RegimeAnalysis:
        """
        Detect the active market regime from OHLCV slices.

        Inputs should be ordered oldest -> newest.
        """
        close_vals = np.asarray([float(v) for v in closes], dtype=float)
        volume_vals = np.asarray([float(v) for v in volumes], dtype=float)
        high_vals = np.asarray([float(v) for v in highs], dtype=float)
        low_vals = np.asarray([float(v) for v in lows], dtype=float)

        min_len = min(len(close_vals), len(volume_vals), len(high_vals), len(low_vals))
        if min_len < 30:
            return self._fallback_unknown(min_len)
        close_vals = close_vals[-min_len:]
        volume_vals = volume_vals[-min_len:]
        high_vals = high_vals[-min_len:]
        low_vals = low_vals[-min_len:]

        trend_strength, trend_direction = self._detect_trend(close_vals)
        current_vol, vol_percentile = self._analyze_volatility(close_vals, high_vals, low_vals)
        hurst = self._hurst_exponent(close_vals)
        breakout_prob = self._breakout_probability(close_vals, volume_vals)
        regime, confidence = self._classify(
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            volatility_percentile=vol_percentile,
            hurst=hurst,
            breakout_probability=breakout_prob,
        )

        indicators = {
            "trend_strength": trend_strength,
            "trend_direction": float(trend_direction),
            "volatility": current_vol,
            "volatility_percentile": vol_percentile,
            "hurst_exponent": hurst,
            "breakout_probability": breakout_prob,
            "regime_code": float(regime.code),
            "regime_confidence": confidence,
        }
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=abs(trend_strength),
            volatility_percentile=vol_percentile,
            mean_reversion_factor=hurst,
            breakout_probability=breakout_prob,
            indicators=indicators,
        )

    def recommended_adaptation(
        self,
        *,
        analysis: RegimeAnalysis,
        base_grid_levels: int,
        base_take_profit_buffer: float,
        base_stop_loss_buffer: float,
    ) -> RegimeAdaptation:
        """Return regime-aware adaptation factors for grid strategy + sizing."""
        regime = analysis.regime
        if regime == MarketRegime.TRENDING_BULLISH:
            return RegimeAdaptation(
                grid_levels=max(3, base_grid_levels - 2),
                spacing_multiplier=1.55,
                tilt_multiplier=1.0,
                position_size_multiplier=0.82,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        if regime == MarketRegime.TRENDING_BEARISH:
            return RegimeAdaptation(
                grid_levels=max(3, base_grid_levels - 2),
                spacing_multiplier=1.55,
                tilt_multiplier=1.0,
                position_size_multiplier=0.80,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        if regime == MarketRegime.RANGING_TIGHT:
            return RegimeAdaptation(
                grid_levels=min(12, base_grid_levels + 2),
                spacing_multiplier=0.75,
                tilt_multiplier=1.0,
                position_size_multiplier=1.15,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        if regime == MarketRegime.RANGING_WIDE:
            return RegimeAdaptation(
                grid_levels=min(10, base_grid_levels + 1),
                spacing_multiplier=1.15,
                tilt_multiplier=1.0,
                position_size_multiplier=0.95,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        if regime == MarketRegime.BREAKOUT_PENDING:
            return RegimeAdaptation(
                grid_levels=max(3, base_grid_levels - 2),
                spacing_multiplier=1.9,
                tilt_multiplier=1.0,
                position_size_multiplier=0.68,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        if regime == MarketRegime.VOLATILE_CHAOS:
            return RegimeAdaptation(
                grid_levels=max(3, base_grid_levels - 3),
                spacing_multiplier=2.4,
                tilt_multiplier=1.0,
                position_size_multiplier=0.45,
                take_profit_buffer=base_take_profit_buffer,
                stop_loss_buffer=base_stop_loss_buffer,
            )
        return RegimeAdaptation(
            grid_levels=base_grid_levels,
            spacing_multiplier=1.0,
            tilt_multiplier=1.0,
            position_size_multiplier=1.0,
            take_profit_buffer=base_take_profit_buffer,
            stop_loss_buffer=base_stop_loss_buffer,
        )

    @staticmethod
    def _fallback_unknown(points: int) -> RegimeAnalysis:
        return RegimeAnalysis(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            trend_strength=0.0,
            volatility_percentile=0.5,
            mean_reversion_factor=0.5,
            breakout_probability=0.0,
            indicators={
                "available_points": float(points),
                "regime_code": float(MarketRegime.UNKNOWN.code),
                "regime_confidence": 0.0,
            },
        )

    def _detect_trend(self, prices: np.ndarray) -> tuple[float, int]:
        if len(prices) < 10:
            return 0.0, 0
        x = np.arange(len(prices), dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        slope_norm = np.tanh(slope / max(1e-8, float(np.mean(prices))) * 150.0)

        ema_fast = self._ema(prices, min(21, len(prices)))
        ema_slow = self._ema(prices, min(55, len(prices)))
        ema_alignment = 1.0 if ema_fast > ema_slow else (-1.0 if ema_fast < ema_slow else 0.0)

        diff = np.diff(prices)
        dm_plus = float(np.sum(np.clip(diff, a_min=0.0, a_max=None)))
        dm_minus = float(np.sum(np.clip(-diff, a_min=0.0, a_max=None)))
        dm_ratio = (dm_plus - dm_minus) / (dm_plus + dm_minus + 1e-8)

        strength = float((slope_norm * 0.55) + (ema_alignment * 0.25) + (dm_ratio * 0.20))
        direction = 1 if strength > 0.15 else (-1 if strength < -0.15 else 0)
        return max(-1.0, min(1.0, strength)), direction

    def _analyze_volatility(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> tuple[float, float]:
        parkinson = self._parkinson_volatility(highs, lows)
        yz = self._yang_zhang_volatility(closes, highs, lows)
        realized = float(np.std(np.diff(np.log(closes + 1e-8)))) if len(closes) > 3 else 0.0
        current_vol = float(max(0.0, (parkinson * 0.4) + (yz * 0.4) + (realized * 0.2)))

        self._volatility_history.append(current_vol)
        if len(self._volatility_history) > self.volatility_history_size:
            self._volatility_history = self._volatility_history[-self.volatility_history_size :]

        if len(self._volatility_history) < 20:
            return current_vol, 0.5

        sorted_hist = sorted(self._volatility_history)
        if not sorted_hist:
            return current_vol, 0.5
        rank = sum(1 for value in sorted_hist if value <= current_vol)
        percentile = rank / len(sorted_hist)
        return current_vol, float(max(0.0, min(1.0, percentile)))

    @staticmethod
    def _hurst_exponent(prices: np.ndarray) -> float:
        if len(prices) < 60:
            return 0.5
        returns = np.diff(np.log(prices + 1e-8))
        if len(returns) < 20:
            return 0.5
        lags = range(2, min(20, len(returns) // 2))
        tau: list[float] = []
        lag_vals: list[float] = []
        for lag in lags:
            chunk = np.array([np.sum(returns[i : i + lag]) for i in range(0, len(returns) - lag)])
            if len(chunk) < 2:
                continue
            sigma = float(np.std(chunk))
            if sigma <= 0:
                continue
            tau.append(np.log(sigma))
            lag_vals.append(np.log(lag))
        if len(tau) < 3:
            return 0.5
        slope = float(np.polyfit(np.asarray(lag_vals), np.asarray(tau), 1)[0])
        return float(max(0.0, min(1.0, slope)))

    def _breakout_probability(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        if len(prices) < 40:
            return 0.0
        width_now = self._bollinger_width(prices[-20:])
        widths_hist = [self._bollinger_width(prices[-(20 + i) : -i or None]) for i in range(1, 16)]
        widths_hist = [w for w in widths_hist if w > 0]
        median_width = float(np.median(widths_hist)) if widths_hist else width_now
        squeeze = 0.0 if median_width <= 0 else max(0.0, min(1.0, 1.0 - (width_now / median_width)))

        vol_recent = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else float(np.mean(volumes))
        vol_base = float(np.mean(volumes[-40:])) if len(volumes) >= 40 else float(np.mean(volumes))
        dry_up = max(0.0, min(1.0, 1.0 - (vol_recent / (vol_base + 1e-8))))

        recent = prices[-20:]
        upper_line = np.array([np.max(recent[max(0, i - 2) : i + 1]) for i in range(len(recent))])
        lower_line = np.array([np.min(recent[max(0, i - 2) : i + 1]) for i in range(len(recent))])
        upper_slope = float(np.polyfit(np.arange(len(upper_line)), upper_line, 1)[0])
        lower_slope = float(np.polyfit(np.arange(len(lower_line)), lower_line, 1)[0])
        compression = 1.0 if upper_slope < 0 and lower_slope > 0 else 0.0
        return float(max(0.0, min(1.0, (squeeze * 0.45) + (dry_up * 0.30) + (compression * 0.25))))

    def _classify(
        self,
        *,
        trend_strength: float,
        trend_direction: int,
        volatility_percentile: float,
        hurst: float,
        breakout_probability: float,
    ) -> tuple[MarketRegime, float]:
        votes: dict[MarketRegime, float] = {}
        abs_trend = abs(trend_strength)

        if abs_trend > 0.45 and trend_direction > 0:
            votes[MarketRegime.TRENDING_BULLISH] = abs_trend
        if abs_trend > 0.45 and trend_direction < 0:
            votes[MarketRegime.TRENDING_BEARISH] = abs_trend

        if abs_trend < 0.22 and hurst < 0.5:
            if volatility_percentile < 0.35:
                votes[MarketRegime.RANGING_TIGHT] = max(
                    0.35,
                    (0.5 - hurst) + (0.35 - volatility_percentile),
                )
            else:
                votes[MarketRegime.RANGING_WIDE] = max(
                    0.30,
                    (0.5 - hurst) + (volatility_percentile * 0.4),
                )

        if breakout_probability > 0.62 and abs_trend < 0.28:
            votes[MarketRegime.BREAKOUT_PENDING] = breakout_probability

        if volatility_percentile > 0.87:
            votes[MarketRegime.VOLATILE_CHAOS] = volatility_percentile

        if not votes:
            default_regime = (
                MarketRegime.RANGING_TIGHT if volatility_percentile < 0.5 else MarketRegime.RANGING_WIDE
            )
            return default_regime, 0.50

        regime, strength = max(votes.items(), key=lambda item: item[1])
        confidence = float(max(0.0, min(1.0, strength)))
        return regime, confidence

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> float:
        period = max(2, min(period, len(values)))
        mult = 2.0 / (period + 1.0)
        ema_val = float(np.mean(values[:period]))
        for value in values[period:]:
            ema_val = ((float(value) - ema_val) * mult) + ema_val
        return ema_val

    @staticmethod
    def _parkinson_volatility(highs: np.ndarray, lows: np.ndarray) -> float:
        if len(highs) < 2 or len(highs) != len(lows):
            return 0.0
        ratios = np.log((highs + 1e-8) / (lows + 1e-8))
        squared = np.square(ratios)
        return float(np.sqrt(np.mean(squared) / (4 * np.log(2))))

    @staticmethod
    def _yang_zhang_volatility(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        if len(closes) < 3:
            return 0.0
        close = closes.astype(float)
        high = highs.astype(float)
        low = lows.astype(float)
        overnight = np.diff(np.log(close + 1e-8))
        hl = np.log((high[1:] + 1e-8) / (low[1:] + 1e-8))
        if len(overnight) < 2 or len(hl) < 2:
            return float(np.std(overnight)) if len(overnight) > 0 else 0.0
        return float(np.sqrt(np.var(overnight) + (0.34 * np.var(hl))))

    @staticmethod
    def _bollinger_width(series: np.ndarray) -> float:
        if len(series) < 5:
            return 0.0
        mean_price = float(np.mean(series))
        if mean_price <= 0:
            return 0.0
        stdev = pstdev(float(v) for v in series)
        return float((4.0 * stdev) / mean_price)
