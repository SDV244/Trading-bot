"""Tests for market regime detection module."""

from decimal import Decimal

from packages.core.market_regime import MarketRegime, MarketRegimeDetector


def _series(start: float, step: float, n: int) -> list[Decimal]:
    return [Decimal(str(start + (step * idx))) for idx in range(n)]


def test_detects_bullish_trending_regime() -> None:
    detector = MarketRegimeDetector()
    closes = _series(100.0, 0.6, 180)
    highs = [price * Decimal("1.004") for price in closes]
    lows = [price * Decimal("0.996") for price in closes]
    volumes = [Decimal("1000") for _ in closes]

    analysis = detector.detect_regime(closes=closes, volumes=volumes, highs=highs, lows=lows)
    assert analysis.regime in {MarketRegime.TRENDING_BULLISH, MarketRegime.RANGING_WIDE}
    assert 0.0 <= analysis.confidence <= 1.0
    assert "regime_code" in analysis.indicators


def test_detects_volatile_regime_on_large_swings() -> None:
    detector = MarketRegimeDetector()
    closes: list[Decimal] = []
    value = Decimal("100")
    for idx in range(220):
        swing = Decimal("4.5") if idx % 2 == 0 else Decimal("-4.2")
        value = max(Decimal("25"), value + swing)
        closes.append(value)
    highs = [price * Decimal("1.02") for price in closes]
    lows = [price * Decimal("0.98") for price in closes]
    volumes = [Decimal("1800") for _ in closes]

    analysis = detector.detect_regime(closes=closes, volumes=volumes, highs=highs, lows=lows)
    assert analysis.regime in {MarketRegime.VOLATILE_CHAOS, MarketRegime.RANGING_WIDE}
    assert analysis.volatility_percentile >= 0.5


def test_transition_probability_and_persistence_scores() -> None:
    detector = MarketRegimeDetector()
    closes = _series(100.0, 0.2, 240)
    highs = [price * Decimal("1.006") for price in closes]
    lows = [price * Decimal("0.994") for price in closes]
    volumes = [Decimal("1200") for _ in closes]

    analysis_first = detector.detect_regime(closes=closes, volumes=volumes, highs=highs, lows=lows)
    analysis_second = detector.detect_regime(closes=closes, volumes=volumes, highs=highs, lows=lows)

    probs = detector.get_regime_transition_probability(analysis_second.regime)
    persistence = detector.get_regime_persistence_score(analysis_second.regime)

    assert analysis_first.regime in probs
    assert 0.0 <= persistence <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6
