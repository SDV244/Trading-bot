"""Tests for EMA trend strategy."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.core.strategies.base import CandleInput, StrategyContext
from packages.core.strategies.trend import EMATrendFastStrategy, EMATrendStrategy


def _candle_series(start: Decimal, step: Decimal, length: int) -> list[CandleInput]:
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    candles: list[CandleInput] = []
    for i in range(length):
        close = start + (step * Decimal(i))
        candles.append(
            CandleInput(
                open_time=now - timedelta(hours=length - i),
                close=close,
                high=close + Decimal("10"),
                low=close - Decimal("10"),
                volume=Decimal("100"),
            )
        )
    return candles


def test_returns_hold_on_insufficient_data() -> None:
    strategy = EMATrendStrategy()
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candle_series(Decimal("50000"), Decimal("5"), 10),
        candles_4h=_candle_series(Decimal("50000"), Decimal("20"), 10),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)

    assert signal.side == "HOLD"
    assert signal.reason == "insufficient_data"


def test_returns_buy_in_bullish_regime_with_timing_confirmation() -> None:
    strategy = EMATrendStrategy(
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        timing_period_1h=5,
        pullback_threshold_bps=50,
    )
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candle_series(Decimal("51000"), Decimal("8"), 20),
        candles_4h=_candle_series(Decimal("48000"), Decimal("40"), 20),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)

    assert signal.side == "BUY"
    assert signal.reason == "bullish_regime_entry"
    assert signal.confidence >= 0.55


def test_returns_sell_in_bearish_regime_with_timing_confirmation() -> None:
    strategy = EMATrendStrategy(
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        timing_period_1h=5,
        pullback_threshold_bps=50,
    )
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candle_series(Decimal("52000"), Decimal("-7"), 20),
        candles_4h=_candle_series(Decimal("55000"), Decimal("-30"), 20),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)

    assert signal.side == "SELL"
    assert signal.reason == "bearish_regime_entry"
    assert signal.confidence >= 0.55


def test_returns_hold_when_regime_strength_below_threshold() -> None:
    strategy = EMATrendStrategy(
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        timing_period_1h=5,
        min_regime_strength_bps=1000,
    )
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candle_series(Decimal("50000"), Decimal("1"), 30),
        candles_4h=_candle_series(Decimal("50000"), Decimal("1"), 30),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "weak_regime"


def test_fast_profile_has_expected_requirements() -> None:
    strategy = EMATrendFastStrategy()
    req = strategy.data_requirements()
    assert req["1h"] == 13
    assert req["4h"] == 34
