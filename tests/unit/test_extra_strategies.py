"""Tests for additional strategy implementations."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.core.strategies import (
    BreakoutStrategy,
    CandleInput,
    MeanReversionStrategy,
    StrategyContext,
)


def _make_candles(length: int, start: Decimal, step: Decimal) -> list[CandleInput]:
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    rows: list[CandleInput] = []
    for i in range(length):
        close = start + (step * Decimal(i))
        rows.append(
            CandleInput(
                open_time=now - timedelta(hours=length - i),
                close=close,
                high=close + Decimal("10"),
                low=close - Decimal("10"),
                volume=Decimal("100"),
            )
        )
    return rows


def test_mean_reversion_emits_hold_or_signal() -> None:
    strategy = MeanReversionStrategy(lookback=20)
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_make_candles(60, Decimal("50000"), Decimal("1")),
        candles_4h=_make_candles(40, Decimal("50000"), Decimal("5")),
        now=datetime.now(UTC),
    )
    signal = strategy.generate_signal(context)
    assert signal.side in {"BUY", "SELL", "HOLD"}
    assert "zscore" in signal.indicators


def test_breakout_buy_signal_on_confirmed_up_break() -> None:
    strategy = BreakoutStrategy(lookback_4h=10, confirmation_bars_1h=2, breakout_buffer_bps=0)
    candles_4h = _make_candles(20, Decimal("50000"), Decimal("2"))
    candles_1h = _make_candles(20, Decimal("50000"), Decimal("2"))
    # Force breakout above 4h range high
    max_high = max(c.high for c in candles_4h[-10:])
    candles_1h[-2] = CandleInput(
        open_time=candles_1h[-2].open_time,
        close=max_high + Decimal("5"),
        high=max_high + Decimal("10"),
        low=max_high - Decimal("1"),
        volume=Decimal("100"),
    )
    candles_1h[-1] = CandleInput(
        open_time=candles_1h[-1].open_time,
        close=max_high + Decimal("6"),
        high=max_high + Decimal("10"),
        low=max_high,
        volume=Decimal("100"),
    )
    context = StrategyContext(symbol="BTCUSDT", candles_1h=candles_1h, candles_4h=candles_4h, now=datetime.now(UTC))
    signal = strategy.generate_signal(context)
    assert signal.side == "BUY"
