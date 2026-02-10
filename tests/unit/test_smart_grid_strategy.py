"""Tests for smart adaptive grid strategy."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.core.strategies.base import CandleInput, StrategyContext
from packages.core.strategies.smart_grid import SmartAdaptiveGridStrategy


def _candles_from_closes(closes: list[Decimal], *, hour_step: int = 1) -> list[CandleInput]:
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    out: list[CandleInput] = []
    for idx, close in enumerate(closes):
        out.append(
            CandleInput(
                open_time=now - timedelta(hours=(len(closes) - idx) * hour_step),
                close=close,
                high=close + Decimal("1"),
                low=close - Decimal("1"),
                volume=Decimal("100"),
            )
        )
    return out


def test_returns_hold_on_insufficient_data() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
    )
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes([Decimal("100")] * 20),
        candles_4h=_candles_from_closes([Decimal("100")] * 6, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "insufficient_data"
    assert signal.indicators["required_1h"] == 40.0


def test_generates_buy_signal_on_grid_cross_down() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=500,
    )
    closes_1h = [Decimal("100")] * 39 + [Decimal("100"), Decimal("95")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "BUY"
    assert signal.reason == "grid_buy_rebalance"
    assert "grid_center" in signal.indicators
    assert "buy_trigger" in signal.indicators


def test_generates_sell_signal_on_grid_cross_up() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=500,
    )
    closes_1h = [Decimal("100")] * 39 + [Decimal("100"), Decimal("105")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "SELL"
    assert signal.reason == "grid_sell_rebalance"
    assert "sell_trigger" in signal.indicators


def test_holds_when_price_inside_grid_band() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
    )
    closes_1h = [Decimal("100")] * 41
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "grid_wait_inside_band"


def test_arithmetic_spacing_mode_generates_valid_signal() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        spacing_mode="arithmetic",
    )
    closes_1h = [Decimal("100")] * 41
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )
    signal = strategy.generate_signal(context)
    assert signal.reason in {
        "grid_wait_inside_band",
        "grid_recenter_wait",
        "grid_recentered_auto",
        "grid_recentered_auto_breakout_buy",
        "grid_recentered_auto_breakdown_sell",
    }
    assert signal.indicators["spacing_mode_geometric"] == 0.0


def test_invalid_spacing_mode_holds() -> None:
    strategy = SmartAdaptiveGridStrategy(spacing_mode="bad_mode")
    closes_1h = [Decimal("100")] * 130
    closes_4h = [Decimal("100")] * 70
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )
    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "invalid_grid_spacing_mode"


def test_auto_recenter_when_price_moves_outside_band() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=150,
        take_profit_buffer=0.0,
        stop_loss_buffer=0.0,
    )
    closes_1h = [Decimal("100")] * 38 + [Decimal("130"), Decimal("132"), Decimal("134")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "grid_recentered_auto"
    assert signal.indicators["recentered"] == 1.0
    assert signal.indicators["grid_center"] == float(closes_1h[-1])


def test_auto_recenter_breakout_up_emits_buy() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=150,
        take_profit_buffer=0.0,
        stop_loss_buffer=0.0,
    )
    closes_1h = [Decimal("100")] * 39 + [Decimal("100"), Decimal("125")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "BUY"
    assert signal.reason == "grid_recentered_auto_breakout_buy"
    assert signal.indicators["recentered"] == 1.0


def test_auto_recenter_breakdown_down_emits_sell() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=150,
        take_profit_buffer=0.0,
        stop_loss_buffer=0.0,
    )
    closes_1h = [Decimal("100")] * 39 + [Decimal("100"), Decimal("75")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "SELL"
    assert signal.reason == "grid_recentered_auto_breakdown_sell"
    assert signal.indicators["recentered"] == 1.0


def test_conservative_recenter_mode_holds_after_out_of_band_move() -> None:
    strategy = SmartAdaptiveGridStrategy(
        lookback_1h=40,
        atr_period_1h=10,
        regime_fast_period_4h=5,
        regime_slow_period_4h=8,
        min_spacing_bps=10,
        max_spacing_bps=150,
        take_profit_buffer=0.0,
        stop_loss_buffer=0.0,
        recenter_mode="conservative",
    )
    closes_1h = [Decimal("100")] * 39 + [Decimal("100"), Decimal("125")]
    closes_4h = [Decimal("100")] * 20
    context = StrategyContext(
        symbol="BTCUSDT",
        candles_1h=_candles_from_closes(closes_1h),
        candles_4h=_candles_from_closes(closes_4h, hour_step=4),
        now=datetime.now(UTC),
    )

    signal = strategy.generate_signal(context)
    assert signal.side == "HOLD"
    assert signal.reason == "grid_recentered_auto"
    assert signal.indicators["recentered"] == 1.0
    assert signal.indicators["recenter_mode_aggressive"] == 0.0
