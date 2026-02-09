"""Tests for paper execution engine."""

from decimal import Decimal

import pytest

from packages.core.execution.paper_engine import OrderRequest, PaperEngine, PaperExecutionError


def test_market_buy_applies_positive_slippage_and_fee() -> None:
    engine = PaperEngine(fee_bps=10, slippage_bps=5)
    order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01000"))

    fill = engine.execute_market_order(order=order, last_price=Decimal("50000"))

    assert fill.status == "FILLED"
    assert fill.price == Decimal("50025.00")
    assert fill.notional == Decimal("500.2500000")
    assert fill.fee == Decimal("0.50025000")
    assert fill.remaining_quantity == Decimal("0")


def test_market_sell_applies_negative_slippage() -> None:
    engine = PaperEngine(slippage_bps=5)
    order = OrderRequest(symbol="BTCUSDT", side="SELL", quantity=Decimal("0.01000"))

    fill = engine.execute_market_order(order=order, last_price=Decimal("50000"))

    assert fill.status == "FILLED"
    assert fill.price == Decimal("49975.00")
    assert fill.slippage_bps == 5


def test_market_order_rejects_below_min_notional() -> None:
    engine = PaperEngine(min_notional=Decimal("10"))
    order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.00010"))

    with pytest.raises(PaperExecutionError, match="below minimum"):
        engine.execute_market_order(order=order, last_price=Decimal("50000"))


def test_rejects_quantity_not_matching_lot_step_size() -> None:
    engine = PaperEngine(lot_step_size=Decimal("0.00010"))
    order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.00015"))

    with pytest.raises(PaperExecutionError, match="lot_step_size"):
        engine.execute_market_order(order=order, last_price=Decimal("50000"))


def test_limit_buy_fills_when_price_range_touches_limit() -> None:
    engine = PaperEngine()
    order = OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.01000"),
        order_type="LIMIT",
        limit_price=Decimal("49900"),
    )

    fill = engine.execute_limit_order(
        order=order,
        candle_low=Decimal("49850"),
        candle_high=Decimal("50100"),
    )

    assert fill.status == "FILLED"
    assert fill.price == Decimal("49900")
    assert fill.filled_quantity == Decimal("0.01000")
    assert fill.remaining_quantity == Decimal("0.00000")


def test_limit_order_remains_open_when_not_touched() -> None:
    engine = PaperEngine()
    order = OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.01000"),
        order_type="LIMIT",
        limit_price=Decimal("49000"),
    )

    fill = engine.execute_limit_order(
        order=order,
        candle_low=Decimal("49500"),
        candle_high=Decimal("50100"),
    )

    assert fill.status == "OPEN"
    assert fill.filled_quantity == Decimal("0")
    assert fill.remaining_quantity == Decimal("0.01000")
    assert fill.price is None


def test_limit_partial_fill_supported() -> None:
    engine = PaperEngine(lot_step_size=Decimal("0.00001"))
    order = OrderRequest(
        symbol="BTCUSDT",
        side="SELL",
        quantity=Decimal("0.02000"),
        order_type="LIMIT",
        limit_price=Decimal("50500"),
    )

    fill = engine.execute_limit_order(
        order=order,
        candle_low=Decimal("50000"),
        candle_high=Decimal("50600"),
        partial_fill_ratio=Decimal("0.5"),
    )

    assert fill.status == "PARTIALLY_FILLED"
    assert fill.filled_quantity == Decimal("0.01000")
    assert fill.remaining_quantity == Decimal("0.01000")


def test_symbol_allowlist_rejects_unknown_symbol() -> None:
    engine = PaperEngine(allowed_symbols={"ETHUSDT"})
    order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01000"))

    with pytest.raises(PaperExecutionError, match="not allowed"):
        engine.execute_market_order(order=order, last_price=Decimal("50000"))


def test_symbol_allowlist_accepts_configured_symbol() -> None:
    engine = PaperEngine(allowed_symbols={"ETHUSDT"})
    order = OrderRequest(symbol="ETHUSDT", side="BUY", quantity=Decimal("0.10000"))

    fill = engine.execute_market_order(order=order, last_price=Decimal("3000"))
    assert fill.status == "FILLED"
    assert fill.symbol == "ETHUSDT"
