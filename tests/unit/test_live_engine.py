"""Tests for live engine safety gate."""

from decimal import Decimal
from typing import Any

import pytest

from packages.core.config import reload_settings
from packages.core.execution import LiveEngine, LiveEngineError, LiveSafetyChecklist, OrderRequest


@pytest.mark.asyncio
async def test_live_engine_rejects_when_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "false")
    reload_settings()
    engine = LiveEngine()
    with pytest.raises(LiveEngineError, match="live_mode is disabled"):
        await engine.execute_market_order(
            OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"), order_type="MARKET"),
            checklist=LiveSafetyChecklist(ui_confirmed=True, reauthenticated=True, safety_acknowledged=True),
        )


@pytest.mark.asyncio
async def test_live_engine_requires_all_checklist_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "true")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    engine = LiveEngine()
    with pytest.raises(LiveEngineError, match="UI confirmation"):
        await engine.execute_market_order(
            OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"), order_type="MARKET"),
            checklist=LiveSafetyChecklist(ui_confirmed=False, reauthenticated=True, safety_acknowledged=True),
        )


class _MockLiveAdapter:
    async def get_exchange_filters(self, symbol: str) -> dict[str, Any]:
        assert symbol == "BTCUSDT"
        return {
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.0001", "stepSize": "0.0001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
            ]
        }

    async def place_market_order(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "FILLED",
            "orderId": 12345,
            "executedQty": kwargs["quantity"],
            "cummulativeQuoteQty": "505.0",
            "clientOrderId": kwargs.get("new_client_order_id"),
        }

    async def query_order(self, **_kwargs: Any) -> dict[str, Any] | None:
        return None


@pytest.mark.asyncio
async def test_live_engine_executes_with_mock_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "true")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    monkeypatch.setattr("packages.core.execution.live_engine.get_binance_live_adapter", lambda: _MockLiveAdapter())
    engine = LiveEngine()
    result = await engine.execute_market_order(
        OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.0100"), order_type="MARKET"),
        checklist=LiveSafetyChecklist(ui_confirmed=True, reauthenticated=True, safety_acknowledged=True),
    )
    assert result.accepted is True
    assert result.order_id == "12345"
    assert result.quantity == Decimal("0.0100")


@pytest.mark.asyncio
async def test_live_engine_normalizes_quantity_to_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "true")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    monkeypatch.setattr("packages.core.execution.live_engine.get_binance_live_adapter", lambda: _MockLiveAdapter())
    engine = LiveEngine()
    result = await engine.execute_market_order(
        OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01009"), order_type="MARKET"),
        checklist=LiveSafetyChecklist(ui_confirmed=True, reauthenticated=True, safety_acknowledged=True),
    )
    assert result.accepted is True
    assert result.quantity == Decimal("0.0100")


class _RecoveringLiveAdapter(_MockLiveAdapter):
    async def place_market_order(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("network timeout")

    async def query_order(self, **kwargs: Any) -> dict[str, Any] | None:
        return {
            "status": "FILLED",
            "orderId": 99999,
            "executedQty": "0.0100",
            "cummulativeQuoteQty": "500.0",
            "clientOrderId": kwargs["client_order_id"],
        }


@pytest.mark.asyncio
async def test_live_engine_recovers_order_after_send_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "true")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    monkeypatch.setattr("packages.core.execution.live_engine.get_binance_live_adapter", lambda: _RecoveringLiveAdapter())
    engine = LiveEngine()
    result = await engine.execute_market_order(
        OrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.0100"), order_type="MARKET"),
        checklist=LiveSafetyChecklist(ui_confirmed=True, reauthenticated=True, safety_acknowledged=True),
    )
    assert result.accepted is True
    assert result.order_id == "99999"
    assert result.client_order_id is not None
