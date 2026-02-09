"""Tests for Binance live adapter."""

from decimal import Decimal
from urllib.parse import parse_qs

import httpx
import pytest

from packages.adapters.binance_live import BinanceLiveAdapter
from packages.core.config import reload_settings


@pytest.mark.asyncio
async def test_sign_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    adapter = BinanceLiveAdapter()
    signed = adapter._sign_params({"symbol": "BTCUSDT", "timestamp": 1})  # noqa: SLF001
    assert "signature" in signed
    assert len(str(signed["signature"])) == 64


@pytest.mark.asyncio
async def test_place_market_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    adapter = BinanceLiveAdapter()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/api/v3/order")
        return httpx.Response(
            status_code=200,
            json={"status": "FILLED", "orderId": 123, "executedQty": "0.01", "cummulativeQuoteQty": "500"},
        )

    client = httpx.AsyncClient(base_url="https://testnet.binance.vision", transport=httpx.MockTransport(handler))
    async def _fake_get_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr(adapter, "_get_client", _fake_get_client)

    order = await adapter.place_market_order(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"))
    assert order["status"] == "FILLED"
    await client.aclose()


@pytest.mark.asyncio
async def test_place_market_order_retries_with_same_client_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    reload_settings()
    adapter = BinanceLiveAdapter()

    seen_client_ids: list[str] = []
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        query = parse_qs(request.url.query.decode() if isinstance(request.url.query, bytes) else request.url.query)
        cid = query["newClientOrderId"][0]
        seen_client_ids.append(cid)
        if call_count == 1:
            return httpx.Response(status_code=500, text="server error")
        return httpx.Response(
            status_code=200,
            json={
                "status": "FILLED",
                "orderId": 321,
                "clientOrderId": cid,
                "executedQty": "0.01",
                "cummulativeQuoteQty": "500",
            },
        )

    client = httpx.AsyncClient(base_url="https://testnet.binance.vision", transport=httpx.MockTransport(handler))

    async def _fake_get_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr(adapter, "_get_client", _fake_get_client)
    order = await adapter.place_market_order(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"))
    assert call_count == 2
    assert len(seen_client_ids) == 2
    assert seen_client_ids[0] == seen_client_ids[1]
    assert order["status"] == "FILLED"
    await client.aclose()
