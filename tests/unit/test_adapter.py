"""
Tests for Binance Spot Adapter.
"""

from datetime import datetime
from decimal import Decimal

import httpx
import pytest
import respx

from packages.adapters.binance_spot import BinanceSpotAdapter, CandleData


@pytest.fixture
async def adapter():
    """Create adapter instance."""
    adapter = BinanceSpotAdapter()
    yield adapter
    await adapter.close()


@pytest.mark.asyncio
async def test_get_server_time(adapter):
    """Can fetch server time."""
    async with respx.mock(base_url=adapter.base_url) as respx_mock:
        respx_mock.get("/api/v3/time").mock(
            return_value=httpx.Response(200, json={"serverTime": 1700000000000})
        )

        server_time = await adapter.get_server_time()
        assert isinstance(server_time, datetime)
        assert server_time.timestamp() == 1700000000.0


@pytest.mark.asyncio
async def test_get_ticker_price(adapter):
    """Can fetch current price."""
    async with respx.mock(base_url=adapter.base_url) as respx_mock:
        respx_mock.get("/api/v3/ticker/price").mock(
            return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "price": "50000.00"})
        )

        price = await adapter.get_ticker_price("BTCUSDT")
        assert isinstance(price, Decimal)
        assert price == Decimal("50000.00")


@pytest.mark.asyncio
async def test_get_klines(adapter):
    """Can fetch klines."""
    # Mock response: [Open time, Open, High, Low, Close, Volume, Close time, ...]
    mock_kline = [
        1700000000000, "50000.00", "51000.00", "49000.00", "50500.00", "100.0",
        1700003599999, "5000000.00", 1000, "0", "0", "0"
    ]

    async with respx.mock(base_url=adapter.base_url) as respx_mock:
        respx_mock.get("/api/v3/klines").mock(
            return_value=httpx.Response(200, json=[mock_kline])
        )

        candles = await adapter.get_klines("BTCUSDT", "1h", limit=1)
        assert len(candles) == 1
        c = candles[0]
        assert isinstance(c, CandleData)
        assert c.symbol == "BTCUSDT"
        assert c.timeframe == "1h"
        assert c.open == Decimal("50000.00")
        assert c.high == Decimal("51000.00")
        assert c.low == Decimal("49000.00")
        assert c.close == Decimal("50500.00")
        assert c.volume == Decimal("100.0")
        assert c.trades_count == 1000


@pytest.mark.asyncio
async def test_rate_limit_handling(adapter):
    """Handles rate limit 429 response."""
    async with respx.mock(base_url=adapter.base_url) as respx_mock:
        # First request fails with 429
        route = respx_mock.get("/api/v3/time")
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"serverTime": 1700000000000})
        ]

        # Test should wait (mock sleep to avoid actual delay?)
        # For this test we just verify it retries
        server_time = await adapter.get_server_time()
        assert server_time.timestamp() == 1700000000.0
        assert route.call_count == 2
