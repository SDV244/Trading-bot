"""
Binance Spot Market Data Adapter

Handles all communication with Binance Spot API for market data.
Includes klines fetching, retry logic, and rate limit handling.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import httpx
from loguru import logger
from pydantic import BaseModel

from packages.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from packages.core.config import get_settings

JSONDict = dict[str, Any]
JSONList = list[Any]
JSONResponse = JSONDict | JSONList


class CandleData(BaseModel):
    """OHLCV candle data."""

    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int


class BinanceSpotAdapter:
    """
    Binance Spot market data adapter.

    Features:
    - Async HTTP client with connection pooling
    - Automatic retry with exponential backoff
    - Rate limit awareness
    - Testnet/production support
    """

    # Rate limit tracking
    MAX_REQUESTS_PER_MINUTE = 1200
    KLINES_WEIGHT = 2  # Weight for /api/v3/klines endpoint

    def __init__(self) -> None:
        """Initialize the Binance adapter."""
        settings = get_settings()
        self.base_url = settings.binance.public_market_data_url
        self.testnet = settings.binance.testnet
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0
        self._last_reset = datetime.now(UTC)
        self._request_lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self._max_retries = max(1, settings.binance.market_max_retries)
        self._min_interval_ms = max(0, settings.binance.market_min_interval_ms)
        self._circuit_breaker = CircuitBreaker(
            "binance_spot",
            CircuitBreakerConfig(
                failure_threshold=settings.binance.circuit_breaker_failure_threshold,
                success_threshold=settings.binance.circuit_breaker_success_threshold,
                timeout_seconds=settings.binance.circuit_breaker_timeout_seconds,
                window_seconds=settings.binance.circuit_breaker_window_seconds,
            ),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
        weight: int = 1,
    ) -> JSONResponse:
        """
        Make a request with retry and exponential backoff.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            max_retries: Maximum retry attempts
            weight: Request weight for rate limiting

        Returns:
            Parsed JSON response

        Raises:
            httpx.HTTPError: If all retries fail
        """
        client = await self._get_client()

        for attempt in range(max_retries):
            try:
                await self._throttle()
                # Check rate limit
                await self._check_rate_limit(weight)

                response = await self._circuit_breaker.call(client.request, method, endpoint, params=params)

                # Update rate limit from headers
                self._update_rate_limit(response)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict | list):
                    raise httpx.HTTPError("Unexpected Binance response payload type")
                return cast(JSONResponse, payload)

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    # Server error - retry with backoff
                    wait_time = 2**attempt
                    logger.warning(f"Server error {e.response.status_code}, retry in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Request error: {e}, retry in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise httpx.HTTPError(f"Failed after {max_retries} retries")

    async def _check_rate_limit(self, weight: int) -> None:
        """Check and wait if approaching rate limit."""
        now = datetime.now(UTC)
        elapsed = (now - self._last_reset).total_seconds()

        if elapsed >= 60:
            # Reset counter every minute
            self._request_count = 0
            self._last_reset = now

        if self._request_count + weight > self.MAX_REQUESTS_PER_MINUTE * 0.9:
            # Approaching limit - wait until reset
            wait_time = 60 - elapsed
            logger.info(f"Approaching rate limit, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = datetime.now(UTC)

        self._request_count += weight

    async def _throttle(self) -> None:
        """Enforce minimum time between public requests."""
        interval_seconds = self._min_interval_ms / 1000
        if interval_seconds <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < interval_seconds:
                await asyncio.sleep(interval_seconds - elapsed)
            self._last_request_ts = time.monotonic()

    def _update_rate_limit(self, response: httpx.Response) -> None:
        """Update rate limit tracking from response headers."""
        used = response.headers.get("X-MBX-USED-WEIGHT-1M")
        if used:
            self._request_count = int(used)

    async def get_server_time(self) -> datetime:
        """Get Binance server time."""
        data = await self._request_with_retry("GET", "/api/v3/time", max_retries=self._max_retries)
        if not isinstance(data, dict):
            raise ValueError("Invalid server time response")
        server_time = data.get("serverTime")
        if not isinstance(server_time, int | float):
            raise ValueError("Missing or invalid serverTime in response")
        return datetime.fromtimestamp(server_time / 1000, tz=UTC)

    async def get_ticker_price(self, symbol: str = "BTCUSDT") -> Decimal:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price as Decimal
        """
        data = await self._request_with_retry(
            "GET",
            "/api/v3/ticker/price",
            params={"symbol": symbol},
            max_retries=self._max_retries,
        )
        if not isinstance(data, dict):
            raise ValueError("Invalid ticker price response")
        price = data.get("price")
        if price is None:
            raise ValueError("Missing price in ticker response")
        return Decimal(str(price))

    async def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[CandleData]:
        """
        Fetch kline/candlestick data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1h, 4h, 1d, etc.)
            limit: Number of candles (max 1000)
            start_time: Start time for historical data
            end_time: End time for historical data

        Returns:
            List of CandleData objects
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = await self._request_with_retry(
            "GET",
            "/api/v3/klines",
            params=params,
            max_retries=self._max_retries,
            weight=self.KLINES_WEIGHT,
        )
        if not isinstance(data, list):
            raise ValueError("Invalid klines response")

        candles = []
        for item in data:
            if not isinstance(item, list) or len(item) < 9:
                raise ValueError("Malformed kline payload item")
            candle = CandleData(
                symbol=symbol,
                timeframe=interval,
                open_time=datetime.fromtimestamp(item[0] / 1000, tz=UTC),
                close_time=datetime.fromtimestamp(item[6] / 1000, tz=UTC),
                open=Decimal(str(item[1])),
                high=Decimal(str(item[2])),
                low=Decimal(str(item[3])),
                close=Decimal(str(item[4])),
                volume=Decimal(str(item[5])),
                quote_volume=Decimal(str(item[7])),
                trades_count=int(item[8]),
            )
            candles.append(candle)

        return candles

    async def get_exchange_info(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """
        Get exchange trading rules and symbol info.

        Args:
            symbol: Trading pair symbol

        Returns:
            Symbol info including filters, lot size, etc.
        """
        data = await self._request_with_retry(
            "GET",
            "/api/v3/exchangeInfo",
            params={"symbol": symbol},
            max_retries=self._max_retries,
            weight=10,
        )
        if not isinstance(data, dict):
            raise ValueError("Invalid exchange info response")

        symbols = data.get("symbols", [])
        if not isinstance(symbols, list):
            raise ValueError("Invalid symbols payload in exchange info response")

        for s in symbols:
            if isinstance(s, dict) and s.get("symbol") == symbol:
                return cast(JSONDict, s)

        raise ValueError(f"Symbol {symbol} not found")

    async def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> dict[str, Any]:
        """
        Get order book depth snapshot.

        Args:
            symbol: Trading pair symbol
            limit: Depth levels (5, 10, 20, 50, 100, 500, 1000, 5000)
        """
        valid_limits = {5, 10, 20, 50, 100, 500, 1000, 5000}
        resolved_limit = limit if limit in valid_limits else 20
        data = await self._request_with_retry(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol, "limit": resolved_limit},
            max_retries=self._max_retries,
            weight=5,
        )
        if not isinstance(data, dict):
            raise ValueError("Invalid order book response")
        return cast(JSONDict, data)

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Expose current circuit-breaker state for diagnostics."""
        return self._circuit_breaker.get_stats()

    async def reset_circuit_breaker(self) -> None:
        """Reset circuit-breaker state to CLOSED."""
        await self._circuit_breaker.reset()


# Singleton instance
_adapter: BinanceSpotAdapter | None = None
_adapter_lock = threading.Lock()


def get_binance_adapter() -> BinanceSpotAdapter:
    """Get or create the Binance adapter singleton."""
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                _adapter = BinanceSpotAdapter()
    return _adapter


async def close_binance_adapter() -> None:
    """Close the Binance adapter."""
    global _adapter
    if _adapter is not None:
        await _adapter.close()
        _adapter = None
