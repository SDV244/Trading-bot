"""Binance Spot live trading adapter with signed requests."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import threading
import time
from decimal import Decimal
from typing import Any, cast
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from loguru import logger

from packages.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from packages.core.config import get_settings


class BinanceLiveAdapterError(ValueError):
    """Raised when live adapter request or validation fails."""


class BinanceLiveAdapter:
    """Signed Binance Spot trading adapter (private endpoints)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.api_key = settings.binance.api_key
        self.api_secret = settings.binance.api_secret
        self.base_url = settings.binance.trading_base_url
        self._client: httpx.AsyncClient | None = None
        self._request_lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self._circuit_breaker = CircuitBreaker(
            "binance_live",
            CircuitBreakerConfig(
                failure_threshold=settings.binance.circuit_breaker_failure_threshold,
                success_threshold=settings.binance.circuit_breaker_success_threshold,
                timeout_seconds=settings.binance.circuit_breaker_timeout_seconds,
                window_seconds=settings.binance.circuit_breaker_window_seconds,
            ),
        )
        if not self.api_key or not self.api_secret:
            raise BinanceLiveAdapterError("Missing BINANCE_API_KEY/BINANCE_API_SECRET for live mode")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(20.0, connect=5.0),
                headers={"X-MBX-APIKEY": self.api_key},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close open HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        new_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        """Place signed MARKET order."""
        client_order_id = new_client_order_id or self._generate_client_order_id()
        recv_window_ms = recv_window or self.settings.live.recv_window_ms
        max_retries = max(1, self.settings.live.max_retries)
        client = await self._get_client()

        last_error: str | None = None
        for attempt in range(max_retries):
            params: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": str(quantity),
                "recvWindow": recv_window_ms,
                "timestamp": int(time.time() * 1000),
                "newClientOrderId": client_order_id,
            }
            signed = self._sign_params(params)

            await self._throttle()
            try:
                response = await self._circuit_breaker.call(client.post, "/api/v3/order", params=signed)
            except httpx.RequestError as e:
                last_error = self._sanitize_text(str(e))
                if attempt < max_retries - 1:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise BinanceLiveAdapterError("Binance order request error") from e

            if response.status_code < 400:
                return cast(dict[str, Any], response.json())

            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._retry_delay(attempt)
                last_error = f"status={response.status_code}"
                await asyncio.sleep(delay)
                continue

            msg = f"Binance order failed: {response.status_code} {response.text}"
            logger.error(msg)
            raise BinanceLiveAdapterError(msg)

        raise BinanceLiveAdapterError(f"Live order failed after retries ({last_error})")

    async def get_exchange_filters(self, symbol: str) -> dict[str, Any]:
        """Fetch exchange filters for symbol validation."""
        client = await self._get_client()
        response = await self._circuit_breaker.call(client.get, "/api/v3/exchangeInfo", params={"symbol": symbol})
        if response.status_code >= 400:
            msg = f"Binance exchangeInfo failed: {response.status_code} {response.text}"
            raise BinanceLiveAdapterError(msg)
        data = response.json()
        symbols = data.get("symbols", [])
        if not symbols:
            raise BinanceLiveAdapterError(f"Symbol {symbol} not found in exchangeInfo")
        return cast(dict[str, Any], symbols[0])

    async def query_order(
        self,
        *,
        symbol: str,
        client_order_id: str,
        recv_window: int | None = None,
    ) -> dict[str, Any] | None:
        """Query order by client order id for idempotent recovery checks."""
        recv_window_ms = recv_window or self.settings.live.recv_window_ms
        client = await self._get_client()
        params: dict[str, Any] = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "recvWindow": recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        signed = self._sign_params(params)
        await self._throttle()
        try:
            response = await self._circuit_breaker.call(client.get, "/api/v3/order", params=signed)
        except httpx.RequestError as e:
            raise BinanceLiveAdapterError("Binance query order request error") from e
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            msg = f"Binance query order failed: {response.status_code} {response.text}"
            raise BinanceLiveAdapterError(msg)
        return cast(dict[str, Any], response.json())

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed = dict(params)
        signed["signature"] = signature
        return signed

    async def _throttle(self) -> None:
        interval_seconds = max(0.0, self.settings.live.min_interval_ms / 1000)
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < interval_seconds:
                await asyncio.sleep(interval_seconds - elapsed)
            self._last_request_ts = time.monotonic()

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return float(min(0.5 * (2**attempt), 5.0))

    @staticmethod
    def _generate_client_order_id() -> str:
        # Binance max length is 36.
        return f"tb_{uuid4().hex[:24]}"

    @staticmethod
    def _sanitize_text(value: str) -> str:
        sanitized = re.sub(r"signature=[^&\\s]+", "signature=REDACTED", value)
        sanitized = re.sub(r"X-MBX-APIKEY=[^&\\s]+", "X-MBX-APIKEY=REDACTED", sanitized)
        return sanitized


_live_adapter: BinanceLiveAdapter | None = None
_live_adapter_lock = threading.Lock()


def get_binance_live_adapter() -> BinanceLiveAdapter:
    """Get or create singleton live adapter."""
    global _live_adapter
    if _live_adapter is None:
        with _live_adapter_lock:
            if _live_adapter is None:
                _live_adapter = BinanceLiveAdapter()
    return _live_adapter


async def close_binance_live_adapter() -> None:
    """Close singleton live adapter."""
    global _live_adapter
    if _live_adapter is not None:
        await _live_adapter.close()
        _live_adapter = None
