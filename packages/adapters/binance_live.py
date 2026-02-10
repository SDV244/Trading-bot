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
        self._time_sync_lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self._server_time_offset_ms = 0
        self._last_time_sync_monotonic: float | None = None
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
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(20.0, connect=5.0),
                headers={"X-MBX-APIKEY": self.api_key},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                event_hooks={
                    "request": [self._log_request_safe],
                    "response": [self._log_response_safe],
                },
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
        await self._sync_time()

        last_error: str | None = None
        for attempt in range(max_retries):
            params: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": str(quantity),
                "recvWindow": recv_window_ms,
                "timestamp": self._timestamp_ms(),
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

            if self._binance_error_code(response) == -1021 and attempt < max_retries - 1:
                await self._sync_time(force=True)
                await asyncio.sleep(self._retry_delay(attempt))
                continue

            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._retry_delay(attempt)
                last_error = f"status={response.status_code}"
                await asyncio.sleep(delay)
                continue

            safe_text = self._sanitize_text(response.text)
            msg = f"Binance order failed: {response.status_code} {safe_text}"
            logger.error(msg)
            raise BinanceLiveAdapterError(msg)

        raise BinanceLiveAdapterError(f"Live order failed after retries ({last_error})")

    async def get_exchange_filters(self, symbol: str) -> dict[str, Any]:
        """Fetch exchange filters for symbol validation."""
        client = await self._get_client()
        response = await self._circuit_breaker.call(client.get, "/api/v3/exchangeInfo", params={"symbol": symbol})
        if response.status_code >= 400:
            safe_text = self._sanitize_text(response.text)
            msg = f"Binance exchangeInfo failed: {response.status_code} {safe_text}"
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
        await self._sync_time()
        params: dict[str, Any] = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "recvWindow": recv_window_ms,
            "timestamp": self._timestamp_ms(),
        }
        signed = self._sign_params(params)
        await self._throttle()
        try:
            response = await self._circuit_breaker.call(client.get, "/api/v3/order", params=signed)
        except httpx.RequestError as e:
            raise BinanceLiveAdapterError("Binance query order request error") from e
        if response.status_code == 404:
            return None
        if self._binance_error_code(response) == -1021:
            await self._sync_time(force=True)
            params["timestamp"] = self._timestamp_ms()
            signed = self._sign_params(params)
            response = await self._circuit_breaker.call(client.get, "/api/v3/order", params=signed)
            if response.status_code == 404:
                return None
        if response.status_code >= 400:
            safe_text = self._sanitize_text(response.text)
            msg = f"Binance query order failed: {response.status_code} {safe_text}"
            raise BinanceLiveAdapterError(msg)
        return cast(dict[str, Any], response.json())

    async def get_account_balances(
        self,
        *,
        recv_window: int | None = None,
    ) -> dict[str, Decimal]:
        """Fetch account balances by asset (free + locked)."""
        recv_window_ms = recv_window or self.settings.live.recv_window_ms
        client = await self._get_client()
        await self._sync_time()
        params: dict[str, Any] = {
            "recvWindow": recv_window_ms,
            "timestamp": self._timestamp_ms(),
        }
        signed = self._sign_params(params)
        await self._throttle()
        response = await self._circuit_breaker.call(client.get, "/api/v3/account", params=signed)
        if self._binance_error_code(response) == -1021:
            await self._sync_time(force=True)
            params["timestamp"] = self._timestamp_ms()
            signed = self._sign_params(params)
            response = await self._circuit_breaker.call(client.get, "/api/v3/account", params=signed)
        if response.status_code >= 400:
            safe_text = self._sanitize_text(response.text)
            raise BinanceLiveAdapterError(
                f"Binance account endpoint failed: {response.status_code} {safe_text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise BinanceLiveAdapterError("Binance account response malformed")
        balances = payload.get("balances")
        if not isinstance(balances, list):
            raise BinanceLiveAdapterError("Binance account balances payload missing")
        parsed: dict[str, Decimal] = {}
        for raw in balances:
            if not isinstance(raw, dict):
                continue
            asset = str(raw.get("asset", "")).upper()
            if not asset:
                continue
            free = Decimal(str(raw.get("free", "0")))
            locked = Decimal(str(raw.get("locked", "0")))
            parsed[asset] = free + locked
        return parsed

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

    async def _sync_time(self, *, force: bool = False) -> None:
        if not force and not self._needs_time_sync():
            return
        async with self._time_sync_lock:
            if not force and not self._needs_time_sync():
                return
            client = await self._get_client()
            await self._throttle()
            response = await self._circuit_breaker.call(client.get, "/api/v3/time")
            if response.status_code >= 400:
                safe_text = self._sanitize_text(response.text)
                raise BinanceLiveAdapterError(
                    f"Binance time sync failed: {response.status_code} {safe_text}"
                )
            payload = response.json()
            if not isinstance(payload, dict) or "serverTime" not in payload:
                raise BinanceLiveAdapterError("Binance time sync returned malformed payload")
            server_time = int(payload["serverTime"])
            local_time = int(time.time() * 1000)
            self._server_time_offset_ms = server_time - local_time
            self._last_time_sync_monotonic = time.monotonic()
            logger.debug(f"Binance live time synced (offset_ms={self._server_time_offset_ms})")

    def _needs_time_sync(self) -> bool:
        if self._last_time_sync_monotonic is None:
            return True
        return (time.monotonic() - self._last_time_sync_monotonic) >= 3600

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._server_time_offset_ms

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return float(min(0.5 * (2**attempt), 5.0))

    @staticmethod
    def _generate_client_order_id() -> str:
        # Binance max length is 36.
        return f"tb_{uuid4().hex[:24]}"

    @staticmethod
    def _sanitize_text(value: str) -> str:
        if not value:
            return value
        sanitized = re.sub(r"signature=[^&\\s]+", "signature=REDACTED", value)
        sanitized = re.sub(r"X-MBX-APIKEY=[^&\\s]+", "X-MBX-APIKEY=REDACTED", sanitized)
        sanitized = re.sub(r"api[_-]?key\"?\s*[:=]\s*\"?[A-Za-z0-9_\-]+", "api_key=REDACTED", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"api[_-]?secret\"?\s*[:=]\s*\"?[A-Za-z0-9_\-]+", "api_secret=REDACTED", sanitized, flags=re.IGNORECASE)
        return sanitized

    @staticmethod
    def _binance_error_code(response: httpx.Response) -> int | None:
        try:
            payload = response.json()
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        if isinstance(code, int):
            return code
        return None

    async def _log_request_safe(self, request: httpx.Request) -> None:
        logger.debug(f"Binance live request {request.method} {request.url.path}")

    async def _log_response_safe(self, response: httpx.Response) -> None:
        request = response.request
        logger.debug(
            f"Binance live response {response.status_code} {request.method} {request.url.path}"
        )

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Expose current circuit-breaker state for diagnostics."""
        return self._circuit_breaker.get_stats()

    async def reset_circuit_breaker(self) -> None:
        """Reset circuit-breaker state to CLOSED."""
        await self._circuit_breaker.reset()


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
