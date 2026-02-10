"""Generic webhook notifier for critical/system alerts."""

from __future__ import annotations

import asyncio
import threading

import httpx
from loguru import logger

from packages.core.config import get_settings


class WebhookNotifier:
    """Send JSON alerts to a configured webhook endpoint."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.webhook.enabled and bool(settings.webhook.url)
        self.url = settings.webhook.url
        self.timeout_seconds = settings.webhook.timeout_seconds
        self.max_retries = settings.webhook.max_retries
        self.critical_only = settings.webhook.critical_only
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(float(self.timeout_seconds), connect=3.0),
                    )
        return self._client

    async def close(self) -> None:
        """Close internal HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send_critical_alert(self, title: str, body: str) -> bool:
        """Send high-priority alert payload."""
        return await self._send(level="critical", title=title, body=body)

    async def send_info(self, title: str, body: str) -> bool:
        """Send informational payload when critical-only mode is disabled."""
        if self.critical_only:
            return False
        return await self._send(level="info", title=title, body=body)

    async def _send(self, *, level: str, title: str, body: str) -> bool:
        if not self.enabled or not self.url:
            return False

        payload = {
            "level": level,
            "title": title,
            "body": body,
            "source": "trading-bot",
        }
        client = await self._get_client()
        for attempt in range(max(1, self.max_retries + 1)):
            try:
                response = await client.post(self.url, json=payload)
                if response.status_code < 400:
                    return True
                logger.warning(
                    f"Webhook returned {response.status_code} on attempt {attempt}: {response.text[:200]}"
                )
            except Exception as e:
                logger.warning(f"Webhook send failed on attempt {attempt}: {e}")
            if attempt < self.max_retries:
                await asyncio.sleep(min(0.5 * (2**attempt), 5.0))
        return False


_webhook_notifier: WebhookNotifier | None = None
_webhook_notifier_lock = threading.Lock()


def get_webhook_notifier() -> WebhookNotifier:
    """Get or create singleton webhook notifier."""
    global _webhook_notifier
    if _webhook_notifier is None:
        with _webhook_notifier_lock:
            if _webhook_notifier is None:
                _webhook_notifier = WebhookNotifier()
    return _webhook_notifier


async def close_webhook_notifier() -> None:
    """Close singleton webhook notifier."""
    global _webhook_notifier
    if _webhook_notifier is not None:
        await _webhook_notifier.close()
        _webhook_notifier = None
