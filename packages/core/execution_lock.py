"""Async execution locks keyed by trading symbol."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_locks: dict[str, asyncio.Lock] = {}
_locks_guard = threading.Lock()


def _get_lock(symbol: str) -> asyncio.Lock:
    normalized = symbol.strip().upper()
    with _locks_guard:
        lock = _locks.get(normalized)
        if lock is None:
            lock = asyncio.Lock()
            _locks[normalized] = lock
    return lock


@asynccontextmanager
async def symbol_execution_lock(symbol: str) -> AsyncIterator[None]:
    """Serialize trade-cycle execution for a specific symbol."""
    lock = _get_lock(symbol)
    async with lock:
        yield
