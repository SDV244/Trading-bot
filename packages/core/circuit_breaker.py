"""Async circuit breaker utility for adapter resilience."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, ParamSpec, TypeVar


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker tuning parameters."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    window_seconds: int = 60


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the breaker is open and rejects calls."""


P = ParamSpec("P")
T = TypeVar("T")


class CircuitBreaker:
    """Fail-fast wrapper for flaky external dependencies."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_times: list[datetime] = []
        self._success_count = 0
        self._last_failure_at: datetime | None = None
        self._last_state_change_at = datetime.now(UTC)
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(
        self,
        func: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute callable through the breaker state machine."""
        await self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        await self._on_success()
        return result

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state != CircuitState.OPEN:
                return
            if not self._can_attempt_recovery():
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._last_state_change_at = datetime.now(UTC)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_times.clear()
                    self._success_count = 0
                    self._last_state_change_at = datetime.now(UTC)
                return
            # CLOSED path: a successful call clears stale failure history.
            self._failure_times.clear()

    async def _on_failure(self) -> None:
        now = datetime.now(UTC)
        async with self._lock:
            self._last_failure_at = now
            cutoff = now - timedelta(seconds=self.config.window_seconds)
            self._failure_times = [value for value in self._failure_times if value > cutoff]
            self._failure_times.append(now)

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._last_state_change_at = now
                return

            if self._state == CircuitState.CLOSED and len(self._failure_times) >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._last_state_change_at = now

    def _can_attempt_recovery(self) -> bool:
        if self._last_failure_at is None:
            return True
        elapsed = datetime.now(UTC) - self._last_failure_at
        return elapsed.total_seconds() >= self.config.timeout_seconds

    async def reset(self) -> None:
        """Manually reset breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_times.clear()
            self._success_count = 0
            self._last_failure_at = None
            self._last_state_change_at = datetime.now(UTC)

    def get_stats(self) -> dict[str, Any]:
        """Expose internal stats for observability/debugging."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count_window": len(self._failure_times),
            "success_count": self._success_count,
            "last_failure_at": self._last_failure_at.isoformat() if self._last_failure_at else None,
            "last_state_change_at": self._last_state_change_at.isoformat(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "window_seconds": self.config.window_seconds,
            },
        }
