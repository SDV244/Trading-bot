"""Tests for circuit breaker behavior."""

import asyncio

import pytest

from packages.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold_failures() -> None:
    breaker = CircuitBreaker(
        "test",
        CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout_seconds=60, window_seconds=60),
    )

    async def _fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(_fail)


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_and_closes_after_success() -> None:
    breaker = CircuitBreaker(
        "test",
        CircuitBreakerConfig(failure_threshold=1, success_threshold=1, timeout_seconds=1, window_seconds=60),
    )

    async def _fail() -> None:
        raise RuntimeError("boom")

    async def _ok() -> str:
        return "ok"

    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN

    await asyncio.sleep(1.1)
    result = await breaker.call(_ok)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED
