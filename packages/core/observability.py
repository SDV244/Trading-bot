"""Prometheus metrics helpers for runtime observability."""

from __future__ import annotations

import threading
from collections.abc import Mapping

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_metrics_lock = threading.Lock()
_metrics_initialized = False


def _ensure_metrics() -> None:
    global _metrics_initialized
    if _metrics_initialized:
        return
    with _metrics_lock:
        if _metrics_initialized:
            return
        _metrics_initialized = True


_ensure_metrics()

SCHEDULER_CYCLES_TOTAL = Counter(
    "trading_scheduler_cycles_total",
    "Total scheduler cycles executed",
    labelnames=("status",),
)
SCHEDULER_CYCLE_DURATION_SECONDS = Histogram(
    "trading_scheduler_cycle_duration_seconds",
    "Scheduler cycle duration in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
SCHEDULER_CYCLE_INTERVAL_SECONDS = Histogram(
    "trading_scheduler_cycle_interval_seconds",
    "Observed time between scheduler cycles in seconds",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300),
)
SCHEDULER_RUNNING = Gauge(
    "trading_scheduler_running",
    "Whether scheduler is currently running (1/0)",
)
TRADING_LIVE_ORDERS_TOTAL = Counter(
    "trading_live_orders_total",
    "Live order submissions",
    labelnames=("status", "side"),
)
SYSTEM_STATE_TRANSITIONS_TOTAL = Counter(
    "trading_system_state_transitions_total",
    "System state transition count",
    labelnames=("state",),
)
APPROVALS_TOTAL = Counter(
    "trading_approvals_total",
    "AI approval actions",
    labelnames=("action",),
)


def observe_scheduler_cycle(*, duration_seconds: float, status: str, interval_seconds: float | None = None) -> None:
    """Record scheduler cycle metrics."""
    SCHEDULER_CYCLES_TOTAL.labels(status=status).inc()
    SCHEDULER_CYCLE_DURATION_SECONDS.observe(max(0.0, duration_seconds))
    if interval_seconds is not None and interval_seconds >= 0:
        SCHEDULER_CYCLE_INTERVAL_SECONDS.observe(interval_seconds)


def set_scheduler_running(is_running: bool) -> None:
    """Set scheduler running gauge."""
    SCHEDULER_RUNNING.set(1 if is_running else 0)


def increment_live_order(*, accepted: bool, side: str) -> None:
    """Record a live order event."""
    status = "accepted" if accepted else "rejected"
    TRADING_LIVE_ORDERS_TOTAL.labels(status=status, side=side.upper()).inc()


def increment_system_state(state: str) -> None:
    """Record system state transition."""
    SYSTEM_STATE_TRANSITIONS_TOTAL.labels(state=state).inc()


def increment_approval(action: str) -> None:
    """Record approval actions."""
    APPROVALS_TOTAL.labels(action=action).inc()


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Serialize metrics in Prometheus text format."""
    return generate_latest(), CONTENT_TYPE_LATEST


def snapshot_runtime_metrics() -> Mapping[str, float]:
    """Return a minimal metrics snapshot for debug logs/tests."""
    return {
        "scheduler_running": SCHEDULER_RUNNING._value.get(),  # noqa: SLF001
    }
