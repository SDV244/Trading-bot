"""
System State Endpoints

Endpoints for managing system state (running, paused, emergency stop).
"""

from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.security.auth import AuthUser, require_min_role
from packages.adapters.telegram_bot import get_telegram_notifier
from packages.adapters.webhook_notifier import get_webhook_notifier
from packages.core.ai import get_emergency_stop_analyzer
from packages.core.audit import log_event
from packages.core.config import AuthRole, get_settings, reload_settings
from packages.core.database.session import get_session, init_database
from packages.core.observability import increment_system_state
from packages.core.scheduler import get_trading_scheduler
from packages.core.state import get_state_manager
from packages.core.trading_cycle import DataReadiness, get_trading_cycle_service

router = APIRouter()


class StateResponse(BaseModel):
    """System state response."""

    state: str
    reason: str
    changed_at: datetime
    changed_by: str
    can_trade: bool


class StateTransitionRequest(BaseModel):
    """Request to change system state."""

    action: str = Field(..., description="Action: pause, resume, emergency_stop, manual_resume")
    reason: str = Field(..., min_length=1, description="Reason for state change")
    changed_by: str = Field(default="api", description="Who initiated the change")


class EmergencyStopRequest(BaseModel):
    """Emergency stop request."""

    reason: str = Field(..., min_length=1, description="Reason for emergency stop")
    changed_by: str = Field(default="api", description="Who initiated the stop")


class SchedulerResponse(BaseModel):
    """Scheduler status response."""

    running: bool
    interval_seconds: int
    last_run_at: datetime | None
    last_error: str | None
    last_result: dict[str, Any] | None


class TimeframeReadinessResponse(BaseModel):
    """Per-timeframe readiness details."""

    required: int
    available: int
    ready: bool


class SystemReadinessResponse(BaseModel):
    """Readiness summary for paper trading startup."""

    ready: bool
    state: str
    can_trade: bool
    scheduler_running: bool
    data_ready: bool
    require_data_ready: bool
    active_strategy: str
    reasons: list[str]
    timeframes: dict[str, TimeframeReadinessResponse]


class NotificationStatusResponse(BaseModel):
    """Telegram notification configuration/availability status."""

    enabled: bool
    has_bot_token: bool
    has_chat_id: bool


class NotificationTestRequest(BaseModel):
    """Request payload for sending a test notification."""

    title: str = Field(default="Trading Bot test notification", min_length=1, max_length=120)
    body: str = Field(default="Test message from dashboard", min_length=1, max_length=500)


class NotificationTestResponse(BaseModel):
    """Result of test notification send operation."""

    delivered: bool
    message: str


class CircuitBreakerStatusItem(BaseModel):
    """Circuit breaker diagnostic snapshot."""

    name: str
    available: bool
    details: dict[str, Any] | None = None
    reason: str | None = None


class CircuitBreakerStatusResponse(BaseModel):
    """All circuit breaker statuses."""

    breakers: dict[str, CircuitBreakerStatusItem]


class CircuitBreakerResetResponse(BaseModel):
    """Circuit breaker reset result."""

    name: str
    reset: bool
    details: dict[str, Any]


class ReloadConfigResponse(BaseModel):
    """Runtime config reload result."""

    reloaded: bool
    live_mode: bool
    active_strategy: str
    require_data_ready: bool
    min_cycle_interval_seconds: int
    message: str


class ReconciliationResponse(BaseModel):
    """Balance reconciliation response."""

    mode: str
    db_equity: str
    exchange_equity: str | None
    difference: str
    within_warning_tolerance: bool
    within_critical_tolerance: bool
    reason: str
    emergency_stop_triggered: bool


def _serialize_readiness(readiness: DataReadiness) -> dict[str, TimeframeReadinessResponse]:
    return {
        timeframe: TimeframeReadinessResponse(
            required=status.required,
            available=status.available,
            ready=status.ready,
        )
        for timeframe, status in readiness.timeframes.items()
    }


@router.get("/state", response_model=StateResponse)
async def get_state(_: AuthUser = Depends(require_min_role(AuthRole.VIEWER))) -> StateResponse:
    """Get current system state."""
    manager = get_state_manager()
    current = manager.current
    return StateResponse(
        state=current.state.value,
        reason=current.reason,
        changed_at=current.changed_at,
        changed_by=current.changed_by,
        can_trade=manager.can_trade,
    )


@router.post("/state", response_model=StateResponse)
async def change_state(
    request: StateTransitionRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> StateResponse:
    """Change system state."""
    await init_database()
    manager = get_state_manager()

    actor = user.username
    try:
        if request.action == "pause":
            manager.pause(request.reason, actor)
        elif request.action == "resume":
            manager.resume(request.reason, actor)
        elif request.action == "emergency_stop":
            manager.force_emergency_stop(request.reason, actor)
        elif request.action == "manual_resume":
            if user.role != AuthRole.ADMIN:
                raise HTTPException(
                    status_code=403,
                    detail="Only admin can perform manual_resume after emergency stop",
                )
            manager.manual_resume(request.reason, actor)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. "
                "Valid actions: pause, resume, emergency_stop, manual_resume",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    current = manager.current
    async with get_session() as session:
        await log_event(
            session,
            event_type="system_state_changed",
            event_category="system",
            summary=f"State changed to {current.state.value}: {request.reason}",
            details={
                "state": current.state.value,
                "reason": current.reason,
                "changed_by": current.changed_by,
                "requested_changed_by": request.changed_by,
            },
            actor=actor,
        )
        if request.action == "emergency_stop":
            with suppress(Exception):
                await get_emergency_stop_analyzer().analyze_and_enqueue(
                    session,
                    reason=request.reason,
                    source="system_state_transition",
                    metadata={
                        "requested_changed_by": request.changed_by,
                        "changed_by": actor,
                    },
                    actor=actor,
                )
    increment_system_state(current.state.value)
    await get_telegram_notifier().send_info(
        "System state changed",
        f"State: {current.state.value}\nReason: {current.reason}\nBy: {current.changed_by}",
    )
    await get_webhook_notifier().send_info(
        "System state changed",
        f"State: {current.state.value}\nReason: {current.reason}\nBy: {current.changed_by}",
    )
    return StateResponse(
        state=current.state.value,
        reason=current.reason,
        changed_at=current.changed_at,
        changed_by=current.changed_by,
        can_trade=manager.can_trade,
    )


@router.post("/emergency-stop", response_model=StateResponse)
async def emergency_stop(
    request: EmergencyStopRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> StateResponse:
    """Trigger emergency stop."""
    await init_database()
    manager = get_state_manager()
    actor = user.username
    manager.force_emergency_stop(request.reason, actor)

    current = manager.current
    async with get_session() as session:
        await log_event(
            session,
            event_type="emergency_stop_triggered",
            event_category="system",
            summary=f"Emergency stop triggered: {request.reason}",
            details={
                "reason": request.reason,
                "changed_by": actor,
                "requested_changed_by": request.changed_by,
            },
            actor=actor,
        )
        with suppress(Exception):
            await get_emergency_stop_analyzer().analyze_and_enqueue(
                session,
                reason=request.reason,
                source="system_emergency_endpoint",
                metadata={"requested_changed_by": request.changed_by, "changed_by": actor},
                actor=actor,
            )
    increment_system_state(current.state.value)
    await get_telegram_notifier().send_critical_alert(
        "Emergency stop triggered",
        f"Reason: {request.reason}\nBy: {actor}",
    )
    await get_webhook_notifier().send_critical_alert(
        "Emergency stop triggered",
        f"Reason: {request.reason}\nBy: {actor}",
    )
    return StateResponse(
        state=current.state.value,
        reason=current.reason,
        changed_at=current.changed_at,
        changed_by=current.changed_by,
        can_trade=manager.can_trade,
    )


@router.get("/state/history")
async def get_state_history(
    limit: int = 20,
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[dict[str, Any]]:
    """Get state change history."""
    manager = get_state_manager()
    history = manager.get_history(limit)
    return [
        {
            "state": s.state.value,
            "reason": s.reason,
            "changed_at": s.changed_at.isoformat(),
            "changed_by": s.changed_by,
            "metadata": s.metadata,
        }
        for s in history
    ]


@router.get("/scheduler", response_model=SchedulerResponse)
async def get_scheduler_status(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> SchedulerResponse:
    """Get scheduler status."""
    status = get_trading_scheduler().status()
    return SchedulerResponse(
        running=status.running,
        interval_seconds=status.interval_seconds,
        last_run_at=status.last_run_at,
        last_error=status.last_error,
        last_result=status.last_result,
    )


@router.get("/readiness", response_model=SystemReadinessResponse)
async def get_system_readiness(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> SystemReadinessResponse:
    """Get system + data readiness status for paper trading."""
    await init_database()
    manager = get_state_manager()
    scheduler = get_trading_scheduler()
    async with get_session() as session:
        readiness = await get_trading_cycle_service().get_data_readiness(session)

    ready = readiness.data_ready if readiness.require_data_ready else True
    reasons = list(readiness.reasons)
    if manager.current.state.value != "running":
        reasons.append(f"system state is {manager.current.state.value}")
    if not manager.can_trade:
        ready = False

    return SystemReadinessResponse(
        ready=ready,
        state=manager.current.state.value,
        can_trade=manager.can_trade,
        scheduler_running=scheduler.running,
        data_ready=readiness.data_ready,
        require_data_ready=readiness.require_data_ready,
        active_strategy=readiness.active_strategy,
        reasons=reasons,
        timeframes=_serialize_readiness(readiness),
    )


@router.post("/scheduler/start", response_model=SchedulerResponse)
async def start_scheduler(
    interval_seconds: int = Query(default=60, ge=5),
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> SchedulerResponse:
    """Start background scheduler."""
    await init_database()
    settings = get_settings()
    async with get_session() as session:
        readiness = await get_trading_cycle_service().get_data_readiness(session)
    if settings.trading.require_data_ready and not readiness.data_ready:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Cannot start scheduler: insufficient strategy warmup data",
                "data_ready": readiness.data_ready,
                "active_strategy": readiness.active_strategy,
                "reasons": readiness.reasons,
                "timeframes": {
                    tf: {
                        "required": status.required,
                        "available": status.available,
                        "ready": status.ready,
                    }
                    for tf, status in readiness.timeframes.items()
                },
            },
        )

    scheduler = get_trading_scheduler()
    scheduler.start(interval_seconds=interval_seconds)
    status = scheduler.status()
    return SchedulerResponse(
        running=status.running,
        interval_seconds=status.interval_seconds,
        last_run_at=status.last_run_at,
        last_error=status.last_error,
        last_result=status.last_result,
    )


@router.post("/scheduler/stop", response_model=SchedulerResponse)
async def stop_scheduler(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> SchedulerResponse:
    """Stop background scheduler."""
    scheduler = get_trading_scheduler()
    await scheduler.stop()
    status = scheduler.status()
    return SchedulerResponse(
        running=status.running,
        interval_seconds=status.interval_seconds,
        last_run_at=status.last_run_at,
        last_error=status.last_error,
        last_result=status.last_result,
    )


@router.get("/circuit-breakers/status", response_model=CircuitBreakerStatusResponse)
async def get_circuit_breakers_status(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> CircuitBreakerStatusResponse:
    """Get adapter circuit-breaker states."""
    from packages.adapters.binance_live import get_binance_live_adapter
    from packages.adapters.binance_spot import get_binance_adapter

    settings = get_settings()
    spot_stats = get_binance_adapter().get_circuit_breaker_stats()
    breakers: dict[str, CircuitBreakerStatusItem] = {
        "binance_spot": CircuitBreakerStatusItem(
            name="binance_spot",
            available=True,
            details=spot_stats,
            reason=None,
        )
    }

    has_live_credentials = bool(settings.binance.api_key and settings.binance.api_secret)
    if has_live_credentials:
        try:
            live_stats = get_binance_live_adapter().get_circuit_breaker_stats()
            breakers["binance_live"] = CircuitBreakerStatusItem(
                name="binance_live",
                available=True,
                details=live_stats,
                reason=None,
            )
        except Exception as e:
            breakers["binance_live"] = CircuitBreakerStatusItem(
                name="binance_live",
                available=False,
                details=None,
                reason=str(e),
            )
    else:
        breakers["binance_live"] = CircuitBreakerStatusItem(
            name="binance_live",
            available=False,
            details=None,
            reason="BINANCE_API_KEY/BINANCE_API_SECRET are not configured",
        )

    return CircuitBreakerStatusResponse(breakers=breakers)


@router.post("/circuit-breakers/{breaker_name}/reset", response_model=CircuitBreakerResetResponse)
async def reset_circuit_breaker(
    breaker_name: str,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> CircuitBreakerResetResponse:
    """Reset one circuit breaker to CLOSED state."""
    from packages.adapters.binance_live import get_binance_live_adapter
    from packages.adapters.binance_spot import get_binance_adapter

    await init_database()
    settings = get_settings()
    normalized = breaker_name.strip().lower()

    if normalized == "binance_spot":
        spot_adapter = get_binance_adapter()
        await spot_adapter.reset_circuit_breaker()
        details = spot_adapter.get_circuit_breaker_stats()
    elif normalized == "binance_live":
        if not settings.binance.api_key or not settings.binance.api_secret:
            raise HTTPException(
                status_code=400,
                detail="Cannot reset binance_live breaker without BINANCE_API_KEY/BINANCE_API_SECRET",
            )
        live_adapter = get_binance_live_adapter()
        await live_adapter.reset_circuit_breaker()
        details = live_adapter.get_circuit_breaker_stats()
    else:
        raise HTTPException(status_code=404, detail=f"Unknown breaker: {breaker_name}")

    async with get_session() as session:
        await log_event(
            session,
            event_type="circuit_breaker_reset",
            event_category="system",
            summary=f"Circuit breaker reset: {normalized}",
            details={"breaker": normalized, "state": details.get("state")},
            actor=user.username,
        )

    return CircuitBreakerResetResponse(name=normalized, reset=True, details=details)


@router.post("/config/reload", response_model=ReloadConfigResponse)
async def reload_runtime_config(
    user: AuthUser = Depends(require_min_role(AuthRole.ADMIN)),
) -> ReloadConfigResponse:
    """Reload environment configuration with safe guards."""
    from packages.core.trading_cycle import reset_trading_cycle_service

    previous = get_settings()
    previous_live_mode = previous.trading.live_mode
    previous_active_strategy = previous.trading.active_strategy

    reloaded = reload_settings()
    if reloaded.trading.live_mode != previous_live_mode:
        # Revert disallowed runtime live-mode toggle.
        previous.trading.live_mode = previous_live_mode
        reloaded.trading.live_mode = previous_live_mode
        message = "Reloaded with safeguard: runtime live_mode changes require restart"
    else:
        message = "Configuration reloaded"

    reset_trading_cycle_service()

    await init_database()
    async with get_session() as session:
        await log_event(
            session,
            event_type="config_reloaded",
            event_category="config",
            summary="Runtime configuration reloaded",
            details={
                "previous_active_strategy": previous_active_strategy,
                "active_strategy": reloaded.trading.active_strategy,
                "live_mode": reloaded.trading.live_mode,
                "require_data_ready": reloaded.trading.require_data_ready,
                "min_cycle_interval_seconds": reloaded.trading.min_cycle_interval_seconds,
            },
            actor=user.username,
        )

    return ReloadConfigResponse(
        reloaded=True,
        live_mode=reloaded.trading.live_mode,
        active_strategy=reloaded.trading.active_strategy,
        require_data_ready=reloaded.trading.require_data_ready,
        min_cycle_interval_seconds=reloaded.trading.min_cycle_interval_seconds,
        message=message,
    )


@router.get("/reconciliation", response_model=ReconciliationResponse)
async def run_balance_reconciliation(
    warning_tolerance: float = Query(default=1.0, ge=0),
    critical_tolerance: float = Query(default=100.0, ge=0),
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> ReconciliationResponse:
    """Run one balance reconciliation check (paper/live)."""
    from packages.core.reconciliation import run_reconciliation_guard

    await init_database()
    async with get_session() as session:
        outcome = await run_reconciliation_guard(
            session,
            warning_tolerance=Decimal(str(warning_tolerance)),
            critical_tolerance=Decimal(str(critical_tolerance)),
            actor=user.username,
        )
    result = outcome.result

    return ReconciliationResponse(
        mode=result.mode,
        db_equity=str(result.db_equity),
        exchange_equity=str(result.exchange_equity) if result.exchange_equity is not None else None,
        difference=str(result.difference),
        within_warning_tolerance=result.within_warning_tolerance,
        within_critical_tolerance=result.within_critical_tolerance,
        reason=result.reason,
        emergency_stop_triggered=outcome.emergency_stop_triggered,
    )


@router.get("/notifications/status", response_model=NotificationStatusResponse)
async def get_notification_status(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> NotificationStatusResponse:
    """Get Telegram notification readiness without exposing secrets."""
    notifier = get_telegram_notifier()
    return NotificationStatusResponse(
        enabled=notifier.enabled,
        has_bot_token=bool(notifier.bot_token),
        has_chat_id=bool(notifier.chat_id),
    )


@router.post("/notifications/test", response_model=NotificationTestResponse)
async def send_test_notification(
    request: NotificationTestRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> NotificationTestResponse:
    """Send a test Telegram message to verify notification wiring."""
    await init_database()
    notifier = get_telegram_notifier()
    if not notifier.enabled:
        return NotificationTestResponse(
            delivered=False,
            message="Telegram notifications are disabled (missing TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID).",
        )

    body = f"{request.body}\nAt: {datetime.now(UTC).isoformat()}\nBy: {user.username}"
    delivered = await notifier.send_info(request.title, body)

    async with get_session() as session:
        await log_event(
            session,
            event_type="notification_test_sent",
            event_category="system",
            summary="Operator triggered Telegram test notification",
            details={
                "title": request.title,
                "delivered": delivered,
            },
            actor=user.username,
        )

    if delivered:
        return NotificationTestResponse(delivered=True, message="Test notification delivered.")
    return NotificationTestResponse(
        delivered=False,
        message="Telegram send attempt failed (token/chat may be invalid or bot cannot reach chat).",
    )
