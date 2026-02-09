"""
System State Endpoints

Endpoints for managing system state (running, paused, emergency stop).
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.security.auth import AuthUser, require_min_role
from packages.adapters.telegram_bot import get_telegram_notifier
from packages.core.audit import log_event
from packages.core.config import AuthRole, get_settings
from packages.core.database.session import get_session, init_database
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
    await get_telegram_notifier().send_info(
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
    await get_telegram_notifier().send_critical_alert(
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
