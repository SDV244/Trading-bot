"""
System State Endpoints

Endpoints for managing system state (running, paused, emergency stop).
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.security.auth import AuthUser, require_min_role
from packages.adapters.telegram_bot import get_telegram_notifier
from packages.core.audit import log_event
from packages.core.config import AuthRole
from packages.core.database.session import get_session, init_database
from packages.core.scheduler import get_trading_scheduler
from packages.core.state import get_state_manager

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


@router.post("/scheduler/start", response_model=SchedulerResponse)
async def start_scheduler(
    interval_seconds: int = Query(default=60, ge=5),
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> SchedulerResponse:
    """Start background scheduler."""
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
