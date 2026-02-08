"""
System State Endpoints

Endpoints for managing system state (running, paused, emergency stop).
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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


@router.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
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
async def change_state(request: StateTransitionRequest) -> StateResponse:
    """Change system state."""
    manager = get_state_manager()

    try:
        if request.action == "pause":
            manager.pause(request.reason, request.changed_by)
        elif request.action == "resume":
            manager.resume(request.reason, request.changed_by)
        elif request.action == "emergency_stop":
            manager.force_emergency_stop(request.reason, request.changed_by)
        elif request.action == "manual_resume":
            manager.manual_resume(request.reason, request.changed_by)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. "
                "Valid actions: pause, resume, emergency_stop, manual_resume",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    current = manager.current
    return StateResponse(
        state=current.state.value,
        reason=current.reason,
        changed_at=current.changed_at,
        changed_by=current.changed_by,
        can_trade=manager.can_trade,
    )


@router.post("/emergency-stop", response_model=StateResponse)
async def emergency_stop(request: EmergencyStopRequest) -> StateResponse:
    """Trigger emergency stop."""
    manager = get_state_manager()
    manager.force_emergency_stop(request.reason, request.changed_by)

    current = manager.current
    return StateResponse(
        state=current.state.value,
        reason=current.reason,
        changed_at=current.changed_at,
        changed_by=current.changed_by,
        can_trade=manager.can_trade,
    )


@router.get("/state/history")
async def get_state_history(limit: int = 20) -> list[dict[str, Any]]:
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
