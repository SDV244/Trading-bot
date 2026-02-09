"""AI approval and audit endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from apps.api.security.auth import AuthUser, require_min_role
from packages.core.config import AuthRole

from packages.core.ai import ApprovalGateError, DRLOptimizer, get_ai_advisor, get_approval_gate
from packages.core.database.models import Approval, EquitySnapshot, EventLog
from packages.core.database.session import get_session, init_database

router = APIRouter()


class ApprovalResponse(BaseModel):
    """Approval record response."""

    id: int
    proposal_type: str
    title: str
    description: str
    diff: dict[str, Any]
    expected_impact: str | None
    evidence: dict[str, Any] | None
    confidence: float
    status: str
    ttl_hours: int
    expires_at: datetime
    decided_by: str | None
    decided_at: datetime | None
    created_at: datetime


class DecisionRequest(BaseModel):
    """Approval decision request."""

    decided_by: str = Field(..., min_length=1)
    reason: str = ""


class EventResponse(BaseModel):
    """Audit event response."""

    id: int
    event_type: str
    event_category: str
    summary: str
    details: dict[str, Any] | None
    inputs: dict[str, Any] | None
    config_version: int
    actor: str
    created_at: datetime


class OptimizerTrainRequest(BaseModel):
    """Request to train DRL optimizer."""

    timesteps: int = Field(default=1024, ge=256, le=50_000)


class OptimizerTrainResponse(BaseModel):
    """Train status response."""

    trained: bool
    data_points: int
    timesteps: int


class OptimizerProposalResponse(BaseModel):
    """DRL proposal response."""

    title: str
    diff: dict[str, Any]
    expected_impact: str
    evidence: dict[str, Any]
    confidence: float


def _to_approval_response(approval: Approval) -> ApprovalResponse:
    return ApprovalResponse(
        id=approval.id,
        proposal_type=approval.proposal_type,
        title=approval.title,
        description=approval.description,
        diff=approval.diff,
        expected_impact=approval.expected_impact,
        evidence=approval.evidence,
        confidence=approval.confidence,
        status=approval.status,
        ttl_hours=approval.ttl_hours,
        expires_at=approval.expires_at,
        decided_by=approval.decided_by,
        decided_at=approval.decided_at,
        created_at=approval.created_at,
    )


@router.post("/proposals/generate", response_model=list[ApprovalResponse])
async def generate_proposals(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> list[ApprovalResponse]:
    """Generate advisor proposals and enqueue them for approval."""
    await init_database()
    gate = get_approval_gate()
    advisor = get_ai_advisor()

    async with get_session() as session:
        proposals = await advisor.generate_proposals(session)
        created: list[Approval] = []
        for proposal in proposals:
            created.append(await gate.create_approval(session, proposal))
        return [_to_approval_response(a) for a in created]


@router.get("/approvals", response_model=list[ApprovalResponse])
async def list_approvals(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[ApprovalResponse]:
    """List approval records."""
    await init_database()
    gate = get_approval_gate()
    async with get_session() as session:
        approvals = await gate.list_approvals(session, status=status, limit=limit)
        return [_to_approval_response(a) for a in approvals]


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalResponse)
async def approve_proposal(
    approval_id: int,
    request: DecisionRequest,
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> ApprovalResponse:
    """Approve a pending proposal."""
    await init_database()
    gate = get_approval_gate()
    async with get_session() as session:
        try:
            approval = await gate.approve(session, approval_id, decided_by=request.decided_by)
        except ApprovalGateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _to_approval_response(approval)


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalResponse)
async def reject_proposal(
    approval_id: int,
    request: DecisionRequest,
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> ApprovalResponse:
    """Reject a pending proposal."""
    await init_database()
    gate = get_approval_gate()
    async with get_session() as session:
        try:
            approval = await gate.reject(
                session,
                approval_id,
                decided_by=request.decided_by,
                reason=request.reason,
            )
        except ApprovalGateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _to_approval_response(approval)


@router.post("/approvals/expire-check", response_model=dict[str, int])
async def expire_check(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> dict[str, int]:
    """Manually trigger expiry check for pending approvals."""
    await init_database()
    gate = get_approval_gate()
    async with get_session() as session:
        expired = await gate.expire_pending(session)
        return {"expired_count": len(expired)}


@router.get("/events", response_model=list[EventResponse])
async def list_events(
    category: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    search: str | None = Query(default=None),
    start_at: datetime | None = Query(default=None),
    end_at: datetime | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[EventResponse]:
    """List audit events."""
    await init_database()
    async with get_session() as session:
        query = select(EventLog).order_by(EventLog.created_at.desc()).limit(limit)
        if category:
            query = query.where(EventLog.event_category == category)
        if event_type:
            query = query.where(EventLog.event_type == event_type)
        if actor:
            query = query.where(EventLog.actor == actor)
        if start_at:
            query = query.where(EventLog.created_at >= start_at)
        if end_at:
            query = query.where(EventLog.created_at <= end_at)
        if search:
            search_pattern = f"%{search}%"
            query = query.where(EventLog.summary.ilike(search_pattern))
        result = await session.execute(query)
        events = result.scalars().all()

    return [
        EventResponse(
            id=e.id,
            event_type=e.event_type,
            event_category=e.event_category,
            summary=e.summary,
            details=e.details,
            inputs=e.inputs,
            config_version=e.config_version,
            actor=e.actor,
            created_at=e.created_at,
        )
        for e in events
    ]


_optimizer: DRLOptimizer | None = None


def _get_optimizer() -> DRLOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = DRLOptimizer()
    return _optimizer


@router.post("/optimizer/train", response_model=OptimizerTrainResponse)
async def train_optimizer(
    request: OptimizerTrainRequest,
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> OptimizerTrainResponse:
    """Train PPO optimizer on recent equity-return series."""
    await init_database()
    async with get_session() as session:
        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.is_paper.is_(True))
            .order_by(EquitySnapshot.snapshot_at.asc())
            .limit(800)
        )
        snapshots = list(result.scalars().all())

    if len(snapshots) < 40:
        raise HTTPException(status_code=400, detail="Not enough equity snapshots to train optimizer")

    equity = [float(s.equity) for s in snapshots]
    rewards: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        curr = equity[i]
        rewards.append((curr - prev) / prev if prev else 0.0)

    optimizer = _get_optimizer()
    optimizer.train(reward_series=rewards, timesteps=request.timesteps)
    return OptimizerTrainResponse(trained=True, data_points=len(rewards), timesteps=request.timesteps)


@router.get("/optimizer/propose", response_model=OptimizerProposalResponse)
async def optimizer_propose(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> OptimizerProposalResponse:
    """Generate DRL proposal from current trained model."""
    await init_database()
    async with get_session() as session:
        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.is_paper.is_(True))
            .order_by(EquitySnapshot.snapshot_at.asc())
            .limit(800)
        )
        snapshots = list(result.scalars().all())
    if len(snapshots) < 40:
        raise HTTPException(status_code=400, detail="Not enough equity snapshots to create proposal")

    equity = [float(s.equity) for s in snapshots]
    rewards: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        curr = equity[i]
        rewards.append((curr - prev) / prev if prev else 0.0)

    optimizer = _get_optimizer()
    proposal = optimizer.propose(base_risk_per_trade=0.005, reward_series=rewards, equity_curve=equity)
    if proposal is None:
        raise HTTPException(status_code=400, detail="Optimizer not trained")

    return OptimizerProposalResponse(
        title=proposal.title,
        diff=proposal.diff,
        expected_impact=proposal.expected_impact,
        evidence=proposal.evidence,
        confidence=proposal.confidence,
    )
