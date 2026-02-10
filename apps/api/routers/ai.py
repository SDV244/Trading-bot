"""AI approval and audit endpoints."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from apps.api.security.auth import AuthUser, require_min_role
from packages.core.ai import ApprovalGateError, get_ai_advisor, get_approval_gate
from packages.core.audit import log_event
from packages.core.config import AuthRole, apply_runtime_config_patch, get_settings
from packages.core.database.models import Approval, EquitySnapshot, EventLog
from packages.core.database.session import get_session, init_database
from packages.core.observability import increment_approval

if TYPE_CHECKING:
    from packages.core.ai.drl_optimizer import DRLOptimizer

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


class AutoApproveStatusResponse(BaseModel):
    """AI auto-approval runtime status."""

    enabled: bool
    decided_by: str


class AutoApproveUpdateRequest(BaseModel):
    """Request payload for AI auto-approval toggle."""

    enabled: bool
    reason: str = Field(default="dashboard_toggle")
    changed_by: str = Field(default="web_dashboard")


class LLMStatusResponse(BaseModel):
    """LLM advisor runtime status."""

    enabled: bool
    provider: str
    model: str
    configured: bool
    base_url: str | None
    prefer_llm: bool
    fallback_to_rules: bool
    last_error: str | None


class LLMTestResponse(BaseModel):
    """LLM connectivity test result."""

    ok: bool
    provider: str
    model: str
    latency_ms: int | None
    message: str
    raw_proposals_count: int | None = None


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
            approval = await gate.create_approval(session, proposal)
            created.append(approval)
            if approval.status == "APPROVED":
                increment_approval("auto_approved")
            else:
                increment_approval("created")
        return [_to_approval_response(a) for a in created]


@router.get("/llm/status", response_model=LLMStatusResponse)
async def get_llm_status(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> LLMStatusResponse:
    """Get configured LLM advisor status."""
    status = get_ai_advisor().llm_status()
    return LLMStatusResponse(**status)


@router.post("/llm/test", response_model=LLMTestResponse)
async def test_llm_connection(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> LLMTestResponse:
    """Run a lightweight LLM connectivity test using current provider settings."""
    result = await get_ai_advisor().test_llm_connection()
    return LLMTestResponse(**result)


@router.get("/auto-approve", response_model=AutoApproveStatusResponse)
async def get_auto_approve_status(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> AutoApproveStatusResponse:
    """Get current AI auto-approval mode."""
    settings = get_settings()
    return AutoApproveStatusResponse(
        enabled=settings.approval.auto_approve_enabled,
        decided_by="ai_auto_approver",
    )


@router.post("/auto-approve", response_model=AutoApproveStatusResponse)
async def set_auto_approve_status(
    request: AutoApproveUpdateRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> AutoApproveStatusResponse:
    """Enable/disable automatic approval of AI proposals."""
    await init_database()
    gate = get_approval_gate()
    applied = apply_runtime_config_patch({"approval": {"auto_approve_enabled": request.enabled}})
    auto_approved_pending = 0
    auto_expired_pending = 0
    async with get_session() as session:
        if request.enabled:
            sweep = await gate.auto_approve_pending(session, decided_by="ai_auto_approver")
            auto_approved_pending = len(sweep.approved)
            auto_expired_pending = len(sweep.expired)
            for _ in sweep.approved:
                increment_approval("auto_approved")
            for _ in sweep.expired:
                increment_approval("expired")
        await log_event(
            session,
            event_type="ai_auto_approve_toggled",
            event_category="config",
            summary=f"AI auto-approve set to {request.enabled}",
            details={
                "enabled": request.enabled,
                "applied_keys": applied.get("approval", []),
                "reason": request.reason,
                "auto_approved_pending_count": auto_approved_pending,
                "auto_expired_pending_count": auto_expired_pending,
            },
            actor=request.changed_by or user.username,
        )
    settings = get_settings()
    return AutoApproveStatusResponse(
        enabled=settings.approval.auto_approve_enabled,
        decided_by="ai_auto_approver",
    )


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
        increment_approval("approved")
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
        increment_approval("rejected")
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
        if expired:
            increment_approval("expired")
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


_optimizer: "DRLOptimizer | None" = None


def _get_optimizer() -> "DRLOptimizer":
    from packages.core.ai.drl_optimizer import DRLOptimizer

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
