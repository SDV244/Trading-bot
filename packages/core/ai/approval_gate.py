"""Approval gate for AI proposals."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.telegram_bot import get_telegram_notifier
from packages.core.ai.advisor import AIProposal
from packages.core.audit import log_event
from packages.core.database.models import Approval, ApprovalStatus, Config
from packages.core.state import get_state_manager


class ApprovalGateError(ValueError):
    """Raised when approval operations fail validation."""


class ApprovalGate:
    """Create and resolve AI proposals with strict TTL safety."""

    async def create_approval(
        self,
        session: AsyncSession,
        proposal: AIProposal,
        *,
        actor: str = "ai_advisor",
    ) -> Approval:
        approval = Approval(
            proposal_type=proposal.proposal_type,
            title=proposal.title,
            description=proposal.description,
            diff=proposal.diff,
            expected_impact=proposal.expected_impact,
            evidence=proposal.evidence,
            confidence=proposal.confidence,
            status=ApprovalStatus.PENDING.value,
            ttl_hours=proposal.ttl_hours,
            expires_at=_utc_now_naive() + timedelta(hours=proposal.ttl_hours),
        )
        session.add(approval)
        await session.flush()

        await log_event(
            session,
            event_type="approval_created",
            event_category="ai",
            summary=f"Approval #{approval.id} created: {approval.title}",
            details={
                "approval_id": approval.id,
                "proposal_type": approval.proposal_type,
                "expires_at": _to_utc(approval.expires_at).isoformat(),
                "confidence": approval.confidence,
            },
            inputs={"diff": approval.diff},
            actor=actor,
        )

        notifier = get_telegram_notifier()
        await notifier.send_info(
            "AI approval created",
            f"#{approval.id} {approval.title}\nExpires at: {approval.expires_at.isoformat()}",
        )
        return approval

    async def list_approvals(
        self,
        session: AsyncSession,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Approval]:
        query = select(Approval).order_by(Approval.created_at.desc()).limit(limit)
        if status:
            query = query.where(Approval.status == status.upper())
        result = await session.execute(query)
        return list(result.scalars().all())

    async def approve(
        self,
        session: AsyncSession,
        approval_id: int,
        *,
        decided_by: str,
    ) -> Approval:
        approval = await self._get_pending_approval(session, approval_id)
        if _to_utc(approval.expires_at) <= _to_utc(_utc_now_naive()):
            approval.status = ApprovalStatus.EXPIRED.value
            approval.decided_by = decided_by
            approval.decided_at = _utc_now_naive()
            raise ApprovalGateError("Approval expired")

        await self._apply_config_diff(session, approval.diff)
        approval.status = ApprovalStatus.APPROVED.value
        approval.decided_by = decided_by
        approval.decided_at = _utc_now_naive()

        await log_event(
            session,
            event_type="approval_approved",
            event_category="ai",
            summary=f"Approval #{approval.id} approved",
            details={"approval_id": approval.id, "title": approval.title},
            inputs={"diff": approval.diff},
            actor=decided_by,
        )

        notifier = get_telegram_notifier()
        await notifier.send_info("AI approval approved", f"#{approval.id} {approval.title}")
        return approval

    async def reject(
        self,
        session: AsyncSession,
        approval_id: int,
        *,
        decided_by: str,
        reason: str = "",
    ) -> Approval:
        approval = await self._get_pending_approval(session, approval_id)
        approval.status = ApprovalStatus.REJECTED.value
        approval.decided_by = decided_by
        approval.decided_at = _utc_now_naive()

        await log_event(
            session,
            event_type="approval_rejected",
            event_category="ai",
            summary=f"Approval #{approval.id} rejected",
            details={"approval_id": approval.id, "reason": reason},
            inputs={"diff": approval.diff},
            actor=decided_by,
        )
        return approval

    async def expire_pending(self, session: AsyncSession) -> list[Approval]:
        now = _utc_now_naive()
        result = await session.execute(
            select(Approval).where(
                Approval.status == ApprovalStatus.PENDING.value,
                Approval.expires_at <= now,
            )
        )
        expired = list(result.scalars().all())
        if not expired:
            return []

        for approval in expired:
            approval.status = ApprovalStatus.EXPIRED.value
            approval.decided_by = "system"
            approval.decided_at = now

            await log_event(
                session,
                event_type="approval_expired",
                event_category="ai",
                summary=f"Approval #{approval.id} expired",
                details={"approval_id": approval.id, "title": approval.title},
                inputs={"diff": approval.diff},
                actor="system",
            )

        # Safety rule: timeout forces emergency stop until manual intervention.
        manager = get_state_manager()
        manager.force_emergency_stop(
            reason=f"Approval timeout ({len(expired)} expired)",
            changed_by="approval_gate",
            metadata={"expired_approvals": [a.id for a in expired]},
        )
        await log_event(
            session,
            event_type="emergency_stop_triggered",
            event_category="system",
            summary="Emergency stop due to expired approvals",
            details={"expired_approvals": [a.id for a in expired]},
            actor="approval_gate",
        )

        notifier = get_telegram_notifier()
        await notifier.send_critical_alert(
            "Emergency stop: approval timeout",
            f"{len(expired)} approval(s) expired and system entered EMERGENCY_STOP.",
        )
        return expired

    async def _get_pending_approval(self, session: AsyncSession, approval_id: int) -> Approval:
        result = await session.execute(select(Approval).where(Approval.id == approval_id))
        approval = result.scalar_one_or_none()
        if approval is None:
            raise ApprovalGateError(f"Approval #{approval_id} not found")
        if approval.status != ApprovalStatus.PENDING.value:
            raise ApprovalGateError(f"Approval #{approval_id} is not pending")
        return approval

    async def _apply_config_diff(self, session: AsyncSession, diff: dict[str, Any]) -> Config:
        current = await self._get_or_create_active_config(session)
        new_data = _deep_merge(dict(current.data), diff)
        current.active = False
        new_config = Config(version=current.version + 1, data=new_data, active=True)
        session.add(new_config)
        await session.flush()
        return new_config

    async def _get_or_create_active_config(self, session: AsyncSession) -> Config:
        result = await session.execute(
            select(Config).where(Config.active.is_(True)).order_by(Config.version.desc()).limit(1)
        )
        active = result.scalars().first()
        if active is not None:
            return active

        bootstrap = Config(version=1, data={}, active=True)
        session.add(bootstrap)
        await session.flush()
        return bootstrap


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _utc_now_naive() -> datetime:
    """Get UTC timestamp without tzinfo for SQLite compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)


def _to_utc(value: datetime) -> datetime:
    """Normalize datetime to UTC-aware for safe comparison and serialization."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_approval_gate: ApprovalGate | None = None
_approval_gate_lock = threading.Lock()


def get_approval_gate() -> ApprovalGate:
    """Get or create singleton approval gate."""
    global _approval_gate
    if _approval_gate is None:
        with _approval_gate_lock:
            if _approval_gate is None:
                _approval_gate = ApprovalGate()
    return _approval_gate
