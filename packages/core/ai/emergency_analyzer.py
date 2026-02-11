"""Automatic AI remediation analysis for emergency-stop incidents."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.ai.advisor import AIProposal
from packages.core.ai.llm_client import get_llm_advisor_client
from packages.core.audit import log_event
from packages.core.config import get_settings
from packages.core.database.models import Approval, ApprovalStatus, EquitySnapshot, EventLog
from packages.core.state import get_state_manager


@dataclass(slots=True, frozen=True)
class EmergencyAnalysisOutcome:
    """Result of one emergency-stop AI analysis run."""

    triggered: bool
    source: str
    reason: str
    used_llm: bool
    llm_error: str | None
    proposals_generated: int
    approvals_created: int
    approvals_auto_approved: int
    proposal_titles: tuple[str, ...]
    approval_ids: tuple[int, ...]


class EmergencyStopAnalyzer:
    """Build emergency context and generate/apply remediation proposals."""

    async def analyze_and_enqueue(
        self,
        session: AsyncSession,
        *,
        reason: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        actor: str = "emergency_ai",
        gate: Any | None = None,
        force: bool = False,
    ) -> EmergencyAnalysisOutcome:
        settings = get_settings()
        if not settings.approval.emergency_ai_enabled and not force:
            return EmergencyAnalysisOutcome(
                triggered=False,
                source=source,
                reason=reason,
                used_llm=False,
                llm_error=None,
                proposals_generated=0,
                approvals_created=0,
                approvals_auto_approved=0,
                proposal_titles=(),
                approval_ids=(),
            )

        payload = metadata or {}
        await log_event(
            session,
            event_type="emergency_ai_analysis_started",
            event_category="ai",
            summary=f"Emergency AI analysis started ({source})",
            details={"source": source, "reason": reason, "metadata": payload},
            actor=actor,
        )

        llm_error: str | None = None
        used_llm = False
        proposals: list[AIProposal] = []

        try:
            context = await self._build_context(
                session=session,
                reason=reason,
                source=source,
                metadata=payload,
            )
            if settings.llm.enabled:
                try:
                    raw = await get_llm_advisor_client().generate_structured(
                        context=context,
                        system_prompt=_emergency_system_prompt(),
                        schema=_emergency_response_schema(),
                    )
                    proposals = self._sanitize_llm_proposals(raw)
                    used_llm = True
                except Exception as exc:  # noqa: BLE001
                    llm_error = _sanitize_error_text(str(exc))

            if not proposals:
                proposals = self._build_fallback_proposals(
                    reason=reason,
                    source=source,
                    metadata=payload,
                    context=context,
                )

            max_items = max(1, settings.approval.emergency_max_proposals)
            proposals = proposals[:max_items]

            created: list[Approval] = []
            auto_approved_count = 0
            if proposals:
                gate_impl = gate
                if gate_impl is None:
                    from packages.core.ai.approval_gate import get_approval_gate

                    gate_impl = get_approval_gate()
                for proposal in proposals:
                    approval = await gate_impl.create_approval(session, proposal, actor=actor)
                    created.append(approval)
                    if approval.status == ApprovalStatus.APPROVED.value:
                        auto_approved_count += 1

            approval_ids = tuple(approval.id for approval in created)
            proposal_titles = tuple(proposal.title for proposal in proposals)

            await log_event(
                session,
                event_type="emergency_ai_analysis_completed",
                event_category="ai",
                summary=f"Emergency AI analysis completed ({source})",
                details={
                    "source": source,
                    "reason": reason,
                    "used_llm": used_llm,
                    "llm_error": llm_error,
                    "proposals_generated": len(proposals),
                    "approvals_created": len(created),
                    "approvals_auto_approved": auto_approved_count,
                    "proposal_titles": list(proposal_titles),
                    "approval_ids": list(approval_ids),
                },
                actor=actor,
            )

            return EmergencyAnalysisOutcome(
                triggered=True,
                source=source,
                reason=reason,
                used_llm=used_llm,
                llm_error=llm_error,
                proposals_generated=len(proposals),
                approvals_created=len(created),
                approvals_auto_approved=auto_approved_count,
                proposal_titles=proposal_titles,
                approval_ids=approval_ids,
            )
        except Exception as exc:  # noqa: BLE001
            await log_event(
                session,
                event_type="emergency_ai_analysis_failed",
                event_category="ai",
                summary=f"Emergency AI analysis failed ({source})",
                details={"source": source, "reason": reason, "error": _sanitize_error_text(str(exc))},
                actor=actor,
            )
            return EmergencyAnalysisOutcome(
                triggered=True,
                source=source,
                reason=reason,
                used_llm=used_llm,
                llm_error=llm_error or _sanitize_error_text(str(exc)),
                proposals_generated=0,
                approvals_created=0,
                approvals_auto_approved=0,
                proposal_titles=(),
                approval_ids=(),
            )

    async def _build_context(
        self,
        *,
        session: AsyncSession,
        reason: str,
        source: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        settings = get_settings()
        manager = get_state_manager()
        current_state = manager.current
        is_paper_mode = not settings.trading.live_mode

        equity_result = await session.execute(
            select(
                EquitySnapshot.equity,
                EquitySnapshot.available_balance,
                EquitySnapshot.unrealized_pnl,
                EquitySnapshot.snapshot_at,
            )
            .where(EquitySnapshot.is_paper.is_(is_paper_mode))
            .order_by(EquitySnapshot.snapshot_at.desc())
            .limit(1)
        )
        latest_equity = equity_result.first()

        pending_result = await session.execute(
            select(func.count()).select_from(Approval).where(Approval.status == ApprovalStatus.PENDING.value)
        )
        pending_approvals = int(pending_result.scalar_one_or_none() or 0)

        recent_events_result = await session.execute(
            select(EventLog)
            .where(EventLog.created_at >= datetime.now(UTC) - timedelta(hours=8))
            .order_by(EventLog.created_at.desc())
            .limit(30)
        )
        recent_events = list(recent_events_result.scalars().all())

        readiness_payload: dict[str, Any] = {}
        try:
            from packages.core.trading_cycle import get_trading_cycle_service

            readiness = await get_trading_cycle_service().get_data_readiness(session)
            readiness_payload = {
                "data_ready": readiness.data_ready,
                "require_data_ready": readiness.require_data_ready,
                "active_strategy": readiness.active_strategy,
                "reasons": readiness.reasons,
                "timeframes": {
                    timeframe: {
                        "required": status.required,
                        "available": status.available,
                        "ready": status.ready,
                    }
                    for timeframe, status in readiness.timeframes.items()
                },
            }
        except Exception as exc:  # noqa: BLE001
            readiness_payload = {"error": str(exc)}

        return {
            "emergency": {
                "source": source,
                "reason": reason,
                "metadata": metadata,
                "triggered_at": datetime.now(UTC).isoformat(),
            },
            "system": {
                "state": current_state.state.value,
                "state_reason": current_state.reason,
                "changed_by": current_state.changed_by,
            },
            "mode": "paper" if is_paper_mode else "live",
            "symbol": settings.trading.pair,
            "active_strategy": settings.trading.active_strategy,
            "risk_settings": {
                "per_trade": settings.risk.per_trade,
                "max_daily_loss": settings.risk.max_daily_loss,
                "max_exposure": settings.risk.max_exposure,
                "fee_bps": settings.risk.fee_bps,
                "slippage_bps": settings.risk.slippage_bps,
            },
            "approval_settings": {
                "timeout_hours": settings.approval.timeout_hours,
                "auto_approve_enabled": settings.approval.auto_approve_enabled,
                "emergency_ai_enabled": settings.approval.emergency_ai_enabled,
                "emergency_max_proposals": settings.approval.emergency_max_proposals,
            },
            "trading_settings": {
                "advisor_interval_cycles": settings.trading.advisor_interval_cycles,
                "grid_min_spacing_bps": settings.trading.grid_min_spacing_bps,
                "grid_max_spacing_bps": settings.trading.grid_max_spacing_bps,
                "grid_recenter_mode": settings.trading.grid_recenter_mode,
                "reconciliation_warning_tolerance": settings.trading.reconciliation_warning_tolerance,
                "reconciliation_critical_tolerance": settings.trading.reconciliation_critical_tolerance,
                "stop_loss_global_equity_pct": settings.trading.stop_loss_global_equity_pct,
                "stop_loss_max_drawdown_pct": settings.trading.stop_loss_max_drawdown_pct,
            },
            "latest_equity": {
                "equity": float(latest_equity[0]) if latest_equity else None,
                "available_balance": float(latest_equity[1]) if latest_equity else None,
                "unrealized_pnl": float(latest_equity[2]) if latest_equity else None,
                "snapshot_at": latest_equity[3].isoformat() if latest_equity and latest_equity[3] else None,
            },
            "pending_approvals": pending_approvals,
            "data_readiness": readiness_payload,
            "recent_events": [
                {
                    "created_at": event.created_at.isoformat(),
                    "event_type": event.event_type,
                    "event_category": event.event_category,
                    "summary": event.summary,
                    "actor": event.actor,
                    "details": event.details,
                }
                for event in recent_events
            ],
        }

    def _sanitize_llm_proposals(self, raw: list[dict[str, Any]]) -> list[AIProposal]:
        settings = get_settings()
        min_confidence = settings.llm.min_confidence
        allowed_types = {
            "emergency_remediation",
            "risk_tuning",
            "execution_tuning",
            "grid_tuning",
            "anomaly_alert",
        }
        max_items = max(1, settings.approval.emergency_max_proposals)

        sanitized: list[AIProposal] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            proposal_type = str(item.get("proposal_type", "")).strip().lower()
            if proposal_type not in allowed_types:
                continue
            confidence = _safe_confidence(item.get("confidence", 0.0))
            if confidence < min_confidence:
                continue
            diff = self._sanitize_diff(item.get("diff"))
            if not diff:
                continue
            title = str(item.get("title", "")).strip()[:180]
            description = str(item.get("description", "")).strip()[:1000]
            if not title or not description:
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {"raw_evidence": str(evidence)[:800]}
            ttl_raw = item.get("ttl_hours", settings.approval.timeout_hours)
            ttl = int(ttl_raw) if isinstance(ttl_raw, int | float) else settings.approval.timeout_hours
            ttl = max(1, min(ttl, settings.approval.timeout_hours))
            expected_impact = str(item.get("expected_impact", "")).strip()[:500]
            if not expected_impact:
                expected_impact = "Emergency remediation proposal based on incident context."
            sanitized.append(
                AIProposal(
                    title=title,
                    proposal_type=proposal_type,
                    description=description,
                    diff=diff,
                    expected_impact=expected_impact,
                    evidence=evidence,
                    confidence=confidence,
                    ttl_hours=ttl,
                )
            )
            if len(sanitized) >= max_items:
                break
        return sanitized

    def _sanitize_diff(self, diff: Any) -> dict[str, Any]:
        if not isinstance(diff, dict):
            return {}
        allowed: dict[str, set[str]] = {
            "risk": {
                "per_trade",
                "min_per_trade",
                "max_per_trade",
                "max_daily_loss",
                "max_exposure",
                "fee_bps",
                "slippage_bps",
            },
            "trading": {
                "advisor_interval_cycles",
                "grid_min_spacing_bps",
                "grid_max_spacing_bps",
                "grid_recenter_mode",
                "reconciliation_warning_tolerance",
                "reconciliation_critical_tolerance",
                "stop_loss_global_equity_pct",
                "stop_loss_max_drawdown_pct",
            },
            "approval": {
                "timeout_hours",
                "auto_approve_enabled",
            },
        }
        string_allowed: dict[str, set[str]] = {
            "trading": {"grid_recenter_mode"},
        }
        cleaned: dict[str, dict[str, Any]] = {}
        for section, value in diff.items():
            if section not in allowed or not isinstance(value, dict):
                continue
            section_clean: dict[str, Any] = {}
            for key, candidate in value.items():
                if key not in allowed[section]:
                    continue
                if isinstance(candidate, bool | int | float):
                    section_clean[key] = candidate
                elif isinstance(candidate, str) and key in string_allowed.get(section, set()):
                    section_clean[key] = candidate[:120]
            if section_clean:
                cleaned[section] = section_clean
        return cleaned

    def _build_fallback_proposals(
        self,
        *,
        reason: str,
        source: str,
        metadata: dict[str, Any],
        context: dict[str, Any],
    ) -> list[AIProposal]:
        settings = get_settings()
        reason_lower = reason.lower()
        proposals: list[AIProposal] = []

        if "approval timeout" in reason_lower or source == "approval_gate":
            if not settings.approval.auto_approve_enabled:
                proposals.append(
                    AIProposal(
                        title="Enable AI auto-approve to prevent timeout deadlocks",
                        proposal_type="emergency_remediation",
                        description=(
                            "Emergency stop was caused by expired approvals. Enabling auto-approve "
                            "prevents future timeout-induced emergency stops for AI proposals."
                        ),
                        diff={"approval": {"auto_approve_enabled": True}},
                        expected_impact="Prevents repeated emergency-stop loops caused by approval expiry.",
                        evidence={
                            "source": source,
                            "reason": reason,
                            "pending_approvals": context.get("pending_approvals"),
                        },
                        confidence=0.74,
                        ttl_hours=min(settings.approval.timeout_hours, 4),
                    )
                )
            proposals.append(
                AIProposal(
                    title="Increase approval timeout window",
                    proposal_type="emergency_remediation",
                    description=(
                        "Approval TTL is too short for current proposal cadence. Increase timeout "
                        "to reduce accidental expiry during normal operation."
                    ),
                    diff={"approval": {"timeout_hours": max(2, settings.approval.timeout_hours + 1)}},
                    expected_impact="Reduces emergency stops caused by proposal expiration.",
                    evidence={"current_timeout_hours": settings.approval.timeout_hours},
                    confidence=0.66,
                    ttl_hours=min(settings.approval.timeout_hours, 4),
                )
            )

        if "reconciliation" in reason_lower or source == "reconciliation_guard":
            warning = settings.trading.reconciliation_warning_tolerance
            critical = settings.trading.reconciliation_critical_tolerance
            proposals.append(
                AIProposal(
                    title="Tune reconciliation tolerances after critical mismatch",
                    proposal_type="emergency_remediation",
                    description=(
                        "Critical reconciliation mismatch triggered an emergency stop. Slightly increase "
                        "warning/critical tolerances while investigating source-of-truth drift."
                    ),
                    diff={
                        "trading": {
                            "reconciliation_warning_tolerance": round(warning * 1.25, 8),
                            "reconciliation_critical_tolerance": round(critical * 1.25, 8),
                        }
                    },
                    expected_impact=(
                        "Reduces false-positive emergency stops from transient reconciliation noise."
                    ),
                    evidence={
                        "current_warning_tolerance": warning,
                        "current_critical_tolerance": critical,
                        "metadata": metadata,
                    },
                    confidence=0.61,
                    ttl_hours=min(settings.approval.timeout_hours, 4),
                )
            )

        if "stop_loss" in reason_lower or source == "global_stop_loss":
            reduced_risk = max(settings.risk.min_per_trade, round(settings.risk.per_trade * 0.6, 6))
            reduced_exposure = max(0.05, round(settings.risk.max_exposure * 0.8, 4))
            wider_grid_min = min(500, settings.trading.grid_min_spacing_bps + 10)
            proposals.append(
                AIProposal(
                    title="Risk-off profile after stop-loss trigger",
                    proposal_type="risk_tuning",
                    description=(
                        "Global stop-loss was triggered. Move temporarily to a lower-risk profile by "
                        "reducing per-trade risk and exposure, and widening grid minimum spacing."
                    ),
                    diff={
                        "risk": {
                            "per_trade": reduced_risk,
                            "max_exposure": reduced_exposure,
                        },
                        "trading": {"grid_min_spacing_bps": wider_grid_min},
                    },
                    expected_impact="Lowers downside risk and slows churn during adverse conditions.",
                    evidence={
                        "reason": reason,
                        "current_per_trade": settings.risk.per_trade,
                        "current_max_exposure": settings.risk.max_exposure,
                        "current_grid_min_spacing_bps": settings.trading.grid_min_spacing_bps,
                    },
                    confidence=0.72,
                    ttl_hours=min(settings.approval.timeout_hours, 4),
                )
            )

        if not proposals:
            proposals.append(
                AIProposal(
                    title="Emergency cooldown and risk reduction",
                    proposal_type="emergency_remediation",
                    description=(
                        "No incident-specific remediation inferred. Apply a conservative temporary "
                        "risk reduction profile and review recent events."
                    ),
                    diff={
                        "risk": {
                            "per_trade": max(
                                settings.risk.min_per_trade,
                                round(settings.risk.per_trade * 0.75, 6),
                            ),
                            "max_exposure": max(0.05, round(settings.risk.max_exposure * 0.9, 4)),
                        }
                    },
                    expected_impact="Adds margin of safety while incident is investigated.",
                    evidence={"source": source, "reason": reason},
                    confidence=0.58,
                    ttl_hours=min(settings.approval.timeout_hours, 4),
                )
            )

        max_items = max(1, settings.approval.emergency_max_proposals)
        return proposals[:max_items]


def _safe_confidence(raw: Any) -> float:
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, int | float):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return 0.0
    else:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sanitize_error_text(value: str) -> str:
    """Redact API keys/tokens that might leak through provider error messages."""
    redacted = value
    redacted = re.sub(r"([?&]key=)[^&\s]+", r"\1REDACTED", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"(api[_-]?key\"?\s*[:=]\s*\"?)[A-Za-z0-9_\-]+", r"\1REDACTED", redacted, flags=re.IGNORECASE)
    redacted = re.sub(
        r"(api[_-]?secret\"?\s*[:=]\s*\"?)[A-Za-z0-9_\-]+",
        r"\1REDACTED",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(authorization\s*[:=]\s*bearer\s+)[^\s\"']+",
        r"\1REDACTED",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _emergency_system_prompt() -> str:
    return (
        "You are an emergency incident-response advisor for a crypto spot trading system. "
        "Return strict JSON only. Propose parameter diffs only. "
        "Never propose disabling core safety protections or bypassing approval workflow. "
        "Prefer small, reversible changes with strong evidence from context."
    )


def _emergency_response_schema() -> dict[str, Any]:
    return {
        "proposals": [
            {
                "title": "string",
                "proposal_type": "emergency_remediation|risk_tuning|execution_tuning|grid_tuning|anomaly_alert",
                "description": "string",
                "diff": {
                    "risk": {
                        "per_trade": "number",
                        "max_daily_loss": "number",
                        "max_exposure": "number",
                        "fee_bps": "integer",
                        "slippage_bps": "integer",
                    },
                    "trading": {
                        "advisor_interval_cycles": "integer",
                        "grid_min_spacing_bps": "integer",
                        "grid_max_spacing_bps": "integer",
                        "grid_recenter_mode": "aggressive|conservative",
                        "reconciliation_warning_tolerance": "number",
                        "reconciliation_critical_tolerance": "number",
                        "stop_loss_global_equity_pct": "number",
                        "stop_loss_max_drawdown_pct": "number",
                    },
                    "approval": {
                        "timeout_hours": "integer",
                        "auto_approve_enabled": "boolean",
                    },
                },
                "expected_impact": "string",
                "evidence": {"key": "value"},
                "confidence": "0.0-1.0",
                "ttl_hours": "integer",
            }
        ]
    }


_emergency_analyzer: EmergencyStopAnalyzer | None = None
_emergency_analyzer_lock = threading.Lock()


def get_emergency_stop_analyzer() -> EmergencyStopAnalyzer:
    """Get or create singleton emergency-stop analyzer."""
    global _emergency_analyzer
    if _emergency_analyzer is None:
        with _emergency_analyzer_lock:
            if _emergency_analyzer is None:
                _emergency_analyzer = EmergencyStopAnalyzer()
    return _emergency_analyzer
