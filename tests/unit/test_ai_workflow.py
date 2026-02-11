"""Tests for AI advisor and approval gate workflow."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import packages.core.state as state_module
from packages.core.ai import AIAdvisor, ApprovalGate
from packages.core.ai.advisor import AIProposal
from packages.core.ai.emergency_analyzer import EmergencyStopAnalyzer
from packages.core.config import get_settings, reload_settings
from packages.core.database.models import Approval, Base, Config, EventLog, MetricsSnapshot


@pytest.fixture
async def db_session() -> AsyncSession:
    """Create in-memory database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        yield session


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset global state manager between tests."""
    state_module._state_manager = state_module.StateManager()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep approval auto-approve disabled by default in this module."""
    monkeypatch.setenv("APPROVAL_AUTO_APPROVE_ENABLED", "false")
    monkeypatch.setenv("LLM_ENABLED", "false")
    reload_settings()


@pytest.mark.asyncio
async def test_advisor_generates_drawdown_proposal(db_session: AsyncSession) -> None:
    db_session.add(
        MetricsSnapshot(
            strategy_name="trend_ema",
            total_trades=150,
            winning_trades=70,
            losing_trades=80,
            total_pnl=Decimal("-120"),
            total_fees=Decimal("40"),
            max_drawdown=0.11,
            sharpe_ratio=-0.2,
            sortino_ratio=-0.3,
            profit_factor=0.8,
            win_rate=0.46,
            avg_trade_pnl=Decimal("-0.8"),
            is_paper=True,
        )
    )
    await db_session.flush()

    advisor = AIAdvisor()
    proposals = await advisor.generate_proposals(db_session)

    assert proposals
    assert any(p.proposal_type == "risk_tuning" for p in proposals)


@pytest.mark.asyncio
async def test_advisor_generates_grid_tuning_for_smart_grid(db_session: AsyncSession) -> None:
    db_session.add(
        MetricsSnapshot(
            strategy_name="smart_grid_ai",
            total_trades=80,
            winning_trades=25,
            losing_trades=55,
            total_pnl=Decimal("-80"),
            total_fees=Decimal("35"),
            max_drawdown=0.05,
            sharpe_ratio=-0.1,
            sortino_ratio=-0.2,
            profit_factor=0.9,
            win_rate=0.31,
            avg_trade_pnl=Decimal("-1.1"),
            is_paper=True,
        )
    )
    await db_session.flush()

    advisor = AIAdvisor()
    proposals = await advisor.generate_proposals(db_session)

    assert any(p.proposal_type == "grid_tuning" for p in proposals)


@pytest.mark.asyncio
async def test_approval_gate_approve_applies_config_diff(db_session: AsyncSession) -> None:
    gate = ApprovalGate()
    db_session.add(Config(version=1, data={"risk": {"per_trade": 0.005}}, active=True))
    await db_session.flush()

    created = await gate.create_approval(
        db_session,
        AIProposal(
            title="Reduce per_trade",
            proposal_type="risk_tuning",
            description="Reduce risk",
            diff={"risk": {"per_trade": 0.0035}},
            expected_impact="Lower drawdown",
            evidence={"max_drawdown": 0.1},
            confidence=0.8,
        ),
    )
    approved = await gate.approve(db_session, created.id, decided_by="qa_user")

    assert approved.status == "APPROVED"
    config_result = await db_session.execute(select(Config).where(Config.active.is_(True)))
    active = config_result.scalar_one()
    assert active.version == 2
    assert active.data["risk"]["per_trade"] == 0.0035
    assert get_settings().risk.per_trade == 0.0035
    reload_settings()


@pytest.mark.asyncio
async def test_approval_expiry_triggers_emergency_stop(db_session: AsyncSession) -> None:
    gate = ApprovalGate()
    created = await gate.create_approval(
        db_session,
        AIProposal(
            title="Expired proposal",
            proposal_type="risk_tuning",
            description="Expire me",
            diff={"risk": {"per_trade": 0.0025}},
            expected_impact="n/a",
            evidence={},
            confidence=0.7,
            ttl_hours=1,
        ),
    )
    created.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await db_session.flush()

    expired = await gate.expire_pending(db_session)
    assert len(expired) == 1
    assert expired[0].status == "EXPIRED"
    assert state_module.get_state_manager().is_emergency_stopped

    approval_result = await db_session.execute(select(Approval).where(Approval.id == created.id))
    stored = approval_result.scalar_one()
    assert stored.status == "EXPIRED"


@pytest.mark.asyncio
async def test_approval_expiry_triggers_emergency_ai_analysis(db_session: AsyncSession) -> None:
    gate = ApprovalGate()
    created = await gate.create_approval(
        db_session,
        AIProposal(
            title="Expired proposal with AI follow-up",
            proposal_type="risk_tuning",
            description="Expire me",
            diff={"risk": {"per_trade": 0.0025}},
            expected_impact="n/a",
            evidence={},
            confidence=0.7,
            ttl_hours=1,
        ),
    )
    created.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await db_session.flush()

    expired = await gate.expire_pending(db_session)
    assert len(expired) == 1

    event_result = await db_session.execute(
        select(EventLog).where(EventLog.event_type == "emergency_ai_analysis_completed")
    )
    analysis_event = event_result.scalars().first()
    assert analysis_event is not None

    approvals = await gate.list_approvals(db_session, limit=50)
    assert any(
        approval.id != created.id and approval.proposal_type in {"emergency_remediation", "risk_tuning"}
        for approval in approvals
    )


@pytest.mark.asyncio
async def test_approval_gate_auto_approves_when_enabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APPROVAL_AUTO_APPROVE_ENABLED", "true")
    reload_settings()
    gate = ApprovalGate()
    db_session.add(Config(version=1, data={"risk": {"per_trade": 0.005}}, active=True))
    await db_session.flush()

    created = await gate.create_approval(
        db_session,
        AIProposal(
            title="Auto approve risk tuning",
            proposal_type="risk_tuning",
            description="Auto apply",
            diff={"risk": {"per_trade": 0.004}},
            expected_impact="Faster adaptation",
            evidence={"trigger": "test"},
            confidence=0.9,
        ),
    )

    assert created.status == "APPROVED"
    assert created.decided_by == "ai_auto_approver"
    config_result = await db_session.execute(select(Config).where(Config.active.is_(True)))
    active = config_result.scalar_one()
    assert active.version == 2
    assert active.data["risk"]["per_trade"] == 0.004
    reload_settings()


@pytest.mark.asyncio
async def test_expire_pending_auto_mode_sweeps_pending_without_emergency_stop(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ApprovalGate()
    db_session.add(Config(version=1, data={"risk": {"per_trade": 0.005}}, active=True))
    await db_session.flush()

    created = await gate.create_approval(
        db_session,
        AIProposal(
            title="Pending before auto mode",
            proposal_type="risk_tuning",
            description="Will be swept",
            diff={"risk": {"per_trade": 0.0038}},
            expected_impact="Lower risk",
            evidence={"source": "test"},
            confidence=0.76,
            ttl_hours=2,
        ),
    )
    assert created.status == "PENDING"

    monkeypatch.setenv("APPROVAL_AUTO_APPROVE_ENABLED", "true")
    reload_settings()

    expired = await gate.expire_pending(db_session)
    assert expired == []
    assert state_module.get_state_manager().is_emergency_stopped is False

    approval_result = await db_session.execute(select(Approval).where(Approval.id == created.id))
    stored = approval_result.scalar_one()
    assert stored.status == "APPROVED"
    assert stored.decided_by == "ai_auto_approver"
    reload_settings()


@pytest.mark.asyncio
async def test_advisor_prefers_llm_proposals_when_enabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_PREFER_LLM", "true")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULES", "true")
    reload_settings()

    db_session.add(
        MetricsSnapshot(
            strategy_name="smart_grid_ai",
            total_trades=140,
            winning_trades=60,
            losing_trades=80,
            total_pnl=Decimal("-20"),
            total_fees=Decimal("15"),
            max_drawdown=0.03,
            sharpe_ratio=0.4,
            sortino_ratio=0.7,
            profit_factor=1.1,
            win_rate=0.43,
            avg_trade_pnl=Decimal("-0.1"),
            is_paper=True,
        )
    )
    await db_session.flush()

    advisor = AIAdvisor()

    async def fake_llm(*, context: dict[str, object]) -> list[dict[str, object]]:
        assert context["active_strategy"] == "smart_grid_ai"
        return [
            {
                "title": "LLM grid tune",
                "proposal_type": "grid_tuning",
                "description": "Widen bands slightly.",
                "diff": {"trading": {"grid_min_spacing_bps": 45, "grid_max_spacing_bps": 280}},
                "expected_impact": "Less churn.",
                "evidence": {"reason": "llm_test"},
                "confidence": 0.89,
                "ttl_hours": 2,
            }
        ]

    monkeypatch.setattr(advisor, "_generate_llm_raw_proposals", fake_llm)
    proposals = await advisor.generate_proposals(db_session)

    assert len(proposals) == 1
    assert proposals[0].title == "LLM grid tune"
    assert proposals[0].proposal_type == "grid_tuning"
    assert proposals[0].diff["trading"]["grid_min_spacing_bps"] == 45
    reload_settings()


@pytest.mark.asyncio
async def test_advisor_llm_fallback_to_rules_when_llm_empty(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_PREFER_LLM", "true")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULES", "true")
    reload_settings()

    db_session.add(
        MetricsSnapshot(
            strategy_name="trend_ema",
            total_trades=200,
            winning_trades=90,
            losing_trades=110,
            total_pnl=Decimal("-55"),
            total_fees=Decimal("25"),
            max_drawdown=0.10,
            sharpe_ratio=-0.2,
            sortino_ratio=-0.1,
            profit_factor=0.9,
            win_rate=0.45,
            avg_trade_pnl=Decimal("-0.3"),
            is_paper=True,
        )
    )
    await db_session.flush()

    advisor = AIAdvisor()

    async def fake_empty(*, context: dict[str, object]) -> list[dict[str, object]]:
        _ = context
        return []

    monkeypatch.setattr(advisor, "_generate_llm_raw_proposals", fake_empty)
    proposals = await advisor.generate_proposals(db_session)

    assert any(p.proposal_type == "risk_tuning" for p in proposals)
    reload_settings()


@pytest.mark.asyncio
async def test_advisor_prefers_multiagent_when_enabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_PREFER_LLM", "true")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULES", "true")
    monkeypatch.setenv("MULTIAGENT_ENABLED", "true")
    reload_settings()

    db_session.add(
        MetricsSnapshot(
            strategy_name="smart_grid_ai",
            total_trades=180,
            winning_trades=100,
            losing_trades=80,
            total_pnl=Decimal("44"),
            total_fees=Decimal("22"),
            max_drawdown=0.05,
            sharpe_ratio=0.9,
            sortino_ratio=1.2,
            profit_factor=1.25,
            win_rate=0.55,
            avg_trade_pnl=Decimal("0.2"),
            is_paper=True,
        )
    )
    await db_session.flush()

    advisor = AIAdvisor()

    async def fake_multi(*_args, **_kwargs):
        return [
            AIProposal(
                title="[strategy_agent] Multi-agent tune",
                proposal_type="grid_tuning",
                description="Agent consensus recommends spacing update.",
                diff={"trading": {"grid_min_spacing_bps": 42, "grid_max_spacing_bps": 260}},
                expected_impact="Reduce churn.",
                evidence={"source": "multiagent_test"},
                confidence=0.81,
            )
        ]

    monkeypatch.setattr(advisor, "_multi_agent_proposals", fake_multi)
    proposals = await advisor.generate_proposals(db_session)
    assert len(proposals) == 1
    assert "Multi-agent tune" in proposals[0].title
    assert proposals[0].diff["trading"]["grid_min_spacing_bps"] == 42
    reload_settings()


@pytest.mark.asyncio
async def test_strategy_analysis_returns_grid_improvement_recommendation(db_session: AsyncSession) -> None:
    db_session.add(
        MetricsSnapshot(
            strategy_name="smart_grid_ai",
            total_trades=200,
            winning_trades=98,
            losing_trades=102,
            total_pnl=Decimal("24"),
            total_fees=Decimal("18"),
            max_drawdown=0.06,
            sharpe_ratio=0.9,
            sortino_ratio=1.1,
            profit_factor=1.05,
            win_rate=0.49,
            avg_trade_pnl=Decimal("0.12"),
            is_paper=True,
        )
    )
    for idx in range(80):
        db_session.add(
            EventLog(
                event_type="risk_hold",
                event_category="risk",
                summary="Signal blocked by risk engine: signal_hold",
                details={"signal_reason": "grid_wait_inside_band", "risk_reason": "signal_hold"},
                config_version=1,
                actor="system",
                created_at=datetime.now(UTC) - timedelta(minutes=idx),
            )
        )
    await db_session.flush()

    advisor = AIAdvisor()
    analysis = await advisor.analyze_strategy(db_session)

    assert analysis["active_strategy"] == "smart_grid_ai"
    assert analysis["hold_diagnostics"]["hold_rate_24h"] > 0.8
    assert any(
        rec["proposal_type"] == "grid_tuning"
        and rec["title"] == "Increase smart-grid participation in low-activity regime"
        for rec in analysis["recommendations"]
    )


@pytest.mark.asyncio
async def test_emergency_analyzer_redacts_llm_keys_in_errors(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    reload_settings()

    analyzer = EmergencyStopAnalyzer()

    class _FakeClient:
        async def generate_structured(self, **_kwargs):
            raise RuntimeError(
                "Client error '429 Too Many Requests' for url "
                "'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                "?key=THIS_SHOULD_NOT_LEAK'"
            )

    with patch("packages.core.ai.emergency_analyzer.get_llm_advisor_client", return_value=_FakeClient()):
        outcome = await analyzer.analyze_and_enqueue(
            db_session,
            reason="manual_verification",
            source="unit_test",
            metadata={},
            actor="tester",
            force=True,
        )

    assert outcome.triggered is True
    assert outcome.llm_error is not None
    assert "THIS_SHOULD_NOT_LEAK" not in outcome.llm_error
    assert "key=REDACTED" in outcome.llm_error
    reload_settings()
