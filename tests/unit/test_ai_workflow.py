"""Tests for AI advisor and approval gate workflow."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import packages.core.state as state_module
from packages.core.ai import AIAdvisor, ApprovalGate
from packages.core.ai.advisor import AIProposal
from packages.core.database.models import Approval, Base, Config, MetricsSnapshot


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
