"""
Tests for state management.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import packages.core.state as state_module
from packages.core.database.models import Base, EventLog
from packages.core.state import StateManager, SystemState, restore_state_from_audit


class TestStateManager:
    """Test cases for StateManager."""

    def test_initial_state_is_paused(self):
        """State manager starts in PAUSED state."""
        manager = StateManager()
        assert manager.state == SystemState.PAUSED
        assert manager.is_paused
        assert not manager.is_running
        assert not manager.can_trade

    def test_valid_transition_pause_to_running(self):
        """Can transition from PAUSED to RUNNING."""
        manager = StateManager()
        manager.resume("Testing resume", "test")
        assert manager.state == SystemState.RUNNING
        assert manager.is_running
        assert manager.can_trade

    def test_valid_transition_running_to_paused(self):
        """Can transition from RUNNING to PAUSED."""
        manager = StateManager()
        manager.resume("Start", "test")
        manager.pause("Testing pause", "test")
        assert manager.state == SystemState.PAUSED
        assert not manager.can_trade

    def test_valid_transition_to_emergency_stop(self):
        """Can transition to EMERGENCY_STOP from any state."""
        manager = StateManager()
        manager.resume("Start", "test")
        manager.force_emergency_stop("Test emergency", "test")
        assert manager.state == SystemState.EMERGENCY_STOP
        assert manager.is_emergency_stopped
        assert not manager.can_trade

    def test_emergency_stop_blocks_normal_transitions(self):
        """Cannot use normal transitions from EMERGENCY_STOP."""
        manager = StateManager()
        manager.force_emergency_stop("Test", "test")

        with pytest.raises(ValueError, match="Invalid state transition"):
            manager.resume("Try resume", "test")

        with pytest.raises(ValueError, match="Invalid state transition"):
            manager.pause("Try pause", "test")

    def test_manual_resume_from_emergency_stop(self):
        """Can manually resume from EMERGENCY_STOP."""
        manager = StateManager()
        manager.force_emergency_stop("Test", "test")
        manager.manual_resume("Manual intervention", "admin")
        assert manager.state == SystemState.PAUSED
        assert not manager.is_emergency_stopped

    def test_cannot_manual_resume_if_not_stopped(self):
        """Cannot use manual_resume if not in EMERGENCY_STOP."""
        manager = StateManager()
        with pytest.raises(ValueError, match="only manually resume"):
            manager.manual_resume("Invalid", "test")

    def test_state_history_tracking(self):
        """State changes are tracked in history."""
        manager = StateManager()
        manager.resume("Start", "user1")
        manager.pause("Stop", "user2")
        manager.force_emergency_stop("Emergency", "system")

        history = manager.get_history()
        assert len(history) >= 4  # Initial + 3 transitions
        assert history[0].state == SystemState.EMERGENCY_STOP
        assert history[0].changed_by == "system"

    def test_same_state_transition_noop(self):
        """Transitioning to same state is a no-op."""
        manager = StateManager()
        snapshot1 = manager.current
        snapshot2 = manager.transition_to(SystemState.PAUSED, "Same state", "test")
        assert snapshot1 == snapshot2


@pytest.fixture
async def db_session() -> AsyncSession:
    """Create in-memory async database session for restore tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        yield session


@pytest.fixture(autouse=True)
def reset_global_state_manager() -> None:
    """Reset global state manager singleton between tests."""
    state_module._state_manager = state_module.StateManager()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_restore_state_from_audit_restores_emergency_stop(db_session: AsyncSession) -> None:
    db_session.add(
        EventLog(
            event_type="emergency_stop_triggered",
            event_category="system",
            summary="Emergency stop due to expired approvals",
            details={"reason": "Approval timeout (1 expired)"},
            inputs={},
            config_version=1,
            actor="approval_gate",
            created_at=datetime.now(UTC),
        )
    )
    await db_session.flush()

    snapshot = await restore_state_from_audit(db_session)
    assert snapshot.state == SystemState.EMERGENCY_STOP
    assert snapshot.reason == "Approval timeout (1 expired)"
    assert snapshot.changed_by == "approval_gate"
    assert state_module.get_state_manager().is_emergency_stopped


@pytest.mark.asyncio
async def test_restore_state_from_audit_restores_running_state(db_session: AsyncSession) -> None:
    db_session.add(
        EventLog(
            event_type="system_state_changed",
            event_category="system",
            summary="State changed to running: resume",
            details={"state": "running", "reason": "resume"},
            inputs={},
            config_version=1,
            actor="operator",
            created_at=datetime.now(UTC),
        )
    )
    await db_session.flush()

    snapshot = await restore_state_from_audit(db_session)
    assert snapshot.state == SystemState.RUNNING
    assert snapshot.reason == "resume"
    assert snapshot.changed_by == "operator"
    assert state_module.get_state_manager().is_running
