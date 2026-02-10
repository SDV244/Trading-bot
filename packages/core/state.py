"""
System State Management

Handles the global system state (RUNNING, PAUSED, EMERGENCY_STOP)
with thread-safe operations and persistence.
"""

import threading
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SystemState(str, Enum):
    """System operating states."""

    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


class StateSnapshot(BaseModel):
    """Immutable snapshot of system state."""

    model_config = ConfigDict(frozen=True)

    state: SystemState
    reason: str = ""
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    changed_by: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateManager:
    """
    Manages system state transitions with validation and history.

    Thread-safe state management with audit trail.
    """

    # Valid state transitions
    VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
        SystemState.RUNNING: {SystemState.PAUSED, SystemState.EMERGENCY_STOP},
        SystemState.PAUSED: {SystemState.RUNNING, SystemState.EMERGENCY_STOP},
        # Emergency stop requires manual intervention - no auto transitions
        SystemState.EMERGENCY_STOP: set(),
    }

    def __init__(self) -> None:
        """Initialize state manager with PAUSED state."""
        self._current = StateSnapshot(
            state=SystemState.PAUSED,
            reason="System initialized",
            changed_by="system",
        )
        self._history: list[StateSnapshot] = [self._current]

    @property
    def current(self) -> StateSnapshot:
        """Get current state snapshot."""
        return self._current

    @property
    def state(self) -> SystemState:
        """Get current state value."""
        return self._current.state

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._current.state == SystemState.RUNNING

    @property
    def is_paused(self) -> bool:
        """Check if system is paused."""
        return self._current.state == SystemState.PAUSED

    @property
    def is_emergency_stopped(self) -> bool:
        """Check if system is in emergency stop."""
        return self._current.state == SystemState.EMERGENCY_STOP

    @property
    def can_trade(self) -> bool:
        """Check if system can execute trades."""
        return self._current.state == SystemState.RUNNING

    def transition_to(
        self,
        new_state: SystemState,
        reason: str,
        changed_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Transition to a new state if valid.

        Args:
            new_state: Target state
            reason: Reason for transition
            changed_by: Who initiated the change
            metadata: Additional context

        Returns:
            New state snapshot

        Raises:
            ValueError: If transition is not allowed
        """
        if new_state == self._current.state:
            return self._current

        allowed = self.VALID_TRANSITIONS.get(self._current.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {self._current.state} -> {new_state}. "
                f"Allowed: {allowed}"
            )

        self._current = StateSnapshot(
            state=new_state,
            reason=reason,
            changed_by=changed_by,
            metadata=metadata or {},
        )
        self._history.append(self._current)
        return self._current

    def force_emergency_stop(
        self,
        reason: str,
        changed_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Force emergency stop from any state.

        This bypasses normal transition rules for safety.
        """
        self._current = StateSnapshot(
            state=SystemState.EMERGENCY_STOP,
            reason=reason,
            changed_by=changed_by,
            metadata=metadata or {},
        )
        self._history.append(self._current)
        return self._current

    def manual_resume(
        self,
        reason: str,
        changed_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Manually resume from emergency stop.

        Requires explicit human intervention.
        """
        if self._current.state != SystemState.EMERGENCY_STOP:
            raise ValueError("Can only manually resume from EMERGENCY_STOP state")

        self._current = StateSnapshot(
            state=SystemState.PAUSED,
            reason=f"Manual resume: {reason}",
            changed_by=changed_by,
            metadata=metadata or {},
        )
        self._history.append(self._current)
        return self._current

    def pause(self, reason: str, changed_by: str = "system") -> StateSnapshot:
        """Pause the system."""
        return self.transition_to(SystemState.PAUSED, reason, changed_by)

    def resume(self, reason: str, changed_by: str = "system") -> StateSnapshot:
        """Resume from paused state."""
        return self.transition_to(SystemState.RUNNING, reason, changed_by)

    def get_history(self, limit: int = 100) -> list[StateSnapshot]:
        """Get state change history."""
        return list(reversed(self._history[-limit:]))


# Global state manager instance
_state_manager: StateManager | None = None
_state_manager_lock = threading.Lock()


def get_state_manager() -> StateManager:
    """Get or create global state manager."""
    global _state_manager
    if _state_manager is None:
        with _state_manager_lock:
            if _state_manager is None:
                _state_manager = StateManager()
    return _state_manager
