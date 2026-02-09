"""Centralized audit logging helpers."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.database.models import EventLog


async def log_event(
    session: AsyncSession,
    *,
    event_type: str,
    event_category: str,
    summary: str,
    details: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    config_version: int = 1,
    actor: str = "system",
) -> EventLog:
    """Append an event to the audit log and flush."""
    event = EventLog(
        event_type=event_type,
        event_category=event_category,
        summary=summary,
        details=details or {},
        inputs=inputs or {},
        config_version=config_version,
        actor=actor,
    )
    session.add(event)
    await session.flush()
    return event
