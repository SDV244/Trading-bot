"""Centralized audit logging helpers."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.database.models import Config, EventLog

_SESSION_CONFIG_VERSION_CACHE_KEY = "active_config_version"


async def resolve_active_config_version(session: AsyncSession, default: int = 1) -> int:
    """Resolve active configuration version and memoize for this DB session."""
    cached = session.info.get(_SESSION_CONFIG_VERSION_CACHE_KEY)
    if isinstance(cached, int) and cached > 0:
        return cached

    result = await session.execute(
        select(Config.version).where(Config.active.is_(True)).order_by(Config.version.desc()).limit(1)
    )
    version_raw = result.scalar_one_or_none()
    version = int(version_raw) if version_raw is not None else default
    session.info[_SESSION_CONFIG_VERSION_CACHE_KEY] = version
    return version


async def log_event(
    session: AsyncSession,
    *,
    event_type: str,
    event_category: str,
    summary: str,
    details: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    config_version: int | None = None,
    actor: str = "system",
) -> EventLog:
    """Append an event to the audit log and flush."""
    resolved_config_version = (
        config_version if config_version is not None else await resolve_active_config_version(session)
    )
    event = EventLog(
        event_type=event_type,
        event_category=event_category,
        summary=summary,
        details=details or {},
        inputs=inputs or {},
        config_version=resolved_config_version,
        actor=actor,
    )
    session.add(event)
    await session.flush()
    return event
