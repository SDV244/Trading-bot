"""
Database Connection and Session Management

Async SQLAlchemy engine and session factory for SQLite.
"""

import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from packages.core.config import get_settings
from packages.core.database.models import Base

# Engine instance (lazy initialization)
_engine = None
_session_factory = None
_session_init_lock = threading.RLock()


def get_engine() -> AsyncEngine:
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        with _session_init_lock:
            if _engine is None:
                settings = get_settings()
                url = make_url(settings.database.url)
                if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
                    Path(url.database).parent.mkdir(parents=True, exist_ok=True)
                engine_kwargs: dict[str, object] = {
                    "echo": settings.log.level == "DEBUG",
                    "future": True,
                }
                if url.get_backend_name() != "sqlite":
                    engine_kwargs.update(
                        {
                            "pool_size": settings.database.pool_size,
                            "max_overflow": settings.database.max_overflow,
                            "pool_pre_ping": settings.database.pool_pre_ping,
                            "pool_recycle": settings.database.pool_recycle_seconds,
                        }
                    )
                _engine = create_async_engine(settings.database.url, **engine_kwargs)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        with _session_init_lock:
            if _session_factory is None:
                _session_factory = async_sessionmaker(
                    bind=get_engine(),
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database() -> None:
    """Initialize database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Close database connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
