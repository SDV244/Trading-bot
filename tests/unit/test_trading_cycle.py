"""Tests for trading cycle orchestration."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import packages.core.state as state_module
import packages.core.trading_cycle as cycle_module
from packages.core.database.models import (
    Base,
    Candle,
    EquitySnapshot,
    Fill,
    MetricsSnapshot,
    Order,
    Position,
)
from packages.core.trading_cycle import TradingCycleService


@pytest.fixture
async def db_session() -> AsyncSession:
    """Create in-memory database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset mutable singletons used by the trading loop."""
    state_module._state_manager = state_module.StateManager()  # type: ignore[attr-defined]
    cycle_module._trading_cycle_service = None  # type: ignore[attr-defined]


async def _seed_candles(
    session: AsyncSession,
    timeframe: str,
    count: int,
    start_price: Decimal,
    step: Decimal,
) -> None:
    interval = timedelta(hours=1 if timeframe == "1h" else 4)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - (interval * count)
    rows = []
    for i in range(count):
        open_time = start + (interval * i)
        close = start_price + (step * Decimal(i))
        rows.append(
            Candle(
                symbol="BTCUSDT",
                timeframe=timeframe,
                open_time=open_time,
                close_time=open_time + interval - timedelta(milliseconds=1),
                open=close - Decimal("5"),
                high=close + Decimal("10"),
                low=close - Decimal("10"),
                close=close,
                volume=Decimal("100"),
                quote_volume=Decimal("100000"),
                trades_count=100,
            )
        )
    session.add_all(rows)
    await session.flush()


@pytest.mark.asyncio
async def test_cycle_skips_when_system_not_running(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 80, Decimal("50000"), Decimal("8"))
    await _seed_candles(db_session, "4h", 80, Decimal("48000"), Decimal("40"))

    service = TradingCycleService()
    result = await service.run_once(db_session)

    assert result.executed is False
    assert result.risk_reason == "system_not_running"
    orders = (await db_session.execute(select(func.count(Order.id)))).scalar_one()
    assert orders == 0


@pytest.mark.asyncio
async def test_cycle_executes_and_persists_trade(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    result = await service.run_once(db_session)

    assert result.executed is True
    assert result.order_id is not None
    assert result.fill_id is not None

    order_count = (await db_session.execute(select(func.count(Order.id)))).scalar_one()
    fill_count = (await db_session.execute(select(func.count(Fill.id)))).scalar_one()
    position_count = (await db_session.execute(select(func.count(Position.id)))).scalar_one()
    equity_count = (await db_session.execute(select(func.count(EquitySnapshot.id)))).scalar_one()
    metrics_count = (await db_session.execute(select(func.count(MetricsSnapshot.id)))).scalar_one()

    assert order_count == 1
    assert fill_count == 1
    assert position_count == 1
    assert equity_count == 1
    assert metrics_count == 1
