"""
Tests for Candle Repository.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from packages.adapters.binance_spot import CandleData
from packages.core.database.models import Base
from packages.core.database.repositories import CandleRepository


@pytest.fixture
async def db_session():
    """Create in-memory database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession)
    async with session_factory() as session:
        yield session


def create_candle_data(symbol="BTCUSDT", open_time=None, timeframe="1h"):
    """Helper to create candle data."""
    if open_time is None:
        open_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    return CandleData(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        close_time=open_time + timedelta(hours=1) - timedelta(milliseconds=1),
        open=Decimal("50000"),
        high=Decimal("51000"),
        low=Decimal("49000"),
        close=Decimal("50500"),
        volume=Decimal("10.5"),
        quote_volume=Decimal("525000"),
        trades_count=100
    )


@pytest.mark.asyncio
async def test_upsert_candles(db_session):
    """Can upsert candles."""
    repo = CandleRepository(db_session)
    candle = create_candle_data()

    # Insert
    count = await repo.upsert_candles([candle])
    assert count == 1

    stored = await repo.get_latest_candles(candle.symbol, candle.timeframe)
    assert len(stored) == 1
    assert stored[0].close == Decimal("50500")

    # Update same candle with new close price
    candle.close = Decimal("50600")
    count = await repo.upsert_candles([candle])
    assert count == 1

    stored = await repo.get_latest_candles(candle.symbol, candle.timeframe)
    assert len(stored) == 1
    assert stored[0].close == Decimal("50600")


@pytest.mark.asyncio
async def test_get_candles_range(db_session):
    """Can fetch candles by range."""
    repo = CandleRepository(db_session)
    base_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    candles = [
        create_candle_data(open_time=base_time - timedelta(hours=i))
        for i in range(5)
    ]

    await repo.upsert_candles(candles)

    # Fetch subset
    range_candles = await repo.get_candles_range(
        "BTCUSDT", "1h",
        start_time=base_time - timedelta(hours=3),
        end_time=base_time - timedelta(hours=1)
    )

    assert len(range_candles) == 3
    # Check ordering (oldest first)
    assert range_candles[0].open_time < range_candles[1].open_time


@pytest.mark.asyncio
async def test_delete_old_candles(db_session):
    """Can delete old candles."""
    repo = CandleRepository(db_session)
    candles = [
        create_candle_data(open_time=datetime.now(UTC) - timedelta(hours=i))
        for i in range(10)
    ]
    await repo.upsert_candles(candles)

    # Keep only 3 most recent
    deleted = await repo.delete_old_candles("BTCUSDT", "1h", keep_count=3)
    assert deleted == 7

    count = await repo.count_candles("BTCUSDT", "1h")
    assert count == 3
