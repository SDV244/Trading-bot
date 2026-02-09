"""
Candle Data Repository

Repository pattern for storing and retrieving OHLCV candle data.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.binance_spot import CandleData
from packages.core.database.models import Candle


class CandleRepository:
    """
    Repository for candle data operations.

    Provides methods for storing and retrieving OHLCV candles
    with upsert support for efficient caching.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self.session = session

    async def upsert_candles(self, candles: list[CandleData]) -> int:
        """
        Insert or update candles (upsert).

        Uses SQLite's ON CONFLICT DO UPDATE for efficiency.

        Args:
            candles: List of CandleData objects

        Returns:
            Number of candles upserted
        """
        if not candles:
            return 0

        values = [
            {
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "open_time": c.open_time,
                "close_time": c.close_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trades_count": c.trades_count,
            }
            for c in candles
        ]

        # SQLite upsert
        stmt = sqlite_insert(Candle).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timeframe", "open_time"],
            set_={
                "close_time": stmt.excluded.close_time,
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "quote_volume": stmt.excluded.quote_volume,
                "trades_count": stmt.excluded.trades_count,
            },
        )

        await self.session.execute(stmt)
        await self.session.commit()

        logger.debug(f"Upserted {len(candles)} candles for {candles[0].symbol}/{candles[0].timeframe}")
        return len(candles)

    async def get_latest_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[Candle]:
        """
        Get the most recent candles.

        Args:
            symbol: Trading pair
            timeframe: Candle interval
            limit: Maximum number of candles

        Returns:
            List of Candle models, newest first
        """
        stmt = (
            select(Candle)
            .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.open_time.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_candles_range(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """
        Get candles within a time range.

        Args:
            symbol: Trading pair
            timeframe: Candle interval
            start_time: Range start (inclusive)
            end_time: Range end (inclusive)

        Returns:
            List of Candle models, oldest first
        """
        stmt = select(Candle).where(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        )

        if start_time:
            stmt = stmt.where(Candle.open_time >= start_time)
        if end_time:
            stmt = stmt.where(Candle.open_time <= end_time)

        stmt = stmt.order_by(Candle.open_time.asc())

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_close_price(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
    ) -> Decimal | None:
        """
        Get the most recent close price.

        Args:
            symbol: Trading pair
            timeframe: Candle interval

        Returns:
            Latest close price or None if no data
        """
        stmt = (
            select(Candle.close)
            .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.open_time.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        return Decimal(str(row)) if row else None

    async def count_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
    ) -> int:
        """
        Count total candles stored.

        Args:
            symbol: Trading pair
            timeframe: Candle interval

        Returns:
            Total count
        """
        from sqlalchemy import func

        stmt = select(func.count(Candle.id)).where(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def delete_old_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        keep_count: int = 1000,
    ) -> int:
        """
        Delete old candles keeping only the most recent.

        Args:
            symbol: Trading pair
            timeframe: Candle interval
            keep_count: Number of recent candles to keep

        Returns:
            Number of deleted candles
        """
        # Get the cutoff time
        subq = (
            select(Candle.open_time)
            .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.open_time.desc())
            .offset(keep_count - 1)
            .limit(1)
            .scalar_subquery()
        )

        stmt = delete(Candle).where(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
            Candle.open_time < subq,
        )

        result = cast(CursorResult[Any], await self.session.execute(stmt))
        await self.session.commit()

        deleted = int(result.rowcount or 0)
        if deleted > 0:
            logger.info(f"Deleted {deleted} old candles for {symbol}/{timeframe}")
        return deleted
