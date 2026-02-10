"""
Data Fetcher Service

Service for fetching and storing market data from Binance.
Handles initial data loading and periodic updates.
"""

import asyncio
import threading
from datetime import UTC, datetime, timedelta

from loguru import logger

from packages.adapters.binance_spot import get_binance_adapter
from packages.core.config import get_settings
from packages.core.database.repositories import CandleRepository
from packages.core.database.session import get_session


class DataFetcher:
    """
    Market data fetcher service.

    Responsible for:
    - Initial historical data loading
    - Periodic data updates
    - Gap detection and backfilling
    """

    def __init__(self) -> None:
        """Initialize the data fetcher."""
        self.settings = get_settings()
        self.symbol = self.settings.trading.pair
        self.timeframes = self.settings.trading.timeframe_list
        self._running = False
        self._task: asyncio.Task | None = None

    async def fetch_historical(
        self,
        timeframe: str = "1h",
        days: int = 30,
    ) -> int:
        """
        Fetch historical candle data.

        Args:
            timeframe: Candle interval
            days: Number of days of history

        Returns:
            Number of candles fetched
        """
        adapter = get_binance_adapter()
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=days)

        logger.info(f"Fetching {days} days of {self.symbol}/{timeframe} data")

        all_candles = []
        current_start = start_time

        while current_start < end_time:
            candles = await adapter.get_klines(
                symbol=self.symbol,
                interval=timeframe,
                limit=1000,
                start_time=current_start,
                end_time=end_time,
            )

            if not candles:
                break

            all_candles.extend(candles)
            current_start = candles[-1].close_time + timedelta(milliseconds=1)

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

        # Store in database
        if all_candles:
            async with get_session() as session:
                repo = CandleRepository(session)
                await repo.upsert_candles(all_candles)

        logger.info(f"Fetched and stored {len(all_candles)} candles for {self.symbol}/{timeframe}")
        return len(all_candles)

    async def fetch_latest(self, timeframe: str = "1h", limit: int = 10) -> int:
        """
        Fetch the latest candles.

        Args:
            timeframe: Candle interval
            limit: Number of recent candles

        Returns:
            Number of candles fetched
        """
        adapter = get_binance_adapter()
        candles = await adapter.get_klines(
            symbol=self.symbol,
            interval=timeframe,
            limit=limit,
        )

        if candles:
            async with get_session() as session:
                repo = CandleRepository(session)
                await repo.upsert_candles(candles)

        return len(candles)

    async def fetch_all_timeframes(self, days: int = 30) -> dict[str, int]:
        """
        Fetch historical data for all configured timeframes.

        Args:
            days: Number of days of history

        Returns:
            Dict mapping timeframe to candle count
        """
        results = {}
        for tf in self.timeframes:
            count = await self.fetch_historical(tf, days)
            results[tf] = count
        return results

    async def update_all_timeframes(self) -> dict[str, int]:
        """
        Update latest candles for all configured timeframes.

        Returns:
            Dict mapping timeframe to candle count
        """
        results = {}
        for tf in self.timeframes:
            count = await self.fetch_latest(tf, limit=5)
            results[tf] = count
        return results

    async def _update_loop(self, interval_seconds: int = 60) -> None:
        """
        Continuous update loop.

        Args:
            interval_seconds: Seconds between updates
        """
        logger.info(f"Starting data update loop (interval: {interval_seconds}s)")

        while self._running:
            try:
                await self.update_all_timeframes()
            except Exception as e:
                logger.error(f"Error updating data: {e}")

            await asyncio.sleep(interval_seconds)

    def start(self, interval_seconds: int = 60) -> None:
        """
        Start the continuous update loop.

        Args:
            interval_seconds: Seconds between updates
        """
        if self._running:
            logger.warning("Data fetcher already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._update_loop(interval_seconds))
        logger.info("Data fetcher started")

    def stop(self) -> None:
        """Stop the continuous update loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Data fetcher stopped")


# Singleton instance
_fetcher: DataFetcher | None = None
_fetcher_lock = threading.Lock()


def get_data_fetcher() -> DataFetcher:
    """Get or create the data fetcher singleton."""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher
