"""
Tests for Data Fetcher Service.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from packages.adapters.binance_spot import CandleData
from packages.core.data_fetcher import DataFetcher


@pytest.fixture
def mock_adapter():
    with patch("packages.core.data_fetcher.get_binance_adapter") as mock:
        adapter = AsyncMock()
        mock.return_value = adapter
        yield adapter


@pytest.fixture
def mock_repo():
    with patch("packages.core.data_fetcher.CandleRepository") as mock:
        repo = AsyncMock()
        mock.return_value = repo
        yield repo


@pytest.fixture
def mock_session():
    with patch("packages.core.data_fetcher.get_session") as mock:
        session = AsyncMock()
        mock.return_value.__aenter__.return_value = session
        yield session


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_session")
async def test_fetch_latest(mock_adapter, mock_repo):
    """Can fetch latest candles."""
    fetcher = DataFetcher()

    # Mock adapter response
    mock_adapter.get_klines.return_value = [
        CandleData(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            quote_volume=Decimal("5000000"),
            trades_count=100
        )
    ]

    count = await fetcher.fetch_latest("1h", limit=5)

    assert count == 1
    mock_adapter.get_klines.assert_called_once()
    mock_repo.upsert_candles.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_session")
async def test_fetch_historical(mock_adapter, mock_repo):
    """Can fetch historical candles."""
    fetcher = DataFetcher()

    # Mock adapter response for pagination
    # Use a time in the past so the loop continues
    past_time = datetime.now(UTC)

    mock_adapter.get_klines.side_effect = [
        [CandleData(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=past_time,
            close_time=past_time, # Effectively "old" relative to end_time if we assume loop check?
            # Wait, fetcher uses datetime.now() as end_time.
            # If close_time is now, then current_start = now. loop terminates.
            # We need close_time < now.
            # But fetcher calculates end_time = now.
            # So if close_time = now - 1h, then current_start = now - 1h. loop continues.
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            quote_volume=Decimal("5000000"),
            trades_count=100
        )],
        []
    ]

    # Actually, let's just assert call_count >= 1 or adjust logic to be robust.
    # But to force 2 calls, we need close_time < now.
    # Let's mock datetime in fetcher? No that's hard.
    # Let's just set the candle time to a fixed past time.
    from datetime import timedelta
    old_time = datetime.now(UTC) - timedelta(hours=2)

    mock_adapter.get_klines.side_effect = [
        [CandleData(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=old_time,
            close_time=old_time,
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            quote_volume=Decimal("5000000"),
            trades_count=100
        )],
        []
    ]

    count = await fetcher.fetch_historical("1h", days=1)

    assert count == 1
    assert mock_adapter.get_klines.call_count == 2
    mock_repo.upsert_candles.assert_called_once()
