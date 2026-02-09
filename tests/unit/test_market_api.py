"""
Tests for Market API Endpoints.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.trading_cycle import DataReadiness, TimeframeReadiness


class MockCandle:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def client_with_mocks():
    with patch("packages.adapters.binance_spot.get_binance_adapter") as mock_adapter_getter, \
         patch("packages.core.database.repositories.CandleRepository") as mock_repo_cls, \
         patch("packages.core.database.session.get_session") as mock_session_getter, \
         patch("packages.core.data_fetcher.get_data_fetcher") as mock_fetcher_getter, \
         patch("packages.core.trading_cycle.get_trading_cycle_service") as mock_cycle_service_getter:

        # Setup mocks
        mock_adapter = AsyncMock()
        mock_adapter_getter.return_value = mock_adapter

        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_fetcher = AsyncMock()
        mock_fetcher_getter.return_value = mock_fetcher

        mock_cycle_service = AsyncMock()
        mock_cycle_service.get_data_readiness.return_value = DataReadiness(
            symbol="BTCUSDT",
            active_strategy="trend_ema",
            require_data_ready=True,
            data_ready=True,
            reasons=[],
            timeframes={
                "1h": TimeframeReadiness(required=20, available=120, ready=True),
                "4h": TimeframeReadiness(required=50, available=80, ready=True),
            },
        )
        mock_cycle_service_getter.return_value = mock_cycle_service

        mock_session = AsyncMock()
        mock_session_getter.return_value.__aenter__.return_value = mock_session

        yield TestClient(app), mock_adapter, mock_repo, mock_fetcher, mock_cycle_service


def test_get_current_price(client_with_mocks):
    """Can hit price endpoint."""
    client, mock_adapter, _, _, _ = client_with_mocks

    mock_adapter.get_ticker_price.return_value = Decimal("50000.50")

    response = client.get("/api/market/price?symbol=BTCUSDT")

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["price"] == "50000.50"
    assert "timestamp" in data


def test_get_candles(client_with_mocks):
    """Can fetch candles from DB endpoint."""
    client, _, mock_repo, _, _ = client_with_mocks

    # Mock repository return using object with attributes
    mock_repo.get_latest_candles.return_value = [
        MockCandle(
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

    response = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1h")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTCUSDT"
    assert data[0]["close"] == "50500"


def test_trigger_data_fetch(client_with_mocks):
    """Can trigger data fetch."""
    client, _, _, mock_fetcher, _ = client_with_mocks

    # Mock return value needs to be awaited since it's an async method
    mock_fetcher.fetch_all_timeframes.return_value = {"1h": 100, "4h": 25}

    response = client.post("/api/market/data/fetch?days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # The endpoint returns "fetched" key with the dict
    assert data["fetched"]["1h"] == 100


def test_get_data_requirements(client_with_mocks):
    """Can fetch active strategy candle requirements."""
    client, _, _, _, mock_cycle_service = client_with_mocks

    response = client.get("/api/market/data/requirements")

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["active_strategy"] == "trend_ema"
    assert data["data_ready"] is True
    assert data["timeframes"]["4h"]["required"] == 50
    mock_cycle_service.get_data_readiness.assert_awaited_once()
