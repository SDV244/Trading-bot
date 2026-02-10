"""
Test fixtures and configuration.
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.config import reload_settings
from packages.core.database.session import close_database


@pytest.fixture(scope="session", autouse=True)
def isolate_test_database(tmp_path_factory: pytest.TempPathFactory) -> None:
    """
    Force tests to use an isolated sqlite database file.

    Prevents unit/integration tests from polluting local runtime DB data.
    """
    db_dir = tmp_path_factory.mktemp("trading_bot_tests")
    db_path = db_dir / "trading_test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    reload_settings()
    asyncio.run(close_database())
    yield
    asyncio.run(close_database())


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("TRADING_PAIR", "BTCUSDT")
    monkeypatch.setenv("TRADING_LIVE_MODE", "false")
