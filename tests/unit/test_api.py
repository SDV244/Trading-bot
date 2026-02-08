"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self):
        """Health endpoint returns healthy status."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    def test_readiness_check(self):
        """Readiness endpoint returns ready status."""
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True


class TestSystemEndpoints:
    """Test system state endpoints."""

    def test_get_system_state(self):
        """Can get current system state."""
        client = TestClient(app)
        response = client.get("/api/system/state")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "reason" in data
        assert "can_trade" in data

    def test_pause_system(self):
        """Can pause the system."""
        client = TestClient(app)

        # First resume to ensure we're not paused
        client.post(
            "/api/system/state",
            json={"action": "resume", "reason": "Test setup"},
        )

        response = client.post(
            "/api/system/state",
            json={"action": "pause", "reason": "Test pause", "changed_by": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "paused"
        assert data["can_trade"] is False

    def test_emergency_stop(self):
        """Can trigger emergency stop."""
        client = TestClient(app)
        response = client.post(
            "/api/system/emergency-stop",
            json={"reason": "Test emergency", "changed_by": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "emergency_stop"
        assert data["can_trade"] is False

        # Manual resume to reset state
        client.post(
            "/api/system/state",
            json={"action": "manual_resume", "reason": "Reset", "changed_by": "test"},
        )

    def test_invalid_action(self):
        """Invalid action returns error."""
        client = TestClient(app)
        response = client.post(
            "/api/system/state",
            json={"action": "invalid_action", "reason": "Test"},
        )
        assert response.status_code == 400


class TestTradingEndpoints:
    """Test trading endpoints."""

    def test_get_config(self):
        """Can get trading configuration."""
        client = TestClient(app)
        response = client.get("/api/trading/config")
        assert response.status_code == 200
        data = response.json()
        assert data["trading_pair"] == "BTCUSDT"
        assert "1h" in data["timeframes"]
        assert data["live_mode"] is False

    def test_get_position(self):
        """Can get current position."""
        client = TestClient(app)
        response = client.get("/api/trading/position")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert "quantity" in data

    def test_get_orders(self):
        """Can get orders list."""
        client = TestClient(app)
        response = client.get("/api/trading/orders")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_metrics(self):
        """Can get trading metrics."""
        client = TestClient(app)
        response = client.get("/api/trading/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "max_drawdown" in data
