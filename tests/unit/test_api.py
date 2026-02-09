"""
Tests for API endpoints.
"""

import asyncio

from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.config import reload_settings


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
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

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

        # Ensure we are not in emergency-stop from prior tests.
        client.post(
            "/api/system/state",
            json={"action": "manual_resume", "reason": "Reset if needed", "changed_by": "test"},
        )

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

    def test_scheduler_endpoints(self):
        """Scheduler status/start/stop endpoints work."""
        client = TestClient(app)

        status = client.get("/api/system/scheduler")
        assert status.status_code == 200
        assert "running" in status.json()

        started = client.post("/api/system/scheduler/start?interval_seconds=5")
        assert started.status_code == 200
        assert started.json()["running"] is True

        stopped = client.post("/api/system/scheduler/stop")
        assert stopped.status_code == 200
        assert stopped.json()["running"] is False


class TestTradingEndpoints:
    """Test trading endpoints."""

    def test_get_config(self, monkeypatch):
        """Can get trading configuration."""
        monkeypatch.setenv("TRADING_LIVE_MODE", "false")
        reload_settings()
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

    def test_get_fills(self):
        """Can get fills list."""
        client = TestClient(app)
        response = client.get("/api/trading/fills")
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

    def test_run_paper_cycle(self):
        """Can trigger one paper trading cycle."""
        client = TestClient(app)
        response = client.post("/api/trading/paper/cycle")
        assert response.status_code == 200
        data = response.json()
        assert "executed" in data
        assert "signal_side" in data
        assert "risk_action" in data

    def test_live_order_rejected_when_live_mode_disabled(self):
        """Live endpoint blocks orders when live mode is disabled."""
        client = TestClient(app)
        response = client.post(
            "/api/trading/live/order",
            json={
                "side": "BUY",
                "quantity": "0.01",
                "ui_confirmed": True,
                "reauthenticated": True,
                "safety_acknowledged": True,
                "reason": "test",
            },
        )
        assert response.status_code == 400


class TestAIEndpoints:
    """Test AI and approvals endpoints."""

    def test_list_approvals(self):
        """Can list approvals."""
        client = TestClient(app)
        response = client.get("/api/ai/approvals")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_events_endpoint(self):
        """Can list audit events."""
        client = TestClient(app)
        response = client.get("/api/ai/events?limit=20")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_optimizer_endpoints_require_data(self):
        """Optimizer endpoints reject when insufficient data."""
        client = TestClient(app)
        train = client.post("/api/ai/optimizer/train", json={"timesteps": 512})
        assert train.status_code == 400

        propose = client.get("/api/ai/optimizer/propose")
        assert propose.status_code == 400

    def test_approve_reject_flow(self):
        """Can reject and approve pending approvals."""

        async def _create_pending() -> int:
            from packages.core.ai.advisor import AIProposal
            from packages.core.ai.approval_gate import get_approval_gate
            from packages.core.database.session import get_session, init_database

            await init_database()
            async with get_session() as session:
                approval = await get_approval_gate().create_approval(
                    session,
                    AIProposal(
                        title="API approval test",
                        proposal_type="risk_tuning",
                        description="test",
                        diff={"risk": {"per_trade": 0.004}},
                        expected_impact="test",
                        evidence={},
                        confidence=0.8,
                    ),
                )
                return approval.id

        pending_id = asyncio.run(_create_pending())
        client = TestClient(app)

        reject = client.post(
            f"/api/ai/approvals/{pending_id}/reject",
            json={"decided_by": "test_user", "reason": "no"},
        )
        assert reject.status_code == 200
        assert reject.json()["status"] == "REJECTED"

        pending_id_2 = asyncio.run(_create_pending())
        approve = client.post(
            f"/api/ai/approvals/{pending_id_2}/approve",
            json={"decided_by": "test_user", "reason": "yes"},
        )
        assert approve.status_code == 200
        assert approve.json()["status"] == "APPROVED"
