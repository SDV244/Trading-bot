"""
Tests for API endpoints.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.config import reload_settings
from packages.core.trading_cycle import DataReadiness, TimeframeReadiness


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
        import packages.core.config as config_module

        config_module._settings = None
        client = TestClient(app)

        status = client.get("/api/system/scheduler")
        assert status.status_code == 200
        assert "running" in status.json()

        # Disable readiness gate for this smoke test.
        with patch.dict("os.environ", {"TRADING_REQUIRE_DATA_READY": "false"}):
            reload_settings()
            started = client.post("/api/system/scheduler/start?interval_seconds=5")
        assert started.status_code == 200
        assert started.json()["running"] is True

        stopped = client.post("/api/system/scheduler/stop")
        assert stopped.status_code == 200
        assert stopped.json()["running"] is False

    def test_scheduler_start_returns_409_when_data_not_ready(self):
        """Scheduler start is blocked when warmup data is missing."""
        client = TestClient(app)

        mock_readiness = DataReadiness(
            symbol="BTCUSDT",
            active_strategy="trend_ema",
            require_data_ready=True,
            data_ready=False,
            reasons=["4h: requires 50 candles, found 12"],
            timeframes={
                "1h": TimeframeReadiness(required=20, available=120, ready=True),
                "4h": TimeframeReadiness(required=50, available=12, ready=False),
            },
        )

        with patch.dict("os.environ", {"TRADING_REQUIRE_DATA_READY": "true"}):
            reload_settings()
            with patch("apps.api.routers.system.get_trading_cycle_service") as mock_service_getter, \
                 patch("apps.api.routers.system.get_session") as mock_session_getter:
                mock_service = AsyncMock()
                mock_service.get_data_readiness.return_value = mock_readiness
                mock_service_getter.return_value = mock_service
                mock_session = AsyncMock()
                mock_session_getter.return_value.__aenter__.return_value = mock_session

                response = client.post("/api/system/scheduler/start?interval_seconds=5")

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["data_ready"] is False
        assert detail["active_strategy"] == "trend_ema"
        assert detail["timeframes"]["4h"]["ready"] is False

    def test_get_system_readiness(self):
        """System readiness endpoint returns data readiness details."""
        client = TestClient(app)
        response = client.get("/api/system/readiness")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "data_ready" in data
        assert "timeframes" in data

    def test_get_notification_status(self):
        """Notification status endpoint returns safe config booleans."""
        client = TestClient(app)
        mock_notifier = AsyncMock()
        mock_notifier.enabled = True
        mock_notifier.bot_token = "token_present"
        mock_notifier.chat_id = "chat_present"
        with patch("apps.api.routers.system.get_telegram_notifier", return_value=mock_notifier):
            response = client.get("/api/system/notifications/status")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["has_bot_token"] is True
        assert data["has_chat_id"] is True

    def test_send_test_notification(self):
        """Test notification endpoint triggers notifier send."""
        client = TestClient(app)
        mock_notifier = AsyncMock()
        mock_notifier.enabled = True
        mock_notifier.bot_token = "token_present"
        mock_notifier.chat_id = "chat_present"
        mock_notifier.send_info = AsyncMock(return_value=True)
        with patch("apps.api.routers.system.get_telegram_notifier", return_value=mock_notifier):
            response = client.post(
                "/api/system/notifications/test",
                json={"title": "API Test", "body": "hello"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["delivered"] is True
        mock_notifier.send_info.assert_awaited_once()


class TestTradingEndpoints:
    """Test trading endpoints."""

    def test_get_config(self, monkeypatch):
        """Can get trading configuration."""
        monkeypatch.setenv("TRADING_LIVE_MODE", "false")
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
        monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "long_flat")
        monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "10000")
        monkeypatch.setenv("TRADING_ADVISOR_INTERVAL_CYCLES", "30")
        monkeypatch.setenv("TRADING_GRID_LEVELS", "6")
        monkeypatch.setenv("TRADING_GRID_MIN_SPACING_BPS", "25")
        monkeypatch.setenv("TRADING_GRID_MAX_SPACING_BPS", "220")
        monkeypatch.setenv("TRADING_GRID_ENFORCE_FEE_FLOOR", "false")
        monkeypatch.setenv("TRADING_GRID_MIN_NET_PROFIT_BPS", "30")
        monkeypatch.setenv("TRADING_GRID_COOLDOWN_SECONDS", "0")
        monkeypatch.setenv("TRADING_GRID_RECENTER_MODE", "aggressive")
        reload_settings()
        client = TestClient(app)
        response = client.get("/api/trading/config")
        assert response.status_code == 200
        data = response.json()
        assert data["trading_pair"] == "BTCUSDT"
        assert "1h" in data["timeframes"]
        assert data["live_mode"] is False
        assert data["active_strategy"] == "smart_grid_ai"
        assert data["require_data_ready"] is True
        assert data["spot_position_mode"] == "long_flat"
        assert data["paper_starting_equity"] == 10000.0
        assert "smart_grid_ai" in data["supported_strategies"]
        assert data["advisor_interval_cycles"] == 30
        assert data["grid_lookback_1h"] == 120
        assert data["grid_atr_period_1h"] == 14
        assert data["grid_levels"] == 6
        assert data["grid_spacing_mode"] == "geometric"
        assert data["grid_auto_inventory_bootstrap"] is True
        assert data["grid_enforce_fee_floor"] is False
        assert data["grid_min_net_profit_bps"] == 30
        assert data["grid_recenter_mode"] == "aggressive"
        assert data["stop_loss_enabled"] is True
        assert data["stop_loss_global_equity_pct"] == 0.15
        assert data["stop_loss_max_drawdown_pct"] == 0.2

    def test_set_grid_recenter_mode(self, monkeypatch):
        """Can update smart-grid recenter mode at runtime from API."""
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
        monkeypatch.setenv("TRADING_GRID_RECENTER_MODE", "aggressive")
        reload_settings()
        client = TestClient(app)

        response = client.post(
            "/api/trading/config/recenter-mode",
            json={"mode": "conservative", "reason": "test", "changed_by": "tester"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "conservative"
        assert data["active_strategy"] == "smart_grid_ai"
        assert data["applied_to_live_strategy"] is True

        updated = client.get("/api/trading/config")
        assert updated.status_code == 200
        assert updated.json()["grid_recenter_mode"] == "conservative"

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
