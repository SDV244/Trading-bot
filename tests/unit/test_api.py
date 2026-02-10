"""
Tests for API endpoints.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

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
        with patch("apps.api.routers.health._check_database_ready", new=AsyncMock(return_value=(True, None))), \
             patch("apps.api.routers.health._check_binance_ready", new=AsyncMock(return_value=(True, None))):
            response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["database"] is True
        assert data["binance"] is True

    def test_readiness_check_returns_503_when_dependency_down(self):
        """Readiness endpoint returns 503 when dependencies fail."""
        client = TestClient(app)
        with patch(
            "apps.api.routers.health._check_database_ready",
            new=AsyncMock(return_value=(True, None)),
        ), patch(
            "apps.api.routers.health._check_binance_ready",
            new=AsyncMock(return_value=(False, "timeout")),
        ):
            response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["binance"] is False
        assert any("binance:" in reason for reason in data["reasons"])

    def test_prometheus_metrics_endpoint(self):
        """Prometheus metrics endpoint is exposed by default."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "trading_scheduler_cycles_total" in response.text

    def test_rate_limit_blocks_excess_requests(self):
        """Rate limiter returns 429 when per-minute quota is exceeded."""
        with patch.dict(
            "os.environ",
            {
                "API_RATE_LIMIT_ENABLED": "true",
                "API_RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE": "5",
                "API_RATE_LIMIT_EXEMPT_PATHS": "/health,/ready,/metrics,/docs,/openapi.json,/redoc",
            },
        ):
            reload_settings()
            client = TestClient(app)
            headers = {"X-Forwarded-For": f"198.51.100.{uuid4().int % 200 + 1}"}

            responses = [client.get("/api/auth/me", headers=headers) for _ in range(6)]

        reload_settings()
        assert all(response.status_code == 200 for response in responses[:5])
        assert responses[5].status_code == 429
        assert responses[5].json()["detail"] == "Rate limit exceeded"


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

    def test_scheduler_start_respects_min_cycle_interval(self):
        """Scheduler interval is clamped by TRADING_MIN_CYCLE_INTERVAL_SECONDS."""
        import packages.core.config as config_module

        config_module._settings = None
        client = TestClient(app)

        with patch.dict(
            "os.environ",
            {
                "TRADING_REQUIRE_DATA_READY": "false",
                "TRADING_MIN_CYCLE_INTERVAL_SECONDS": "11",
            },
        ):
            reload_settings()
            client.post("/api/system/scheduler/stop")
            started = client.post("/api/system/scheduler/start?interval_seconds=5")
            assert started.status_code == 200
            assert started.json()["interval_seconds"] == 11
            stopped = client.post("/api/system/scheduler/stop")
            assert stopped.status_code == 200

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

    def test_get_circuit_breakers_status(self):
        """Circuit breaker status endpoint returns adapter diagnostics."""
        client = TestClient(app)
        response = client.get("/api/system/circuit-breakers/status")
        assert response.status_code == 200
        data = response.json()
        assert "breakers" in data
        assert "binance_spot" in data["breakers"]
        assert "binance_live" in data["breakers"]

    def test_reset_spot_circuit_breaker(self):
        """Spot circuit breaker can be reset through API."""
        client = TestClient(app)
        response = client.post("/api/system/circuit-breakers/binance_spot/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "binance_spot"
        assert data["reset"] is True
        assert data["details"]["state"] == "closed"

    def test_config_reload_endpoint_with_live_mode_safeguard(self, monkeypatch):
        """Runtime reload blocks live_mode toggle without restart."""
        monkeypatch.setenv("TRADING_LIVE_MODE", "false")
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
        reload_settings()
        client = TestClient(app)

        monkeypatch.setenv("TRADING_LIVE_MODE", "true")
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema_fast")
        response = client.post("/api/system/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["reloaded"] is True
        assert data["live_mode"] is False
        assert data["active_strategy"] == "trend_ema_fast"
        assert "live_mode changes require restart" in data["message"]

    def test_balance_reconciliation_endpoint(self):
        """Reconciliation endpoint returns status payload."""
        client = TestClient(app)
        response = client.get("/api/system/reconciliation")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] in {"paper", "live"}
        assert "difference" in data
        assert "within_warning_tolerance" in data
        assert "within_critical_tolerance" in data

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
        assert data["min_cycle_interval_seconds"] == 5
        assert data["reconciliation_interval_cycles"] == 30
        assert data["reconciliation_warning_tolerance"] == 1.0
        assert data["reconciliation_critical_tolerance"] == 100.0
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
        assert data["approval_auto_approve_enabled"] is False

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

    def test_get_grid_preview_endpoint(self):
        """Smart-grid preview endpoint is reachable for dashboard UX."""
        client = TestClient(app)
        response = client.get("/api/trading/grid/preview")
        assert response.status_code == 200
        data = response.json()
        assert data is None or "signal_reason" in data

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

    def test_close_all_paper_positions_endpoint(self):
        """Close-all endpoint is available and returns cycle-style payload."""
        client = TestClient(app)
        response = client.post(
            "/api/trading/paper/close-all-positions",
            json={"reason": "test_close_all"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "executed" in data
        assert "risk_reason" in data

    def test_paper_reset_requires_paused_state(self):
        """Paper reset should be blocked unless the system is paused."""
        client = TestClient(app)
        client.post(
            "/api/system/state",
            json={"action": "manual_resume", "reason": "reset if emergency", "changed_by": "test"},
        )
        client.post(
            "/api/system/state",
            json={"action": "resume", "reason": "prepare running", "changed_by": "test"},
        )
        response = client.post(
            "/api/trading/paper/reset",
            json={"reason": "test_reset_requires_paused"},
        )
        assert response.status_code == 409
        assert "requires system state PAUSED" in response.json()["detail"]

    def test_paper_reset_endpoint(self):
        """Paper reset clears paper-mode history and returns counters."""
        client = TestClient(app)
        client.post(
            "/api/system/state",
            json={"action": "manual_resume", "reason": "reset if emergency", "changed_by": "test"},
        )
        client.post(
            "/api/system/state",
            json={"action": "pause", "reason": "prepare paper reset", "changed_by": "test"},
        )
        response = client.post(
            "/api/trading/paper/reset",
            json={"reason": "test_reset"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reset"] is True
        assert data["reason"] == "test_reset"
        assert "deleted_orders" in data
        assert "deleted_fills" in data
        assert "deleted_positions" in data
        assert "deleted_equity_snapshots" in data
        assert "deleted_metrics_snapshots" in data

        # Position endpoint should return an empty paper position after reset.
        position = client.get("/api/trading/position")
        assert position.status_code == 200
        assert position.json()["quantity"] in {"0", "0E-8"}

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

    def test_live_order_idempotency_replays_without_second_execution(self, monkeypatch):
        """Same idempotency key should not execute the exchange call twice."""
        from decimal import Decimal

        from packages.core.execution.live_engine import LiveOrderResult

        monkeypatch.setenv("TRADING_LIVE_MODE", "true")
        monkeypatch.setenv("BINANCE_API_KEY", "key")
        monkeypatch.setenv("BINANCE_API_SECRET", "secret")
        reload_settings()

        client = TestClient(app)
        client.post(
            "/api/system/state",
            json={"action": "manual_resume", "reason": "reset live idempotency test", "changed_by": "test"},
        )
        client.post(
            "/api/system/state",
            json={"action": "resume", "reason": "live idempotency test", "changed_by": "test"},
        )

        mock_execute = AsyncMock(
            return_value=LiveOrderResult(
                accepted=True,
                reason="exchange_status_FILLED",
                order_id="123456",
                client_order_id=f"live_test_idem_{uuid4().hex[:12]}",
                quantity=Decimal("0.0100"),
                price=Decimal("50000"),
            )
        )
        idem_key = f"idem-{uuid4().hex}"
        payload = {
            "side": "BUY",
            "quantity": "0.01",
            "ui_confirmed": True,
            "reauthenticated": True,
            "safety_acknowledged": True,
            "idempotency_key": idem_key,
            "reason": "idempotency_test",
        }

        with patch("packages.core.execution.live_engine.LiveEngine.execute_market_order", mock_execute):
            first = client.post("/api/trading/live/order", json=payload)
            second = client.post("/api/trading/live/order", json=payload)

        assert first.status_code == 200
        assert first.json()["accepted"] is True
        assert second.status_code == 200
        assert second.json()["accepted"] is True
        assert second.json()["reason"] in {
            "idempotent_replay_confirmed",
            "idempotent_replay_confirmed_exchange_only",
        }
        assert mock_execute.await_count == 1


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

    def test_llm_status_endpoint(self):
        """LLM status endpoint is available."""
        client = TestClient(app)
        response = client.get("/api/ai/llm/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "provider" in data
        assert "model" in data

    def test_llm_test_endpoint(self):
        """LLM test endpoint returns adapter diagnostics."""
        client = TestClient(app)
        mock_advisor = Mock()
        mock_advisor.test_llm_connection = AsyncMock(
            return_value={
                "ok": True,
                "provider": "ollama",
                "model": "llama3.1:8b",
                "latency_ms": 15,
                "message": "ok",
                "raw_proposals_count": 1,
            }
        )
        with patch("apps.api.routers.ai.get_ai_advisor", return_value=mock_advisor):
            response = client.post("/api/ai/llm/test")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["provider"] == "ollama"
        assert data["raw_proposals_count"] == 1

    def test_optimizer_endpoints_require_data(self):
        """Optimizer endpoints reject when insufficient data."""
        client = TestClient(app)
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(equity=100.0)] * 10
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        with patch("apps.api.routers.ai.get_session", return_value=mock_session_ctx):
            train = client.post("/api/ai/optimizer/train", json={"timesteps": 512})
            assert train.status_code == 400

            propose = client.get("/api/ai/optimizer/propose")
            assert propose.status_code == 400

    def test_auto_approve_toggle_endpoint(self):
        """Auto-approve mode can be queried and toggled."""
        client = TestClient(app)

        status_before = client.get("/api/ai/auto-approve")
        assert status_before.status_code == 200
        assert "enabled" in status_before.json()

        enabled = client.post(
            "/api/ai/auto-approve",
            json={"enabled": True, "reason": "test_enable", "changed_by": "tester"},
        )
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True

        disabled = client.post(
            "/api/ai/auto-approve",
            json={"enabled": False, "reason": "test_disable", "changed_by": "tester"},
        )
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False

    def test_auto_approve_enable_sweeps_pending_approvals(self, monkeypatch):
        """Enabling auto-approve should immediately process existing pending proposals."""
        monkeypatch.setenv("APPROVAL_AUTO_APPROVE_ENABLED", "false")
        reload_settings()

        async def _create_pending() -> int:
            from packages.core.ai.advisor import AIProposal
            from packages.core.ai.approval_gate import get_approval_gate
            from packages.core.database.session import get_session, init_database

            await init_database()
            async with get_session() as session:
                approval = await get_approval_gate().create_approval(
                    session,
                    AIProposal(
                        title="Pending before toggle",
                        proposal_type="risk_tuning",
                        description="sweep me",
                        diff={"risk": {"per_trade": 0.0042}},
                        expected_impact="test",
                        evidence={},
                        confidence=0.83,
                    ),
                )
                return approval.id

        pending_id = asyncio.run(_create_pending())
        client = TestClient(app)

        enable = client.post(
            "/api/ai/auto-approve",
            json={"enabled": True, "reason": "sweep_pending", "changed_by": "tester"},
        )
        assert enable.status_code == 200
        assert enable.json()["enabled"] is True

        pending_after = client.get("/api/ai/approvals?status=PENDING&limit=200")
        assert pending_after.status_code == 200
        assert all(item["id"] != pending_id for item in pending_after.json())

        all_approvals = client.get("/api/ai/approvals?limit=200")
        assert all_approvals.status_code == 200
        target = next(item for item in all_approvals.json() if item["id"] == pending_id)
        assert target["status"] == "APPROVED"
        assert target["decided_by"] == "ai_auto_approver"

        disable = client.post(
            "/api/ai/auto-approve",
            json={"enabled": False, "reason": "cleanup", "changed_by": "tester"},
        )
        assert disable.status_code == 200
        assert disable.json()["enabled"] is False

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
