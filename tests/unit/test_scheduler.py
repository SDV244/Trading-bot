"""Tests for background scheduler behavior."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from packages.core.config import reload_settings
from packages.core.reconciliation import ReconciliationGuardResult, ReconciliationResult
from packages.core.scheduler import TradingScheduler
from packages.core.trading_cycle import CycleResult


@pytest.mark.asyncio
async def test_scheduler_runs_periodic_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduler should run configured periodic reconciliation and expose it in last_result."""
    monkeypatch.setenv("TRADING_MIN_CYCLE_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("TRADING_RECONCILIATION_INTERVAL_CYCLES", "1")
    monkeypatch.setenv("TRADING_ADVISOR_INTERVAL_CYCLES", "1000")
    reload_settings()

    scheduler = TradingScheduler(interval_seconds=1, advisor_interval_cycles=1000)
    cycle_result = CycleResult(
        symbol="BTCUSDT",
        signal_side="HOLD",
        signal_reason="test",
        risk_action="HOLD",
        risk_reason="test",
        executed=False,
        order_id=None,
        fill_id=None,
        quantity="0",
        price=None,
    )
    mock_approval_gate = AsyncMock()
    mock_approval_gate.expire_pending = AsyncMock()
    mock_approval_gate.list_approvals = AsyncMock(return_value=[])
    mock_advisor = AsyncMock()
    mock_advisor.generate_proposals = AsyncMock(return_value=[])
    mock_service = AsyncMock()
    mock_service.symbol = "BTCUSDT"
    mock_service.run_once = AsyncMock(return_value=cycle_result)
    mock_fetcher = AsyncMock()
    mock_fetcher.update_all_timeframes = AsyncMock(return_value={"1h": 5, "4h": 5})
    mock_webhook = AsyncMock()
    mock_webhook.send_info = AsyncMock(return_value=False)
    mock_webhook.send_critical_alert = AsyncMock(return_value=False)
    mock_telegram = AsyncMock()
    mock_telegram.enabled = False
    mock_telegram.send_info = AsyncMock(return_value=False)
    mock_reconciliation = ReconciliationGuardResult(
        result=ReconciliationResult(
            mode="paper",
            db_equity=10000,
            exchange_equity=10000,
            difference=0,
            within_warning_tolerance=True,
            within_critical_tolerance=True,
            reason="paper_db_vs_position_equity",
        ),
        event_type="balance_reconciliation_ok",
        summary="ok",
        emergency_stop_triggered=False,
    )

    with (
        patch("packages.core.scheduler.get_session") as mock_get_session,
        patch("packages.core.scheduler.get_trading_cycle_service", return_value=mock_service),
        patch("packages.core.scheduler.get_approval_gate", return_value=mock_approval_gate),
        patch("packages.core.scheduler.get_ai_advisor", return_value=mock_advisor),
        patch("packages.core.scheduler.get_data_fetcher", return_value=mock_fetcher),
        patch("packages.core.scheduler.run_reconciliation_guard", new_callable=AsyncMock) as mock_guard,
        patch("packages.core.scheduler.get_webhook_notifier", return_value=mock_webhook),
        patch("packages.core.scheduler.get_telegram_notifier", return_value=mock_telegram),
    ):
        mock_get_session.return_value.__aenter__.return_value = AsyncMock()
        mock_get_session.return_value.__aexit__.return_value = False
        mock_guard.return_value = mock_reconciliation

        scheduler.start(interval_seconds=1)
        await asyncio.sleep(1.2)
        await scheduler.stop()

    assert mock_guard.await_count >= 1
    status = scheduler.status()
    assert status.last_result is not None
    assert status.last_result["reconciliation"]["within_critical_tolerance"] is True
    assert status.last_result["market_data"]["1h"] == 5


@pytest.mark.asyncio
async def test_scheduler_sends_periodic_trade_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduler should emit Telegram summary when summary interval elapses."""
    monkeypatch.setenv("TELEGRAM_SUMMARIES_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_SUMMARY_HOURS", "1")
    monkeypatch.setenv("TELEGRAM_DAILY_WRAP_ENABLED", "false")
    reload_settings()

    scheduler = TradingScheduler(interval_seconds=60)
    scheduler._last_trade_summary_at = datetime.now(UTC) - timedelta(hours=2)

    mock_notifier = AsyncMock()
    mock_notifier.enabled = True
    mock_notifier.send_info = AsyncMock(return_value=True)

    mock_webhook = AsyncMock()
    mock_webhook.send_info = AsyncMock(return_value=True)

    payload = {
        "fills_count": 1,
        "buy_count": 1,
        "sell_count": 0,
        "fees_paid": "0.10",
        "traded_notional": "100.00",
        "equity": "10000.00",
    }

    with (
        patch("packages.core.scheduler.get_telegram_notifier", return_value=mock_notifier),
        patch("packages.core.scheduler.get_webhook_notifier", return_value=mock_webhook),
        patch(
            "packages.core.scheduler.TradingScheduler._build_trade_summary_window",
            new=AsyncMock(return_value=("summary_text", payload)),
        ),
    ):
        await scheduler._maybe_send_trade_summaries()

    mock_notifier.send_info.assert_awaited_once()
