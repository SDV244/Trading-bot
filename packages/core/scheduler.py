"""Background scheduler for periodic paper trading cycles."""

import asyncio
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy import func, select

from packages.adapters.telegram_bot import get_telegram_notifier
from packages.adapters.webhook_notifier import get_webhook_notifier
from packages.core.ai import get_ai_advisor, get_approval_gate
from packages.core.config import get_settings
from packages.core.data_fetcher import get_data_fetcher
from packages.core.database.models import EquitySnapshot, Fill, Order, Position
from packages.core.database.session import get_session
from packages.core.execution_lock import symbol_execution_lock
from packages.core.observability import (
    increment_approval,
    observe_scheduler_cycle,
    set_scheduler_running,
)
from packages.core.reconciliation import run_reconciliation_guard
from packages.core.state import get_state_manager
from packages.core.trading_cycle import CycleResult, get_trading_cycle_service


@dataclass(slots=True)
class SchedulerStatus:
    """Current scheduler status."""

    running: bool
    interval_seconds: int
    last_run_at: datetime | None
    last_error: str | None
    last_result: dict[str, Any] | None


class TradingScheduler:
    """Simple async loop scheduler for paper cycle execution."""

    def __init__(self, interval_seconds: int = 60, advisor_interval_cycles: int | None = None) -> None:
        settings = get_settings()
        self.interval_seconds = interval_seconds
        interval = (
            advisor_interval_cycles
            if advisor_interval_cycles is not None
            else settings.trading.advisor_interval_cycles
        )
        self.advisor_interval_cycles = max(1, interval)
        self.min_cycle_interval_seconds = max(1, settings.trading.min_cycle_interval_seconds)
        self.reconciliation_interval_cycles = max(0, settings.trading.reconciliation_interval_cycles)
        self.reconciliation_warning_tolerance = Decimal(
            str(settings.trading.reconciliation_warning_tolerance)
        )
        self.reconciliation_critical_tolerance = Decimal(
            str(settings.trading.reconciliation_critical_tolerance)
        )
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_run_at: datetime | None = None
        self._last_error: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._cycles = 0
        self._last_heartbeat_at: datetime | None = None
        self._last_trade_summary_at: datetime | None = None
        self._last_daily_wrap_at: datetime | None = None
        self._last_strategy_digest_at: datetime | None = None
        self._last_cycle_started_monotonic: float | None = None
        set_scheduler_running(False)

    @property
    def running(self) -> bool:
        """Whether scheduler is currently active."""
        return self._running

    def status(self) -> SchedulerStatus:
        """Get scheduler status snapshot."""
        return SchedulerStatus(
            running=self._running,
            interval_seconds=self.interval_seconds,
            last_run_at=self._last_run_at,
            last_error=self._last_error,
            last_result=self._last_result,
        )

    def start(self, interval_seconds: int | None = None) -> bool:
        """
        Start scheduler loop.

        Returns True if started, False if already running.
        """
        if self._running:
            return False
        settings = get_settings()
        self.min_cycle_interval_seconds = max(1, settings.trading.min_cycle_interval_seconds)
        self.reconciliation_interval_cycles = max(0, settings.trading.reconciliation_interval_cycles)
        self.reconciliation_warning_tolerance = Decimal(
            str(settings.trading.reconciliation_warning_tolerance)
        )
        self.reconciliation_critical_tolerance = Decimal(
            str(settings.trading.reconciliation_critical_tolerance)
        )
        if interval_seconds is not None:
            self.interval_seconds = max(self.min_cycle_interval_seconds, interval_seconds)
        else:
            self.interval_seconds = max(self.interval_seconds, self.min_cycle_interval_seconds)

        self._running = True
        set_scheduler_running(True)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Trading scheduler started (interval={self.interval_seconds}s)")
        return True

    async def stop(self) -> bool:
        """
        Stop scheduler loop.

        Returns True if stopped, False if it was not running.
        """
        if not self._running:
            return False

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        set_scheduler_running(False)
        logger.info("Trading scheduler stopped")
        return True

    async def _run_loop(self) -> None:
        next_run_monotonic = time.monotonic() + self.interval_seconds
        while self._running:
            wait_for = max(0.0, next_run_monotonic - time.monotonic())
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            if not self._running:
                break
            runtime_settings = get_settings()
            self.min_cycle_interval_seconds = max(1, runtime_settings.trading.min_cycle_interval_seconds)
            self.reconciliation_interval_cycles = max(0, runtime_settings.trading.reconciliation_interval_cycles)
            self.reconciliation_warning_tolerance = Decimal(
                str(runtime_settings.trading.reconciliation_warning_tolerance)
            )
            self.reconciliation_critical_tolerance = Decimal(
                str(runtime_settings.trading.reconciliation_critical_tolerance)
            )

            cycle_started_monotonic = time.monotonic()
            observed_interval: float | None = None
            if self._last_cycle_started_monotonic is not None:
                observed_interval = cycle_started_monotonic - self._last_cycle_started_monotonic
            self._last_cycle_started_monotonic = cycle_started_monotonic
            self._last_run_at = datetime.now(UTC)
            try:
                market_data_payload: dict[str, int] | None = None
                try:
                    market_data_payload = await get_data_fetcher().update_all_timeframes()
                except Exception as fetch_error:
                    logger.warning(f"Market data refresh failed before cycle: {fetch_error}")

                async with get_session() as session:
                    await get_approval_gate().expire_pending(session)
                    next_cycle = self._cycles + 1
                    reconciliation_payload: dict[str, Any] | None = None
                    if (
                        self.reconciliation_interval_cycles > 0
                        and next_cycle % self.reconciliation_interval_cycles == 0
                    ):
                        reconciliation_outcome = await run_reconciliation_guard(
                            session,
                            warning_tolerance=self.reconciliation_warning_tolerance,
                            critical_tolerance=self.reconciliation_critical_tolerance,
                            actor="scheduler",
                        )
                        reconciliation_result = reconciliation_outcome.result
                        reconciliation_payload = {
                            "mode": reconciliation_result.mode,
                            "difference": str(reconciliation_result.difference),
                            "within_warning_tolerance": reconciliation_result.within_warning_tolerance,
                            "within_critical_tolerance": reconciliation_result.within_critical_tolerance,
                            "reason": reconciliation_result.reason,
                            "emergency_stop_triggered": reconciliation_outcome.emergency_stop_triggered,
                        }

                    service = get_trading_cycle_service()
                    async with symbol_execution_lock(service.symbol):
                        result = await service.run_once(session)
                    self._cycles = next_cycle

                    if self._cycles % self.advisor_interval_cycles == 0:
                        pending = await get_approval_gate().list_approvals(
                            session, status="PENDING", limit=1
                        )
                        if not pending:
                            proposals = await get_ai_advisor().generate_proposals(session)
                            if proposals:
                                max_to_enqueue = max(1, min(len(proposals), get_settings().multiagent.max_proposals))
                                for proposal in proposals[:max_to_enqueue]:
                                    created = await get_approval_gate().create_approval(session, proposal)
                                    if created.status == "APPROVED":
                                        increment_approval("auto_approved")
                                    else:
                                        increment_approval("created")

                self._last_result = self._serialize_result(
                    result,
                    reconciliation=reconciliation_payload,
                    market_data=market_data_payload,
                )
                self._last_error = None
                observe_scheduler_cycle(
                    duration_seconds=time.monotonic() - cycle_started_monotonic,
                    status="success",
                    interval_seconds=observed_interval,
                )
                await self._maybe_send_heartbeat()
                await self._maybe_send_trade_summaries()
                await self._maybe_send_strategy_digest()
            except Exception as e:
                self._last_error = str(e)
                logger.exception("Trading scheduler cycle failed")
                observe_scheduler_cycle(
                    duration_seconds=time.monotonic() - cycle_started_monotonic,
                    status="error",
                    interval_seconds=observed_interval,
                )
                await get_webhook_notifier().send_critical_alert(
                    "Scheduler cycle failed",
                    str(e),
                )
            next_run_monotonic = max(
                cycle_started_monotonic + self.interval_seconds,
                cycle_started_monotonic + self.min_cycle_interval_seconds,
            )

    @staticmethod
    def _serialize_result(
        result: CycleResult,
        reconciliation: dict[str, Any] | None = None,
        market_data: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        payload = asdict(result)
        if reconciliation is not None:
            payload["reconciliation"] = reconciliation
        if market_data is not None:
            payload["market_data"] = market_data
        payload["executed_at"] = datetime.now(UTC).isoformat()
        return payload

    async def _maybe_send_heartbeat(self) -> None:
        settings = get_settings()
        if not settings.telegram.heartbeat_enabled:
            return
        interval_hours = max(1, settings.telegram.heartbeat_hours)
        now = datetime.now(UTC)
        if self._last_heartbeat_at is not None:
            elapsed_hours = (now - self._last_heartbeat_at).total_seconds() / 3600
            if elapsed_hours < interval_hours:
                return

        notifier = get_telegram_notifier()
        if not notifier.enabled:
            return
        state = get_state_manager().current
        last_result = self._last_result or {}
        sent = await notifier.send_info(
            "Trading bot heartbeat",
            (
                f"State: {state.state.value}\n"
                f"Scheduler running: {self._running}\n"
                f"Cycles: {self._cycles}\n"
                f"Last signal: {last_result.get('signal_side', 'N/A')}\n"
                f"Last risk reason: {last_result.get('risk_reason', 'N/A')}"
            ),
        )
        if sent:
            self._last_heartbeat_at = now
        await get_webhook_notifier().send_info(
            "Trading bot heartbeat",
            (
                f"State: {state.state.value}\n"
                f"Scheduler running: {self._running}\n"
                f"Cycles: {self._cycles}"
            ),
        )

    async def _maybe_send_trade_summaries(self) -> None:
        settings = get_settings()
        notifier = get_telegram_notifier()
        if not notifier.enabled:
            return

        now = datetime.now(UTC)
        if settings.telegram.summaries_enabled:
            summary_interval_hours = max(1, settings.telegram.summary_hours)
            if self._last_trade_summary_at is None:
                self._last_trade_summary_at = now
            else:
                elapsed_hours = (now - self._last_trade_summary_at).total_seconds() / 3600
                if elapsed_hours >= summary_interval_hours:
                    summary_text, payload = await self._build_trade_summary_window(summary_interval_hours)
                    sent = await notifier.send_info(
                        f"{summary_interval_hours}h trading summary",
                        summary_text,
                    )
                    if sent:
                        self._last_trade_summary_at = now
                    await get_webhook_notifier().send_info(
                        f"{summary_interval_hours}h trading summary",
                        (
                            f"fills={payload['fills_count']} "
                            f"buys={payload['buy_count']} sells={payload['sell_count']} "
                            f"fees={payload['fees_paid']}"
                        ),
                    )

        if settings.telegram.daily_wrap_enabled:
            if self._last_daily_wrap_at is None:
                self._last_daily_wrap_at = now
            else:
                elapsed_hours = (now - self._last_daily_wrap_at).total_seconds() / 3600
                if elapsed_hours >= 24:
                    summary_text, payload = await self._build_trade_summary_window(24)
                    sent = await notifier.send_info("Daily trading wrap-up", summary_text)
                    if sent:
                        self._last_daily_wrap_at = now
                    await get_webhook_notifier().send_info(
                        "Daily trading wrap-up",
                        (
                            f"fills={payload['fills_count']} "
                            f"notional={payload['traded_notional']} "
                            f"equity={payload['equity']}"
                        ),
                    )

    async def _maybe_send_strategy_digest(self) -> None:
        """Send periodic AI strategy diagnostics and top recommendations."""
        notifier = get_telegram_notifier()
        if not notifier.enabled:
            return

        now = datetime.now(UTC)
        interval_hours = 6
        if self._last_strategy_digest_at is None:
            self._last_strategy_digest_at = now
            return

        elapsed_hours = (now - self._last_strategy_digest_at).total_seconds() / 3600
        if elapsed_hours < interval_hours:
            return

        async with get_session() as session:
            analysis = await get_ai_advisor().analyze_strategy(session)

        recommendations = analysis.get("recommendations", [])
        top_recommendations = recommendations[:3]
        active_strategy = str(analysis.get("active_strategy", "unknown"))
        symbol = str(analysis.get("symbol", "unknown"))
        regime_info = analysis.get("regime_analysis", {})
        regime_label = str(regime_info.get("label", "unknown")) if isinstance(regime_info, dict) else "unknown"
        hold_diagnostics = analysis.get("hold_diagnostics", {})
        hold_reason = (
            str(hold_diagnostics.get("primary_reason", "none"))
            if isinstance(hold_diagnostics, dict)
            else "none"
        )

        if top_recommendations:
            recommendation_lines: list[str] = []
            for idx, item in enumerate(top_recommendations, start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "Untitled recommendation"))
                confidence = float(item.get("confidence", 0.0))
                recommendation_lines.append(f"{idx}. {title} (conf={confidence:.2f})")
            recommendation_text = "\n".join(recommendation_lines) if recommendation_lines else "No actionable recommendation."
        else:
            recommendation_text = "No actionable recommendation."

        body = (
            f"Symbol: {symbol}\n"
            f"Strategy: {active_strategy}\n"
            f"Regime: {regime_label}\n"
            f"Primary hold reason: {hold_reason}\n"
            "Top recommendations:\n"
            f"{recommendation_text}"
        )
        sent = await notifier.send_info(f"{interval_hours}h strategy digest", body)
        if sent:
            self._last_strategy_digest_at = now

        await get_webhook_notifier().send_info(
            f"{interval_hours}h strategy digest",
            (
                f"strategy={active_strategy} "
                f"symbol={symbol} "
                f"regime={regime_label} "
                f"recommendations={len(top_recommendations)}"
            ),
        )

    async def _build_trade_summary_window(self, window_hours: int) -> tuple[str, dict[str, str | int]]:
        now = datetime.now(UTC)
        window_start = now - timedelta(hours=window_hours)
        is_paper_mode = not get_settings().trading.live_mode
        mode_label = "paper" if is_paper_mode else "live"
        async with get_session() as session:
            fills_count = (
                await session.execute(
                    select(func.count(Fill.id)).where(
                        Fill.is_paper.is_(is_paper_mode),
                        Fill.filled_at >= window_start,
                    )
                )
            ).scalar_one()
            buy_count = (
                await session.execute(
                    select(func.count(Fill.id))
                    .join(Order, Fill.order_id == Order.id)
                    .where(
                        Fill.is_paper.is_(is_paper_mode),
                        Fill.filled_at >= window_start,
                        Order.side == "BUY",
                    )
                )
            ).scalar_one()
            sell_count = (
                await session.execute(
                    select(func.count(Fill.id))
                    .join(Order, Fill.order_id == Order.id)
                    .where(
                        Fill.is_paper.is_(is_paper_mode),
                        Fill.filled_at >= window_start,
                        Order.side == "SELL",
                    )
                )
            ).scalar_one()
            traded_notional_raw = (
                await session.execute(
                    select(func.coalesce(func.sum(Fill.quantity * Fill.price), 0)).where(
                        Fill.is_paper.is_(is_paper_mode),
                        Fill.filled_at >= window_start,
                    )
                )
            ).scalar_one()
            fees_paid_raw = (
                await session.execute(
                    select(func.coalesce(func.sum(Fill.fee), 0)).where(
                        Fill.is_paper.is_(is_paper_mode),
                        Fill.filled_at >= window_start,
                    )
                )
            ).scalar_one()
            position = (
                await session.execute(
                    select(Position).where(
                        Position.symbol == get_trading_cycle_service().symbol,
                        Position.is_paper.is_(is_paper_mode),
                    )
                )
            ).scalar_one_or_none()
            equity_snapshot = (
                await session.execute(
                    select(EquitySnapshot)
                    .where(EquitySnapshot.is_paper.is_(is_paper_mode))
                    .order_by(EquitySnapshot.snapshot_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

        traded_notional = Decimal(str(traded_notional_raw))
        fees_paid = Decimal(str(fees_paid_raw))
        realized_pnl = Decimal(str(position.realized_pnl)) if position is not None else Decimal("0")
        unrealized_pnl = Decimal(str(position.unrealized_pnl)) if position is not None else Decimal("0")
        position_qty = Decimal(str(position.quantity)) if position is not None else Decimal("0")
        equity = (
            Decimal(str(equity_snapshot.equity))
            if equity_snapshot is not None
            else Decimal(str(get_settings().trading.paper_starting_equity))
        )
        window_text = f"{window_start.strftime('%Y-%m-%d %H:%MZ')} -> {now.strftime('%Y-%m-%d %H:%MZ')}"
        summary_text = (
            f"Window: {window_text}\n"
            f"Mode: {mode_label}\n"
            f"Symbol: {get_trading_cycle_service().symbol}\n"
            f"Fills: {fills_count} (BUY {buy_count} | SELL {sell_count})\n"
            f"Traded notional: {self._fmt_decimal(traded_notional, 2)} USDT\n"
            f"Fees: {self._fmt_decimal(fees_paid, 4)} USDT\n"
            f"Position qty: {self._fmt_decimal(position_qty, 6)}\n"
            f"Realized PnL: {self._fmt_decimal(realized_pnl, 2)} USDT\n"
            f"Unrealized PnL: {self._fmt_decimal(unrealized_pnl, 2)} USDT\n"
            f"Equity: {self._fmt_decimal(equity, 2)} USDT"
        )
        payload: dict[str, str | int] = {
            "window": window_text,
            "fills_count": int(fills_count),
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
            "traded_notional": self._fmt_decimal(traded_notional, 2),
            "fees_paid": self._fmt_decimal(fees_paid, 4),
            "position_qty": self._fmt_decimal(position_qty, 6),
            "realized_pnl": self._fmt_decimal(realized_pnl, 2),
            "unrealized_pnl": self._fmt_decimal(unrealized_pnl, 2),
            "equity": self._fmt_decimal(equity, 2),
        }
        return summary_text, payload

    @staticmethod
    def _fmt_decimal(value: Decimal, places: int) -> str:
        quant = Decimal(1).scaleb(-places)
        return str(value.quantize(quant))


_trading_scheduler: TradingScheduler | None = None
_trading_scheduler_lock = threading.Lock()


def get_trading_scheduler() -> TradingScheduler:
    """Get or create trading scheduler singleton."""
    global _trading_scheduler
    if _trading_scheduler is None:
        with _trading_scheduler_lock:
            if _trading_scheduler is None:
                _trading_scheduler = TradingScheduler()
    return _trading_scheduler


async def close_trading_scheduler() -> None:
    """Stop scheduler if running."""
    global _trading_scheduler
    if _trading_scheduler is not None:
        await _trading_scheduler.stop()
        _trading_scheduler = None
