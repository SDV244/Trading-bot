"""Balance reconciliation utilities for paper/live operation checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.binance_live import get_binance_live_adapter
from packages.adapters.binance_spot import get_binance_adapter
from packages.adapters.telegram_bot import get_telegram_notifier
from packages.adapters.webhook_notifier import get_webhook_notifier
from packages.core.audit import log_event
from packages.core.config import get_settings
from packages.core.database.models import EquitySnapshot, Position
from packages.core.state import get_state_manager


@dataclass(slots=True, frozen=True)
class ReconciliationResult:
    """Single reconciliation run result."""

    mode: Literal["paper", "live"]
    db_equity: Decimal
    exchange_equity: Decimal | None
    difference: Decimal
    within_warning_tolerance: bool
    within_critical_tolerance: bool
    reason: str


@dataclass(slots=True, frozen=True)
class ReconciliationGuardResult:
    """Reconciliation run with emitted audit/event information."""

    result: ReconciliationResult
    event_type: str
    summary: str
    emergency_stop_triggered: bool


class BalanceReconciler:
    """Compare DB equity and exchange/derived balances."""

    def __init__(
        self,
        *,
        warning_tolerance: Decimal = Decimal("1"),
        critical_tolerance: Decimal = Decimal("100"),
    ) -> None:
        self.settings = get_settings()
        self.warning_tolerance = warning_tolerance
        self.critical_tolerance = critical_tolerance

    async def reconcile(self, session: AsyncSession) -> ReconciliationResult:
        if self.settings.trading.live_mode:
            return await self._reconcile_live(session)
        return await self._reconcile_paper(session)

    async def _latest_equity(self, session: AsyncSession, *, is_paper: bool) -> Decimal:
        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.is_paper.is_(is_paper))
            .order_by(EquitySnapshot.snapshot_at.desc())
            .limit(1)
        )
        snapshot = result.scalar_one_or_none()
        if snapshot is None:
            return Decimal(str(self.settings.trading.paper_starting_equity))
        return Decimal(str(snapshot.equity))

    async def _position(self, session: AsyncSession, *, is_paper: bool) -> Position | None:
        result = await session.execute(
            select(Position).where(
                Position.is_paper.is_(is_paper),
                Position.symbol == self.settings.trading.pair,
            )
        )
        return result.scalar_one_or_none()

    async def _reconcile_paper(self, session: AsyncSession) -> ReconciliationResult:
        db_equity = await self._latest_equity(session, is_paper=True)
        position = await self._position(session, is_paper=True)
        if position is None:
            derived_equity = Decimal(str(self.settings.trading.paper_starting_equity))
        else:
            derived_equity = (
                Decimal(str(self.settings.trading.paper_starting_equity))
                + Decimal(str(position.realized_pnl))
                + Decimal(str(position.unrealized_pnl))
            )
        difference = abs(db_equity - derived_equity)
        return ReconciliationResult(
            mode="paper",
            db_equity=db_equity,
            exchange_equity=derived_equity,
            difference=difference,
            within_warning_tolerance=difference <= self.warning_tolerance,
            within_critical_tolerance=difference <= self.critical_tolerance,
            reason="paper_db_vs_position_equity",
        )

    async def _reconcile_live(self, session: AsyncSession) -> ReconciliationResult:
        db_equity = await self._latest_equity(session, is_paper=False)
        balances = await get_binance_live_adapter().get_account_balances()
        symbol = self.settings.trading.pair
        base_asset = symbol[:-4]
        quote_asset = symbol[-4:]
        mark_price = await get_binance_adapter().get_ticker_price(symbol)
        exchange_equity = (
            balances.get(quote_asset, Decimal("0"))
            + (balances.get(base_asset, Decimal("0")) * Decimal(str(mark_price)))
        )
        difference = abs(db_equity - exchange_equity)
        return ReconciliationResult(
            mode="live",
            db_equity=db_equity,
            exchange_equity=exchange_equity,
            difference=difference,
            within_warning_tolerance=difference <= self.warning_tolerance,
            within_critical_tolerance=difference <= self.critical_tolerance,
            reason="live_db_vs_exchange_equity",
        )


async def run_reconciliation_guard(
    session: AsyncSession,
    *,
    warning_tolerance: Decimal = Decimal("1"),
    critical_tolerance: Decimal = Decimal("100"),
    actor: str = "system",
) -> ReconciliationGuardResult:
    """Run reconciliation and apply warning/critical guard side effects."""
    reconciler = BalanceReconciler(
        warning_tolerance=warning_tolerance,
        critical_tolerance=critical_tolerance,
    )
    result = await reconciler.reconcile(session)

    event_type = "balance_reconciliation_ok"
    summary = f"Balance reconciliation {result.mode} diff={result.difference}"
    emergency_stop_triggered = False

    if not result.within_warning_tolerance:
        event_type = "balance_reconciliation_warning"
        summary = f"Balance reconciliation warning diff={result.difference}"

    if not result.within_critical_tolerance:
        event_type = "balance_reconciliation_critical"
        summary = f"Balance reconciliation critical diff={result.difference}"
        get_state_manager().force_emergency_stop(
            reason="balance_reconciliation_critical_mismatch",
            changed_by="reconciliation_guard",
            metadata={
                "mode": result.mode,
                "difference": str(result.difference),
                "critical_tolerance": str(critical_tolerance),
            },
        )
        emergency_stop_triggered = True
        await get_telegram_notifier().send_critical_alert(
            "Balance reconciliation critical mismatch",
            (
                f"Mode: {result.mode}\n"
                f"Difference: {result.difference}\n"
                f"Tolerance: {critical_tolerance}"
            ),
        )
        await get_webhook_notifier().send_critical_alert(
            "Balance reconciliation critical mismatch",
            (
                f"Mode: {result.mode}\n"
                f"Difference: {result.difference}\n"
                f"Tolerance: {critical_tolerance}"
            ),
        )

    await log_event(
        session,
        event_type=event_type,
        event_category="risk",
        summary=summary,
        details={
            "mode": result.mode,
            "db_equity": str(result.db_equity),
            "exchange_equity": str(result.exchange_equity) if result.exchange_equity is not None else None,
            "difference": str(result.difference),
            "within_warning_tolerance": result.within_warning_tolerance,
            "within_critical_tolerance": result.within_critical_tolerance,
            "warning_tolerance": str(warning_tolerance),
            "critical_tolerance": str(critical_tolerance),
        },
        actor=actor,
    )

    if emergency_stop_triggered:
        try:
            from packages.core.ai import get_emergency_stop_analyzer

            await get_emergency_stop_analyzer().analyze_and_enqueue(
                session,
                reason="balance_reconciliation_critical_mismatch",
                source="reconciliation_guard",
                metadata={
                    "mode": result.mode,
                    "difference": str(result.difference),
                    "warning_tolerance": str(warning_tolerance),
                    "critical_tolerance": str(critical_tolerance),
                },
                actor=actor,
            )
        except Exception:  # noqa: BLE001
            # Do not break reconciliation guard if AI analysis fails.
            pass

    return ReconciliationGuardResult(
        result=result,
        event_type=event_type,
        summary=summary,
        emergency_stop_triggered=emergency_stop_triggered,
    )
