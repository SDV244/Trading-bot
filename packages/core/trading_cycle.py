"""Single-cycle paper trading orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.audit import log_event
from packages.core.config import get_settings
from packages.core.database.models import (
    Candle,
    EquitySnapshot,
    Fill,
    MetricsSnapshot,
    Order,
    Position,
)
from packages.core.database.repositories import CandleRepository
from packages.core.execution.paper_engine import OrderRequest, PaperEngine, PaperFill
from packages.core.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskInput
from packages.core.state import get_state_manager
from packages.core.strategies.base import CandleInput, StrategyContext
from packages.core.strategies.trend import EMATrendStrategy


@dataclass(slots=True, frozen=True)
class CycleResult:
    """Outcome of a trading cycle."""

    symbol: str
    signal_side: str
    signal_reason: str
    risk_action: str
    risk_reason: str
    executed: bool
    order_id: int | None
    fill_id: str | None
    quantity: str
    price: str | None


class TradingCycleService:
    """Run one full paper-trading cycle (signal -> risk -> paper fill -> persist)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.symbol = settings.trading.pair
        self.strategy = EMATrendStrategy()
        self.risk_engine = RiskEngine(
            RiskConfig(
                risk_per_trade=Decimal(str(settings.risk.per_trade)),
                max_daily_loss=Decimal(str(settings.risk.max_daily_loss)),
                max_exposure=Decimal(str(settings.risk.max_exposure)),
                fee_bps=settings.risk.fee_bps,
                slippage_bps=settings.risk.slippage_bps,
            )
        )
        self.paper_engine = PaperEngine(
            fee_bps=settings.risk.fee_bps,
            slippage_bps=settings.risk.slippage_bps,
        )
        self.starting_equity = Decimal("10000")

    async def run_once(self, session: AsyncSession) -> CycleResult:
        state = get_state_manager()
        if not state.can_trade:
            await log_event(
                session,
                event_type="trading_cycle_skipped",
                event_category="trade",
                summary="Trading cycle skipped because system is not RUNNING",
                details={"state": state.current.state.value, "reason": state.current.reason},
            )
            return CycleResult(
                symbol=self.symbol,
                signal_side="HOLD",
                signal_reason="system_not_running",
                risk_action="HOLD",
                risk_reason="system_not_running",
                executed=False,
                order_id=None,
                fill_id=None,
                quantity="0",
                price=None,
            )

        context = await self._load_strategy_context(session)
        if context is None:
            return CycleResult(
                symbol=self.symbol,
                signal_side="HOLD",
                signal_reason="insufficient_candles",
                risk_action="HOLD",
                risk_reason="insufficient_candles",
                executed=False,
                order_id=None,
                fill_id=None,
                quantity="0",
                price=None,
            )

        signal = self.strategy.generate_signal(context)
        last_price = context.candles_1h[-1].close
        position = await self._get_or_create_position(session)
        equity = await self._get_current_equity(session)
        current_exposure = position.quantity * last_price
        daily_pnl = Decimal("0")

        risk_decision = self.risk_engine.evaluate(
            signal=signal,
            risk_input=RiskInput(
                equity=equity,
                daily_realized_pnl=daily_pnl,
                current_exposure_notional=current_exposure,
                price=last_price,
            ),
        )

        if signal.side == "SELL" and position.quantity <= 0:
            risk_decision = RiskDecision(
                action="HOLD",
                allowed=False,
                quantity=Decimal("0"),
                notional=Decimal("0"),
                estimated_fee=Decimal("0"),
                estimated_slippage_cost=Decimal("0"),
                reason="no_inventory_to_sell",
            )

        if not risk_decision.allowed:
            await log_event(
                session,
                event_type="risk_hold",
                event_category="risk",
                summary=f"Signal blocked by risk engine: {risk_decision.reason}",
                details={
                    "signal": signal.side,
                    "signal_reason": signal.reason,
                    "risk_action": risk_decision.action,
                    "risk_reason": risk_decision.reason,
                },
            )
            return CycleResult(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_action=risk_decision.action,
                risk_reason=risk_decision.reason,
                executed=False,
                order_id=None,
                fill_id=None,
                quantity="0",
                price=None,
            )

        if signal.side not in {"BUY", "SELL"}:
            return CycleResult(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_action="HOLD",
                risk_reason="unsupported_signal_side",
                executed=False,
                order_id=None,
                fill_id=None,
                quantity="0",
                price=None,
            )
        trade_side = cast(Literal["BUY", "SELL"], signal.side)

        quantity = risk_decision.quantity
        if trade_side == "SELL":
            quantity = min(quantity, position.quantity)
            if quantity <= 0:
                return CycleResult(
                    symbol=self.symbol,
                    signal_side=trade_side,
                    signal_reason=signal.reason,
                    risk_action="HOLD",
                    risk_reason="no_inventory_to_sell",
                    executed=False,
                    order_id=None,
                    fill_id=None,
                    quantity="0",
                    price=None,
                )

        fill = self.paper_engine.execute_market_order(
            OrderRequest(symbol=self.symbol, side=trade_side, quantity=quantity, order_type="MARKET"),
            last_price=last_price,
        )

        order = Order(
            client_order_id=f"paper_{uuid4().hex[:20]}",
            exchange_order_id=None,
            symbol=self.symbol,
            side=trade_side,
            order_type="MARKET",
            quantity=fill.filled_quantity,
            price=fill.price,
            status="FILLED",
            is_paper=True,
            strategy_name=self.strategy.name,
            signal_reason=signal.reason,
            config_version=1,
        )
        session.add(order)
        await session.flush()

        fill_model = Fill(
            order_id=order.id,
            fill_id=fill.fill_id,
            quantity=fill.filled_quantity,
            price=fill.price or Decimal("0"),
            fee=fill.fee,
            fee_asset="USDT",
            is_paper=True,
            slippage_bps=float(fill.slippage_bps),
        )
        session.add(fill_model)

        realized_change = self._apply_fill_to_position(position, fill)
        session.add(position)

        equity_value = self.starting_equity + position.realized_pnl + position.unrealized_pnl
        exposure = position.quantity * (fill.price or last_price)
        available_balance = equity_value - exposure
        session.add(
            EquitySnapshot(
                equity=equity_value,
                available_balance=available_balance,
                unrealized_pnl=position.unrealized_pnl,
                is_paper=True,
            )
        )

        metrics = await self._build_metrics_snapshot(session, realized_change)
        session.add(metrics)

        await log_event(
            session,
            event_type="paper_trade_executed",
            event_category="trade",
            summary=f"{trade_side} {fill.filled_quantity} {self.symbol} at {fill.price}",
            details={
                "order_id": order.id,
                "fill_id": fill.fill_id,
                "signal_reason": signal.reason,
                "risk_reason": risk_decision.reason,
                "fee": str(fill.fee),
            },
        )

        return CycleResult(
            symbol=self.symbol,
            signal_side=trade_side,
            signal_reason=signal.reason,
            risk_action=risk_decision.action,
            risk_reason=risk_decision.reason,
            executed=True,
            order_id=order.id,
            fill_id=fill.fill_id,
            quantity=str(fill.filled_quantity),
            price=str(fill.price) if fill.price else None,
        )

    async def _load_strategy_context(self, session: AsyncSession) -> StrategyContext | None:
        repo = CandleRepository(session)
        candles_1h = await repo.get_latest_candles(self.symbol, "1h", limit=120)
        candles_4h = await repo.get_latest_candles(self.symbol, "4h", limit=120)
        if not candles_1h or not candles_4h:
            return None

        def to_input(candle: Candle) -> CandleInput:
            return CandleInput(
                open_time=candle.open_time,
                close=Decimal(str(candle.close)),
                high=Decimal(str(candle.high)),
                low=Decimal(str(candle.low)),
                volume=Decimal(str(candle.volume)),
            )

        return StrategyContext(
            symbol=self.symbol,
            candles_1h=[to_input(c) for c in reversed(candles_1h)],
            candles_4h=[to_input(c) for c in reversed(candles_4h)],
            now=datetime.now(UTC),
        )

    async def _get_or_create_position(self, session: AsyncSession) -> Position:
        result = await session.execute(
            select(Position).where(
                Position.symbol == self.symbol,
                Position.is_paper.is_(True),
            )
        )
        position = result.scalar_one_or_none()
        if position is not None:
            return position

        position = Position(
            symbol=self.symbol,
            side=None,
            quantity=Decimal("0"),
            avg_entry_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_fees=Decimal("0"),
            is_paper=True,
        )
        session.add(position)
        await session.flush()
        return position

    async def _get_current_equity(self, session: AsyncSession) -> Decimal:
        result = await session.execute(
            select(EquitySnapshot).where(EquitySnapshot.is_paper.is_(True)).order_by(EquitySnapshot.id.desc())
        )
        snapshot = result.scalars().first()
        if snapshot is None:
            return self.starting_equity
        return Decimal(str(snapshot.equity))

    def _apply_fill_to_position(self, position: Position, fill: PaperFill) -> Decimal:
        fill_side = fill.side
        fill_qty = Decimal(str(fill.filled_quantity))
        fill_price = Decimal(str(fill.price or Decimal("0")))
        fee = Decimal(str(fill.fee))

        realized_delta = Decimal("0")
        if fill_side == "BUY":
            prev_qty = Decimal(str(position.quantity))
            prev_avg = Decimal(str(position.avg_entry_price))
            new_qty = prev_qty + fill_qty
            if new_qty > 0:
                position.avg_entry_price = ((prev_qty * prev_avg) + (fill_qty * fill_price)) / new_qty
            position.quantity = new_qty
            position.side = "LONG"
        else:
            prev_qty = Decimal(str(position.quantity))
            prev_avg = Decimal(str(position.avg_entry_price))
            sell_qty = min(fill_qty, prev_qty)
            realized_delta = (fill_price - prev_avg) * sell_qty - fee
            position.quantity = prev_qty - sell_qty
            if position.quantity <= 0:
                position.quantity = Decimal("0")
                position.avg_entry_price = Decimal("0")
                position.side = None

        position.total_fees = Decimal(str(position.total_fees)) + fee
        if position.quantity > 0 and position.avg_entry_price > 0:
            position.unrealized_pnl = (
                (fill_price - Decimal(str(position.avg_entry_price))) * Decimal(str(position.quantity))
            )
        else:
            position.unrealized_pnl = Decimal("0")
        position.realized_pnl = Decimal(str(position.realized_pnl)) + realized_delta
        return realized_delta

    async def _build_metrics_snapshot(
        self,
        session: AsyncSession,
        realized_delta: Decimal,
    ) -> MetricsSnapshot:
        total_trades = (
            await session.execute(select(func.count(Fill.id)).where(Fill.is_paper.is_(True)))
        ).scalar_one()
        total_fees = (
            await session.execute(select(func.coalesce(func.sum(Fill.fee), 0)).where(Fill.is_paper.is_(True)))
        ).scalar_one()
        position_result = await session.execute(
            select(Position).where(Position.symbol == self.symbol, Position.is_paper.is_(True))
        )
        position = position_result.scalar_one_or_none()
        total_pnl = Decimal("0")
        if position is not None:
            total_pnl = Decimal(str(position.realized_pnl)) + Decimal(str(position.unrealized_pnl))

        winning_trades = 1 if realized_delta > 0 else 0
        losing_trades = 1 if realized_delta < 0 else 0
        win_rate = None
        if total_trades > 0:
            win_rate = winning_trades / total_trades

        return MetricsSnapshot(
            strategy_name=self.strategy.name,
            total_trades=int(total_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_fees=Decimal(str(total_fees)),
            max_drawdown=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            profit_factor=None,
            win_rate=win_rate,
            avg_trade_pnl=(total_pnl / total_trades) if total_trades > 0 else None,
            is_paper=True,
        )

_trading_cycle_service: TradingCycleService | None = None


def get_trading_cycle_service() -> TradingCycleService:
    """Get or create singleton trading cycle service."""
    global _trading_cycle_service
    if _trading_cycle_service is None:
        _trading_cycle_service = TradingCycleService()
    return _trading_cycle_service
