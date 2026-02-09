"""Single-cycle paper trading orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.telegram_bot import get_telegram_notifier
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
from packages.core.risk.engine import RiskConfig, RiskEngine, RiskInput
from packages.core.state import get_state_manager
from packages.core.strategies import Strategy, registry
from packages.core.strategies.base import CandleInput, Signal, StrategyContext


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


@dataclass(slots=True, frozen=True)
class TimeframeReadiness:
    """Required/available candle status for one timeframe."""

    required: int
    available: int
    ready: bool


@dataclass(slots=True, frozen=True)
class DataReadiness:
    """Strategy data readiness snapshot."""

    symbol: str
    active_strategy: str
    require_data_ready: bool
    data_ready: bool
    reasons: list[str]
    timeframes: dict[str, TimeframeReadiness]


class TradingCycleService:
    """Run one full paper-trading cycle (signal -> risk -> paper fill -> persist)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.symbol = settings.trading.pair
        self.spot_position_mode = settings.trading.spot_position_mode.strip().lower()
        self.timing_timeframe, self.regime_timeframe = self._resolve_strategy_timeframes(
            settings.trading.timeframe_list
        )
        self.strategy = self._build_strategy(settings.trading.active_strategy)
        self.strategy_requirements = self._normalize_requirements(self.strategy.data_requirements())
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
            allowed_symbols={self.symbol},
        )
        self.starting_equity = Decimal(str(settings.trading.paper_starting_equity))
        self.grid_cooldown_seconds = max(0, settings.trading.grid_cooldown_seconds)
        self._last_out_of_bounds_alert_at: datetime | None = None

    def _build_strategy(self, strategy_name: str) -> Strategy:
        try:
            if strategy_name == "smart_grid_ai":
                return registry.create(
                    strategy_name,
                    lookback_1h=self.settings.trading.grid_lookback_1h,
                    atr_period_1h=self.settings.trading.grid_atr_period_1h,
                    grid_levels=self.settings.trading.grid_levels,
                    spacing_mode=self.settings.trading.grid_spacing_mode,
                    min_spacing_bps=self.settings.trading.grid_min_spacing_bps,
                    max_spacing_bps=self.settings.trading.grid_max_spacing_bps,
                    trend_tilt=self.settings.trading.grid_trend_tilt,
                    volatility_blend=self.settings.trading.grid_volatility_blend,
                    take_profit_buffer=self.settings.trading.grid_take_profit_buffer,
                    stop_loss_buffer=self.settings.trading.grid_stop_loss_buffer,
                )
            return registry.create(strategy_name)
        except ValueError as exc:
            msg = f"Invalid TRADING_ACTIVE_STRATEGY '{strategy_name}': {exc}"
            raise ValueError(msg) from exc

    def _normalize_requirements(self, raw: dict[str, int]) -> dict[str, int]:
        requirements: dict[str, int] = {}
        logical_map = {
            "1h": self.timing_timeframe,
            "4h": self.regime_timeframe,
        }
        for timeframe, value in raw.items():
            normalized = max(0, int(value))
            mapped_timeframe = logical_map.get(timeframe, timeframe)
            current = requirements.get(mapped_timeframe, 0)
            requirements[mapped_timeframe] = max(current, normalized)

        requirements[self.timing_timeframe] = max(1, requirements.get(self.timing_timeframe, 0))

        for timeframe in self.settings.trading.timeframe_list:
            requirements.setdefault(timeframe, 0)
        return requirements

    def _resolve_strategy_timeframes(self, configured: list[str]) -> tuple[str, str]:
        if not configured:
            return "1h", "4h"
        if len(configured) == 1:
            return configured[0], configured[0]
        return configured[0], configured[1]

    async def get_data_readiness(self, session: AsyncSession) -> DataReadiness:
        repo = CandleRepository(session)
        readiness: dict[str, TimeframeReadiness] = {}
        reasons: list[str] = []

        for timeframe, required in self.strategy_requirements.items():
            available = await repo.count_candles(self.symbol, timeframe)
            is_ready = required <= 0 or available >= required
            readiness[timeframe] = TimeframeReadiness(
                required=required,
                available=available,
                ready=is_ready,
            )
            if required > 0 and not is_ready:
                reasons.append(f"{timeframe}: requires {required} candles, found {available}")

        data_ready = all(item.ready for item in readiness.values())
        if self.strategy.name == "smart_grid_ai" and self.settings.trading.grid_enforce_fee_floor:
            round_trip_cost_bps = (self.settings.risk.fee_bps * 2) + (self.settings.risk.slippage_bps * 2)
            min_net_bps = self.settings.trading.grid_min_spacing_bps - round_trip_cost_bps
            required_net_bps = self.settings.trading.grid_min_net_profit_bps
            if min_net_bps < required_net_bps:
                data_ready = False
                reasons.append(
                    (
                        "fee_floor_not_met: "
                        f"min_net_bps={min_net_bps}, required_net_bps={required_net_bps}, "
                        f"grid_min_spacing_bps={self.settings.trading.grid_min_spacing_bps}, "
                        f"round_trip_cost_bps={round_trip_cost_bps}"
                    )
                )
        return DataReadiness(
            symbol=self.symbol,
            active_strategy=self.strategy.name,
            require_data_ready=self.settings.trading.require_data_ready,
            data_ready=data_ready,
            reasons=reasons,
            timeframes=readiness,
        )

    def get_strategy_requirements(self) -> dict[str, int]:
        """Return normalized candle requirements for the active strategy."""
        return dict(self.strategy_requirements)

    @staticmethod
    def _build_hold_result(
        symbol: str,
        signal_side: str,
        signal_reason: str,
        risk_reason: str,
    ) -> CycleResult:
        return CycleResult(
            symbol=symbol,
            signal_side=signal_side,
            signal_reason=signal_reason,
            risk_action="HOLD",
            risk_reason=risk_reason,
            executed=False,
            order_id=None,
            fill_id=None,
            quantity="0",
            price=None,
        )

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
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side="HOLD",
                signal_reason="insufficient_candles",
                risk_reason="insufficient_candles",
            )

        signal = self.strategy.generate_signal(context)
        last_price = context.candles_1h[-1].close
        position = await self._get_or_create_position(session)
        await self._maybe_alert_out_of_bounds(session, signal, last_price)
        if self.grid_cooldown_seconds > 0 and signal.side in {"BUY", "SELL"}:
            remaining_cooldown = await self._cooldown_remaining_seconds(session)
            if remaining_cooldown > 0:
                await self._persist_cycle_snapshot(
                    session=session,
                    position=position,
                    mark_price=last_price,
                    realized_delta=Decimal("0"),
                )
                return self._build_hold_result(
                    symbol=self.symbol,
                    signal_side=signal.side,
                    signal_reason=signal.reason,
                    risk_reason=f"cooldown_active_{remaining_cooldown}s",
                )

        if signal.side == "SELL" and position.quantity <= 0:
            bootstrap = await self._maybe_bootstrap_inventory(
                session=session,
                position=position,
                mark_price=last_price,
            )
            if bootstrap is not None:
                return bootstrap
            await self._persist_cycle_snapshot(
                session=session,
                position=position,
                mark_price=last_price,
                realized_delta=Decimal("0"),
            )
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_reason="no_inventory_to_sell",
            )

        if self.spot_position_mode == "long_flat" and signal.side == "BUY" and position.quantity > 0:
            await self._persist_cycle_snapshot(
                session=session,
                position=position,
                mark_price=last_price,
                realized_delta=Decimal("0"),
            )
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_reason="already_in_position",
            )

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
            await self._persist_cycle_snapshot(
                session=session,
                position=position,
                mark_price=last_price,
                realized_delta=Decimal("0"),
            )
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_reason=risk_decision.reason,
            )

        if signal.side not in {"BUY", "SELL"}:
            await self._persist_cycle_snapshot(
                session=session,
                position=position,
                mark_price=last_price,
                realized_delta=Decimal("0"),
            )
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
            if self.spot_position_mode == "long_flat":
                quantity = Decimal(str(position.quantity))
            else:
                quantity = min(quantity, position.quantity)
            if quantity <= 0:
                await self._persist_cycle_snapshot(
                    session=session,
                    position=position,
                    mark_price=last_price,
                    realized_delta=Decimal("0"),
                )
                return self._build_hold_result(
                    symbol=self.symbol,
                    signal_side=trade_side,
                    signal_reason=signal.reason,
                    risk_reason="no_inventory_to_sell",
                )

        return await self._execute_paper_trade(
            session=session,
            position=position,
            trade_side=trade_side,
            quantity=quantity,
            mark_price=last_price,
            signal_reason=signal.reason,
            risk_action=risk_decision.action,
            risk_reason=risk_decision.reason,
        )

    async def _maybe_alert_out_of_bounds(
        self,
        session: AsyncSession,
        signal: Signal,
        mark_price: Decimal,
    ) -> None:
        if self.strategy.name != "smart_grid_ai":
            return
        if signal.reason != "grid_recenter_wait":
            return
        now = datetime.now(UTC)
        cooldown_minutes = self.settings.trading.grid_out_of_bounds_alert_cooldown_minutes
        if self._last_out_of_bounds_alert_at is not None:
            elapsed_minutes = (now - self._last_out_of_bounds_alert_at).total_seconds() / 60
            if elapsed_minutes < cooldown_minutes:
                return

        lower = signal.indicators.get("grid_lower")
        upper = signal.indicators.get("grid_upper")
        notifier = get_telegram_notifier()
        if notifier.enabled:
            await notifier.send_critical_alert(
                "Grid out-of-bounds",
                (
                    f"Symbol: {self.symbol}\n"
                    f"Price: {mark_price}\n"
                    f"Grid lower: {lower}\n"
                    f"Grid upper: {upper}\n"
                    f"Action: waiting for recenter"
                ),
            )
        await log_event(
            session,
            event_type="grid_out_of_bounds_alert",
            event_category="risk",
            summary="Smart-grid price moved outside active band",
            details={
                "symbol": self.symbol,
                "price": str(mark_price),
                "grid_lower": lower,
                "grid_upper": upper,
                "signal_reason": signal.reason,
            },
        )
        self._last_out_of_bounds_alert_at = now

    async def _execute_paper_trade(
        self,
        session: AsyncSession,
        position: Position,
        trade_side: Literal["BUY", "SELL"],
        quantity: Decimal,
        mark_price: Decimal,
        signal_reason: str,
        risk_action: str,
        risk_reason: str,
    ) -> CycleResult:
        fill = self.paper_engine.execute_market_order(
            OrderRequest(symbol=self.symbol, side=trade_side, quantity=quantity, order_type="MARKET"),
            last_price=mark_price,
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
            signal_reason=signal_reason,
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
        await self._persist_cycle_snapshot(
            session=session,
            position=position,
            mark_price=(fill.price or mark_price),
            realized_delta=realized_change,
        )
        await log_event(
            session,
            event_type="paper_trade_executed",
            event_category="trade",
            summary=f"{trade_side} {fill.filled_quantity} {self.symbol} at {fill.price}",
            details={
                "order_id": order.id,
                "fill_id": fill.fill_id,
                "signal_reason": signal_reason,
                "risk_reason": risk_reason,
                "fee": str(fill.fee),
            },
        )
        return CycleResult(
            symbol=self.symbol,
            signal_side=trade_side,
            signal_reason=signal_reason,
            risk_action=risk_action,
            risk_reason=risk_reason,
            executed=True,
            order_id=order.id,
            fill_id=fill.fill_id,
            quantity=str(fill.filled_quantity),
            price=str(fill.price) if fill.price else None,
        )

    async def _maybe_bootstrap_inventory(
        self,
        session: AsyncSession,
        position: Position,
        mark_price: Decimal,
    ) -> CycleResult | None:
        if self.strategy.name != "smart_grid_ai":
            return None
        if not self.settings.trading.grid_auto_inventory_bootstrap:
            return None
        equity = await self._get_current_equity(session)
        current_exposure = position.quantity * mark_price
        decision = self.risk_engine.evaluate(
            signal=Signal(
                side="BUY",
                confidence=0.5,
                reason="grid_inventory_bootstrap",
                indicators={},
            ),
            risk_input=RiskInput(
                equity=equity,
                daily_realized_pnl=Decimal("0"),
                current_exposure_notional=current_exposure,
                price=mark_price,
            ),
        )
        if not decision.allowed:
            return None
        fraction = Decimal(str(self.settings.trading.grid_bootstrap_fraction))
        quantity = (decision.quantity * fraction).quantize(
            self.risk_engine.config.lot_step_size,
            rounding=ROUND_DOWN,
        )
        if quantity <= 0:
            return None
        await log_event(
            session,
            event_type="grid_inventory_bootstrap",
            event_category="trade",
            summary=f"Bootstrapping smart-grid inventory with BUY {quantity} {self.symbol}",
            details={
                "reason": "initial_inventory_for_sell_grid_levels",
                "risk_reason": decision.reason,
                "fraction": str(fraction),
            },
        )
        return await self._execute_paper_trade(
            session=session,
            position=position,
            trade_side="BUY",
            quantity=quantity,
            mark_price=mark_price,
            signal_reason="grid_inventory_bootstrap",
            risk_action="ALLOW",
            risk_reason="grid_inventory_bootstrap",
        )

    async def _load_strategy_context(self, session: AsyncSession) -> StrategyContext | None:
        repo = CandleRepository(session)
        required_timing = self.strategy_requirements.get(self.timing_timeframe, 0)
        required_regime = self.strategy_requirements.get(self.regime_timeframe, 0)

        candles_1h: list[Candle] = []
        candles_4h: list[Candle] = []

        if self.timing_timeframe == self.regime_timeframe:
            required_common = max(required_timing, required_regime)
            candles_common = await repo.get_latest_candles(
                self.symbol,
                self.timing_timeframe,
                limit=required_common,
            )
            if len(candles_common) < required_common:
                return None

            candles_1h = candles_common[:required_timing] if required_timing > 0 else []
            candles_4h = candles_common[:required_regime] if required_regime > 0 else []
        else:
            if required_timing > 0:
                candles_1h = await repo.get_latest_candles(
                    self.symbol,
                    self.timing_timeframe,
                    limit=required_timing,
                )
                if len(candles_1h) < required_timing:
                    return None

            if required_regime > 0:
                candles_4h = await repo.get_latest_candles(
                    self.symbol,
                    self.regime_timeframe,
                    limit=required_regime,
                )
                if len(candles_4h) < required_regime:
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

    async def _cooldown_remaining_seconds(self, session: AsyncSession) -> int:
        if self.grid_cooldown_seconds <= 0:
            return 0
        result = await session.execute(
            select(Fill.filled_at)
            .join(Order, Fill.order_id == Order.id)
            .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            .order_by(Fill.filled_at.desc())
            .limit(1)
        )
        last_filled_at = result.scalar_one_or_none()
        if last_filled_at is None:
            return 0
        if last_filled_at.tzinfo is None:
            last_filled_at = last_filled_at.replace(tzinfo=UTC)
        elapsed_seconds = (datetime.now(UTC) - last_filled_at).total_seconds()
        remaining = self.grid_cooldown_seconds - int(elapsed_seconds)
        return max(0, remaining)

    async def _persist_cycle_snapshot(
        self,
        session: AsyncSession,
        position: Position,
        mark_price: Decimal,
        realized_delta: Decimal,
    ) -> None:
        """Persist equity and metrics snapshot for every analyzed cycle."""
        equity_value = self.starting_equity + position.realized_pnl + position.unrealized_pnl
        exposure = position.quantity * mark_price
        available_balance = equity_value - exposure
        session.add(
            EquitySnapshot(
                equity=equity_value,
                available_balance=available_balance,
                unrealized_pnl=position.unrealized_pnl,
                is_paper=True,
            )
        )
        metrics = await self._build_metrics_snapshot(session, realized_delta)
        session.add(metrics)

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

        previous_metrics_result = await session.execute(
            select(MetricsSnapshot)
            .where(MetricsSnapshot.is_paper.is_(True), MetricsSnapshot.strategy_name == self.strategy.name)
            .order_by(MetricsSnapshot.snapshot_at.desc())
            .limit(1)
        )
        previous_metrics = previous_metrics_result.scalar_one_or_none()
        previous_wins = previous_metrics.winning_trades if previous_metrics else 0
        previous_losses = previous_metrics.losing_trades if previous_metrics else 0
        winning_trades = previous_wins + (1 if realized_delta > 0 else 0)
        losing_trades = previous_losses + (1 if realized_delta < 0 else 0)
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
