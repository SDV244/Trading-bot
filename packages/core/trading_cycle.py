"""Single-cycle paper trading orchestration."""

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.telegram_bot import get_telegram_notifier
from packages.adapters.webhook_notifier import get_webhook_notifier
from packages.core.audit import log_event, resolve_active_config_version
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
from packages.core.execution.paper_engine import (
    OrderRequest,
    PaperEngine,
    PaperExecutionError,
    PaperFill,
)
from packages.core.metrics.calculator import MetricsCalculator
from packages.core.risk.engine import RiskConfig, RiskEngine, RiskInput
from packages.core.risk.global_stop_loss import GlobalStopLossConfig, GlobalStopLossGuard
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


@dataclass(slots=True, frozen=True)
class GridLevel:
    """One projected smart-grid level."""

    level: int
    price: Decimal
    distance_bps: Decimal


@dataclass(slots=True, frozen=True)
class GridPreview:
    """Snapshot of smart-grid projected levels and triggers."""

    symbol: str
    strategy: str
    last_price: Decimal
    signal_side: str
    signal_reason: str
    confidence: float
    grid_center: Decimal | None
    grid_upper: Decimal | None
    grid_lower: Decimal | None
    buy_trigger: Decimal | None
    sell_trigger: Decimal | None
    spacing_bps: float | None
    grid_step: Decimal | None
    recentered: bool
    recenter_mode: str
    buy_levels: list[GridLevel]
    sell_levels: list[GridLevel]
    take_profit_trigger: Decimal | None
    stop_loss_trigger: Decimal | None
    position_quantity: Decimal
    bootstrap_eligible: bool


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
        self.stop_loss_guard = GlobalStopLossGuard(
            GlobalStopLossConfig(
                enabled=settings.trading.stop_loss_enabled,
                global_equity_pct=Decimal(str(settings.trading.stop_loss_global_equity_pct)),
                max_drawdown_pct=Decimal(str(settings.trading.stop_loss_max_drawdown_pct)),
                auto_close_positions=settings.trading.stop_loss_auto_close_positions,
            )
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
                    recenter_mode=self.settings.trading.grid_recenter_mode,
                )
            return registry.create(strategy_name)
        except ValueError as exc:
            msg = f"Invalid TRADING_ACTIVE_STRATEGY '{strategy_name}': {exc}"
            raise ValueError(msg) from exc

    def set_grid_recenter_mode(self, mode: str) -> str:
        """Update smart-grid recenter mode at runtime for the active service."""
        normalized = mode.strip().lower()
        if normalized not in {"conservative", "aggressive"}:
            raise ValueError("mode must be 'conservative' or 'aggressive'")
        self.settings.trading.grid_recenter_mode = normalized
        if self.strategy.name == "smart_grid_ai":
            self.strategy.recenter_mode = normalized  # type: ignore[attr-defined]
        return normalized

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

                        "fee_floor_not_met: "
                        f"min_net_bps={min_net_bps}, required_net_bps={required_net_bps}, "
                        f"grid_min_spacing_bps={self.settings.trading.grid_min_spacing_bps}, "
                        f"round_trip_cost_bps={round_trip_cost_bps}"

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

    async def get_grid_preview(self, session: AsyncSession) -> GridPreview | None:
        """Build smart-grid level preview for operator visibility."""
        if self.strategy.name != "smart_grid_ai":
            return None

        context = await self._load_strategy_context(session)
        if context is None or len(context.candles_1h) == 0:
            return None

        signal = self.strategy.generate_signal(context)
        indicators = signal.indicators
        last_price = context.candles_1h[-1].close

        position_result = await session.execute(
            select(Position).where(
                Position.symbol == self.symbol,
                Position.is_paper.is_(True),
            )
        )
        position = position_result.scalar_one_or_none()
        quantity = Decimal("0") if position is None else Decimal(str(position.quantity))

        bootstrap_eligible = await self._is_bootstrap_candidate(
            session=session,
            signal_side=signal.side,
            signal_reason=signal.reason,
            current_quantity=quantity,
        )

        return GridPreview(
            symbol=self.symbol,
            strategy=self.strategy.name,
            last_price=last_price,
            signal_side=signal.side,
            signal_reason=signal.reason,
            confidence=signal.confidence,
            grid_center=self._decimal_indicator(indicators, "grid_center"),
            grid_upper=self._decimal_indicator(indicators, "grid_upper"),
            grid_lower=self._decimal_indicator(indicators, "grid_lower"),
            buy_trigger=self._decimal_indicator(indicators, "buy_trigger"),
            sell_trigger=self._decimal_indicator(indicators, "sell_trigger"),
            spacing_bps=self._float_indicator(indicators, "spacing_bps"),
            grid_step=self._decimal_indicator(indicators, "grid_step"),
            recentered=(self._float_indicator(indicators, "recentered") or 0.0) > 0.0,
            recenter_mode=self.settings.trading.grid_recenter_mode,
            buy_levels=self._extract_grid_levels(indicators, "grid_buy_level_", last_price),
            sell_levels=self._extract_grid_levels(indicators, "grid_sell_level_", last_price),
            take_profit_trigger=self._decimal_indicator(indicators, "take_profit_trigger"),
            stop_loss_trigger=self._decimal_indicator(indicators, "stop_loss_trigger"),
            position_quantity=quantity,
            bootstrap_eligible=bootstrap_eligible,
        )

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

    @staticmethod
    def _float_indicator(indicators: dict[str, float], key: str) -> float | None:
        raw = indicators.get(key)
        if isinstance(raw, int | float):
            return float(raw)
        return None

    @classmethod
    def _decimal_indicator(cls, indicators: dict[str, float], key: str) -> Decimal | None:
        raw = cls._float_indicator(indicators, key)
        if raw is None:
            return None
        return Decimal(str(raw))

    @staticmethod
    def _extract_grid_levels(
        indicators: dict[str, float],
        prefix: str,
        reference_price: Decimal,
    ) -> list[GridLevel]:
        levels: list[GridLevel] = []
        if reference_price <= 0:
            return levels
        for key, raw_value in indicators.items():
            if not key.startswith(prefix):
                continue
            level_token = key.removeprefix(prefix)
            if not level_token.isdigit():
                continue
            if not isinstance(raw_value, int | float):
                continue
            level = int(level_token)
            level_price = Decimal(str(raw_value))
            distance_bps = ((level_price - reference_price) / reference_price) * Decimal("10000")
            levels.append(
                GridLevel(
                    level=level,
                    price=level_price,
                    distance_bps=distance_bps,
                )
            )
        levels.sort(key=lambda item: item.level)
        return levels

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

        last_price = context.candles_1h[-1].close
        position = await self._get_or_create_position(session)
        stop_loss_result = await self._check_global_stop_loss(
            session=session,
            position=position,
            mark_price=last_price,
        )
        if stop_loss_result is not None:
            return stop_loss_result

        signal = self.strategy.generate_signal(context)
        await self._maybe_alert_out_of_bounds(session, signal, last_price)
        if position.quantity <= 0:
            bootstrap = await self._maybe_bootstrap_inventory(
                session=session,
                position=position,
                mark_price=last_price,
                signal_side=signal.side,
                signal_reason=signal.reason,
            )
            if bootstrap is not None:
                return bootstrap
        if self.grid_cooldown_seconds > 0 and signal.side in {"BUY", "SELL"}:
            remaining_cooldown = await self._cooldown_remaining_seconds(session)
            if remaining_cooldown > 0:
                await self._persist_cycle_snapshot(
                    session=session,
                    position=position,
                    mark_price=last_price,
                )
                return self._build_hold_result(
                    symbol=self.symbol,
                    signal_side=signal.side,
                    signal_reason=signal.reason,
                    risk_reason=f"cooldown_active_{remaining_cooldown}s",
                )

        if signal.side == "SELL" and position.quantity <= 0:
            await self._persist_cycle_snapshot(
                session=session,
                position=position,
                mark_price=last_price,
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
            )
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side=signal.side,
                signal_reason=signal.reason,
                risk_reason="already_in_position",
            )

        equity = await self._get_current_equity(session)
        current_exposure = position.quantity * last_price
        daily_pnl = await self._compute_daily_realized_pnl(session)

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
            signal_indicators=signal.indicators,
            candle_low=context.candles_1h[-1].low,
            candle_high=context.candles_1h[-1].high,
        )

    async def close_open_paper_position(
        self,
        session: AsyncSession,
        *,
        reason: str = "manual_close_all_positions",
        actor: str = "system",
    ) -> CycleResult:
        """Force-close the current paper position, if any."""
        position = await self._get_or_create_position(session)
        quantity = Decimal(str(position.quantity))
        if quantity <= 0:
            await log_event(
                session,
                event_type="paper_close_position_noop",
                event_category="trade",
                summary="Close-all-positions requested but no paper inventory is open",
                details={"symbol": self.symbol, "reason": reason},
                actor=actor,
            )
            return self._build_hold_result(
                symbol=self.symbol,
                signal_side="SELL",
                signal_reason=reason,
                risk_reason="no_inventory_to_sell",
            )

        fallback_price = Decimal(str(position.avg_entry_price))
        mark_price = await self._resolve_mark_price(session, fallback_price=fallback_price)
        return await self._execute_paper_trade(
            session=session,
            position=position,
            trade_side="SELL",
            quantity=quantity,
            mark_price=mark_price,
            signal_reason=reason,
            risk_action="FORCE_CLOSE",
            risk_reason=reason,
        )

    async def _check_global_stop_loss(
        self,
        *,
        session: AsyncSession,
        position: Position,
        mark_price: Decimal,
    ) -> CycleResult | None:
        if not self.stop_loss_guard.config.enabled:
            return None
        current_equity = self._compute_equity_from_position(position=position, mark_price=mark_price)
        peak_equity = await self._get_peak_equity(session)
        decision = self.stop_loss_guard.evaluate(
            current_equity=current_equity,
            starting_equity=self.starting_equity,
            peak_equity=peak_equity,
        )
        if not decision.triggered:
            return None
        manager = get_state_manager()
        manager.force_emergency_stop(
            reason=decision.reason,
            changed_by="global_stop_loss",
            metadata={
                "trigger_type": decision.trigger_type.value if decision.trigger_type else None,
                "current_equity": str(decision.current_equity),
                "starting_equity": str(decision.starting_equity),
                "peak_equity": str(decision.peak_equity),
                "loss_pct": str(decision.loss_pct),
                "drawdown_pct": str(decision.drawdown_pct),
            },
        )
        await log_event(
            session,
            event_type="global_stop_loss_triggered",
            event_category="risk",
            summary=(
                f"Global stop-loss triggered ({decision.reason}) "
                f"equity={decision.current_equity} peak={decision.peak_equity}"
            ),
            details={
                "trigger_type": decision.trigger_type.value if decision.trigger_type else None,
                "reason": decision.reason,
                "current_equity": str(decision.current_equity),
                "starting_equity": str(decision.starting_equity),
                "peak_equity": str(decision.peak_equity),
                "loss_pct": float(decision.loss_pct),
                "drawdown_pct": float(decision.drawdown_pct),
            },
            actor="global_stop_loss",
        )
        notifier = get_telegram_notifier()
        await notifier.send_critical_alert(
            "Global stop-loss triggered",
            (
                f"Symbol: {self.symbol}\n"
                f"Reason: {decision.reason}\n"
                f"Current equity: {decision.current_equity}\n"
                f"Peak equity: {decision.peak_equity}\n"
                f"Loss: {decision.loss_pct * Decimal('100'):.2f}%\n"
                f"Drawdown: {decision.drawdown_pct * Decimal('100'):.2f}%"
            ),
        )
        await get_webhook_notifier().send_critical_alert(
            "Global stop-loss triggered",
            (
                f"Symbol: {self.symbol}\n"
                f"Reason: {decision.reason}\n"
                f"Current equity: {decision.current_equity}\n"
                f"Peak equity: {decision.peak_equity}"
            ),
        )

        if self.stop_loss_guard.config.auto_close_positions and position.quantity > 0:
            return await self._execute_paper_trade(
                session=session,
                position=position,
                trade_side="SELL",
                quantity=Decimal(str(position.quantity)),
                mark_price=mark_price,
                signal_reason="global_stop_loss_forced_close",
                risk_action="HOLD",
                risk_reason=decision.reason,
            )

        await self._persist_cycle_snapshot(
            session=session,
            position=position,
            mark_price=mark_price,
        )
        return self._build_hold_result(
            symbol=self.symbol,
            signal_side="HOLD",
            signal_reason="global_stop_loss_triggered",
            risk_reason=decision.reason,
        )

    async def _maybe_alert_out_of_bounds(
        self,
        session: AsyncSession,
        signal: Signal,
        mark_price: Decimal,
    ) -> None:
        if self.strategy.name != "smart_grid_ai":
            return
        if signal.reason != "grid_recenter_wait" and not signal.reason.startswith("grid_recentered_auto"):
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
                    f"Action: auto recenter engaged"
                ),
            )
        await get_webhook_notifier().send_critical_alert(
            "Grid out-of-bounds",
            (
                f"Symbol: {self.symbol}\n"
                f"Price: {mark_price}\n"
                f"Grid lower: {lower}\n"
                f"Grid upper: {upper}"
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
        signal_indicators: dict[str, float] | None = None,
        candle_low: Decimal | None = None,
        candle_high: Decimal | None = None,
    ) -> CycleResult:
        fill, order_type = self._execute_preferred_fill(
            trade_side=trade_side,
            quantity=quantity,
            mark_price=mark_price,
            signal_indicators=signal_indicators,
            candle_low=candle_low,
            candle_high=candle_high,
        )
        config_version = await resolve_active_config_version(session)
        order = Order(
            client_order_id=f"paper_{uuid4().hex[:20]}",
            exchange_order_id=None,
            symbol=self.symbol,
            side=trade_side,
            order_type=order_type,
            quantity=fill.filled_quantity,
            price=fill.price,
            status=fill.status,
            is_paper=True,
            strategy_name=self.strategy.name,
            signal_reason=signal_reason,
            config_version=config_version,
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
        self._apply_fill_to_position(position, fill)
        session.add(position)
        await self._persist_cycle_snapshot(
            session=session,
            position=position,
            mark_price=(fill.price or mark_price),
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
                "order_type": order_type,
            },
            config_version=config_version,
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

    def _execute_preferred_fill(
        self,
        *,
        trade_side: Literal["BUY", "SELL"],
        quantity: Decimal,
        mark_price: Decimal,
        signal_indicators: dict[str, float] | None,
        candle_low: Decimal | None,
        candle_high: Decimal | None,
    ) -> tuple[PaperFill, Literal["LIMIT", "MARKET"]]:
        """Prefer maker-like limit fills for smart-grid, fallback to market fill."""
        if (
            self.strategy.name == "smart_grid_ai"
            and signal_indicators is not None
            and candle_low is not None
            and candle_high is not None
        ):
            trigger_key = "buy_trigger" if trade_side == "BUY" else "sell_trigger"
            trigger_raw = signal_indicators.get(trigger_key)
            if isinstance(trigger_raw, int | float):
                limit_price = Decimal(str(trigger_raw))
                try:
                    limit_fill = self.paper_engine.execute_limit_order(
                        OrderRequest(
                            symbol=self.symbol,
                            side=trade_side,
                            quantity=quantity,
                            order_type="LIMIT",
                            limit_price=limit_price,
                        ),
                        candle_low=candle_low,
                        candle_high=candle_high,
                    )
                    if limit_fill.status in {"FILLED", "PARTIALLY_FILLED"} and limit_fill.filled_quantity > 0:
                        return limit_fill, "LIMIT"
                except PaperExecutionError:
                    # Keep cycle alive with market fallback when limit validation fails.
                    pass

        market_fill = self.paper_engine.execute_market_order(
            OrderRequest(symbol=self.symbol, side=trade_side, quantity=quantity, order_type="MARKET"),
            last_price=mark_price,
        )
        return market_fill, "MARKET"

    async def _maybe_bootstrap_inventory(
        self,
        session: AsyncSession,
        position: Position,
        mark_price: Decimal,
        signal_side: str,
        signal_reason: str,
    ) -> CycleResult | None:
        if not await self._is_bootstrap_candidate(
            session=session,
            signal_side=signal_side,
            signal_reason=signal_reason,
            current_quantity=Decimal(str(position.quantity)),
        ):
            return None
        equity = await self._get_current_equity(session)
        current_exposure = position.quantity * mark_price
        daily_realized = await self._compute_daily_realized_pnl(session)
        decision = self.risk_engine.evaluate(
            signal=Signal(
                side="BUY",
                confidence=0.5,
                reason="grid_inventory_bootstrap",
                indicators={},
            ),
            risk_input=RiskInput(
                equity=equity,
                daily_realized_pnl=daily_realized,
                current_exposure_notional=current_exposure,
                price=mark_price,
            ),
        )
        if not decision.allowed:
            return None
        fraction = Decimal(str(self.settings.trading.grid_bootstrap_fraction))
        lot_step = self.risk_engine.config.lot_step_size
        min_notional_qty = (self.risk_engine.config.min_notional / mark_price).quantize(
            lot_step,
            rounding=ROUND_UP,
        )
        quantity = max(
            (decision.quantity * fraction).quantize(
                lot_step,
                rounding=ROUND_DOWN,
            ),
            min_notional_qty,
        )
        if quantity <= 0:
            return None
        notional = quantity * mark_price
        if notional < self.risk_engine.config.min_notional:
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
                "trigger_signal_side": signal_side,
                "trigger_signal_reason": signal_reason,
                "notional": str(notional),
            },
        )
        try:
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
        except PaperExecutionError as exc:
            await log_event(
                session,
                event_type="grid_inventory_bootstrap_skipped",
                event_category="risk",
                summary=f"Bootstrap skipped: {exc}",
                details={
                    "symbol": self.symbol,
                    "quantity": str(quantity),
                    "mark_price": str(mark_price),
                    "min_notional": str(self.risk_engine.config.min_notional),
                },
            )
            return None

    async def _is_bootstrap_candidate(
        self,
        *,
        session: AsyncSession,
        signal_side: str,
        signal_reason: str,
        current_quantity: Decimal,
    ) -> bool:
        if self.strategy.name != "smart_grid_ai":
            return False
        if not self.settings.trading.grid_auto_inventory_bootstrap:
            return False
        if current_quantity > 0:
            return False

        if signal_side == "SELL" and signal_reason in {"grid_sell_rebalance", "grid_take_profit_buffer_hit"}:
            return True

        # Initial paper bootstrapping: if the strategy starts inside-band with no
        # inventory and no prior fills, seed a tiny position so the first SELL legs
        # can execute and operators can observe normal grid behavior.
        if signal_side == "HOLD" and signal_reason in {"grid_wait_inside_band", "grid_recentered_auto"}:
            return not await self._has_paper_fills(session)

        return False

    async def _has_paper_fills(self, session: AsyncSession) -> bool:
        result = await session.execute(
            select(Fill.id)
            .join(Order, Fill.order_id == Order.id)
            .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

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

    def _compute_equity_from_position(self, *, position: Position, mark_price: Decimal) -> Decimal:
        quantity = Decimal(str(position.quantity))
        realized = Decimal(str(position.realized_pnl))
        avg_entry = Decimal(str(position.avg_entry_price))
        unrealized = Decimal("0")
        if quantity > 0 and avg_entry > 0:
            unrealized = (mark_price - avg_entry) * quantity
        return self.starting_equity + realized + unrealized

    async def _resolve_paper_equity_scope(
        self,
        session: AsyncSession,
    ) -> tuple[Decimal | None, int | None]:
        """Resolve global peak and optional scope anchor for mixed-baseline paper history."""
        result = await session.execute(
            select(EquitySnapshot.id, EquitySnapshot.equity)
            .where(EquitySnapshot.is_paper.is_(True))
            .order_by(EquitySnapshot.equity.desc(), EquitySnapshot.id.desc())
            .limit(1)
        )
        row = result.first()
        if row is None:
            return None, None

        peak_snapshot_id = int(row[0])
        global_peak = Decimal(str(row[1]))
        if self.starting_equity <= 0:
            return global_peak, None

        legacy_peak_ratio_threshold = Decimal("4")
        if global_peak <= self.starting_equity * legacy_peak_ratio_threshold:
            return global_peak, None

        baseline_band_lower = self.starting_equity * Decimal("0.5")
        baseline_band_upper = self.starting_equity * Decimal("1.5")
        anchor_result = await session.execute(
            select(EquitySnapshot.id)
            .where(
                EquitySnapshot.is_paper.is_(True),
                EquitySnapshot.id > peak_snapshot_id,
                EquitySnapshot.equity >= baseline_band_lower,
                EquitySnapshot.equity <= baseline_band_upper,
            )
            .order_by(EquitySnapshot.id.asc())
            .limit(1)
        )
        anchor_snapshot_id = anchor_result.scalar_one_or_none()
        if anchor_snapshot_id is None:
            return global_peak, None
        return global_peak, int(anchor_snapshot_id)

    async def _get_peak_equity(self, session: AsyncSession) -> Decimal:
        global_peak, anchor_snapshot_id = await self._resolve_paper_equity_scope(session)
        if global_peak is None:
            return self.starting_equity
        if anchor_snapshot_id is None:
            return max(self.starting_equity, global_peak)

        scoped_peak_result = await session.execute(
            select(func.max(EquitySnapshot.equity)).where(
                EquitySnapshot.is_paper.is_(True),
                EquitySnapshot.id >= anchor_snapshot_id,
            )
        )
        scoped_peak = scoped_peak_result.scalar_one_or_none()
        if scoped_peak is None:
            return self.starting_equity
        return max(self.starting_equity, Decimal(str(scoped_peak)))

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

    async def _resolve_mark_price(
        self,
        session: AsyncSession,
        *,
        fallback_price: Decimal = Decimal("0"),
    ) -> Decimal:
        repo = CandleRepository(session)
        checked_timeframes: list[str] = []
        for timeframe in [self.timing_timeframe, self.regime_timeframe, *self.settings.trading.timeframe_list]:
            if timeframe in checked_timeframes:
                continue
            checked_timeframes.append(timeframe)
            price = await repo.get_latest_close_price(self.symbol, timeframe)
            if price is not None and price > 0:
                return price

        if fallback_price > 0:
            return fallback_price

        result = await session.execute(
            select(Fill.price)
            .join(Order, Fill.order_id == Order.id)
            .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            .order_by(Fill.filled_at.desc())
            .limit(1)
        )
        last_fill_price = result.scalar_one_or_none()
        if last_fill_price is not None:
            return Decimal(str(last_fill_price))

        raise RuntimeError(f"No mark price available to close position for {self.symbol}")

    async def _persist_cycle_snapshot(
        self,
        session: AsyncSession,
        position: Position,
        mark_price: Decimal,
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
        await session.flush()
        metrics = await self._build_metrics_snapshot(session)
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

    async def _compute_daily_realized_pnl(self, session: AsyncSession) -> Decimal:
        """Compute realized PnL since UTC day start from fill history."""
        day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        fills_result = await session.execute(
            select(Fill.filled_at, Fill.quantity, Fill.price, Fill.fee, Order.side)
            .join(Order, Fill.order_id == Order.id)
            .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            .order_by(Fill.filled_at.asc(), Fill.id.asc())
        )
        running_qty = Decimal("0")
        avg_entry = Decimal("0")
        daily_realized = Decimal("0")
        for filled_at_raw, quantity_raw, price_raw, fee_raw, side_raw in fills_result.all():
            filled_at = self._to_utc_datetime(filled_at_raw)
            side = str(side_raw).upper()
            quantity = Decimal(str(quantity_raw))
            price = Decimal(str(price_raw))
            fee = Decimal(str(fee_raw))
            if side == "BUY":
                new_qty = running_qty + quantity
                if new_qty > 0:
                    avg_entry = ((running_qty * avg_entry) + (quantity * price)) / new_qty
                running_qty = new_qty
                continue

            sell_qty = min(quantity, running_qty)
            realized_delta = ((price - avg_entry) * sell_qty) - fee
            if filled_at >= day_start:
                daily_realized += realized_delta
            running_qty = running_qty - sell_qty
            if running_qty <= 0:
                running_qty = Decimal("0")
                avg_entry = Decimal("0")
        return daily_realized

    async def _load_closed_trade_pnls(self, session: AsyncSession) -> list[float]:
        """Reconstruct realized PnL values from BUY/SELL fill history."""
        fills_result = await session.execute(
            select(Fill.quantity, Fill.price, Fill.fee, Order.side)
            .join(Order, Fill.order_id == Order.id)
            .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            .order_by(Fill.filled_at.asc(), Fill.id.asc())
        )
        running_qty = Decimal("0")
        avg_entry = Decimal("0")
        closed_trade_pnls: list[float] = []
        for quantity_raw, price_raw, fee_raw, side_raw in fills_result.all():
            side = str(side_raw).upper()
            quantity = Decimal(str(quantity_raw))
            price = Decimal(str(price_raw))
            fee = Decimal(str(fee_raw))
            if side == "BUY":
                new_qty = running_qty + quantity
                if new_qty > 0:
                    avg_entry = ((running_qty * avg_entry) + (quantity * price)) / new_qty
                running_qty = new_qty
                continue

            sell_qty = min(quantity, running_qty)
            realized_delta = ((price - avg_entry) * sell_qty) - fee
            closed_trade_pnls.append(float(realized_delta))
            running_qty = running_qty - sell_qty
            if running_qty <= 0:
                running_qty = Decimal("0")
                avg_entry = Decimal("0")
        return closed_trade_pnls

    @staticmethod
    def _to_utc_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    async def _build_metrics_snapshot(
        self,
        session: AsyncSession,
    ) -> MetricsSnapshot:
        total_trades = (
            await session.execute(
                select(func.count(Fill.id))
                .join(Order, Fill.order_id == Order.id)
                .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            )
        ).scalar_one()
        total_fees = (
            await session.execute(
                select(func.coalesce(func.sum(Fill.fee), 0))
                .join(Order, Fill.order_id == Order.id)
                .where(Fill.is_paper.is_(True), Order.symbol == self.symbol)
            )
        ).scalar_one()
        position_result = await session.execute(
            select(Position).where(Position.symbol == self.symbol, Position.is_paper.is_(True))
        )
        position = position_result.scalar_one_or_none()
        total_pnl = Decimal("0")
        if position is not None:
            total_pnl = Decimal(str(position.realized_pnl)) + Decimal(str(position.unrealized_pnl))
        _, anchor_snapshot_id = await self._resolve_paper_equity_scope(session)
        equity_query = select(EquitySnapshot.equity).where(EquitySnapshot.is_paper.is_(True))
        if anchor_snapshot_id is not None:
            equity_query = equity_query.where(EquitySnapshot.id >= anchor_snapshot_id)
        equity_query = equity_query.order_by(EquitySnapshot.snapshot_at.asc(), EquitySnapshot.id.asc())
        equity_result = await session.execute(equity_query)
        equity_curve = [float(Decimal(str(value))) for value in equity_result.scalars().all()]
        if not equity_curve:
            equity_curve = [float(self.starting_equity)]

        closed_trade_pnls = await self._load_closed_trade_pnls(session)
        winning_trades = sum(1 for pnl in closed_trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in closed_trade_pnls if pnl < 0)
        closed_trade_count = len(closed_trade_pnls)

        exposure_notional = Decimal("0")
        if position is not None:
            position_qty = Decimal(str(position.quantity))
            avg_entry = Decimal(str(position.avg_entry_price))
            if position_qty > 0 and avg_entry > 0:
                exposure_notional = position_qty * avg_entry
        latest_equity = Decimal(str(equity_curve[-1])) if equity_curve else self.starting_equity
        exposures_pct: list[float] = []
        if latest_equity > 0:
            exposures_pct.append(float((exposure_notional / latest_equity) * Decimal("100")))

        calculator = MetricsCalculator()
        computed = calculator.calculate(
            trade_pnls=closed_trade_pnls,
            equity_curve=equity_curve,
            fees_paid=float(Decimal(str(total_fees))),
            exposures_pct=exposures_pct,
        )
        win_rate = (winning_trades / closed_trade_count) if closed_trade_count > 0 else None
        avg_trade_pnl = None
        if closed_trade_count > 0:
            closed_total = sum(Decimal(str(pnl)) for pnl in closed_trade_pnls)
            avg_trade_pnl = closed_total / Decimal(closed_trade_count)

        return MetricsSnapshot(
            strategy_name=self.strategy.name,
            total_trades=int(total_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_fees=Decimal(str(total_fees)),
            max_drawdown=float(computed.max_drawdown),
            sharpe_ratio=computed.sharpe_ratio,
            sortino_ratio=computed.sortino_ratio,
            profit_factor=computed.profit_factor if closed_trade_count > 0 else None,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            is_paper=True,
        )

_trading_cycle_service: TradingCycleService | None = None
_trading_cycle_service_lock = threading.Lock()


def get_trading_cycle_service() -> TradingCycleService:
    """Get or create singleton trading cycle service."""
    global _trading_cycle_service
    if _trading_cycle_service is None:
        with _trading_cycle_service_lock:
            if _trading_cycle_service is None:
                _trading_cycle_service = TradingCycleService()
    return _trading_cycle_service


def reset_trading_cycle_service() -> None:
    """Reset singleton so next access reloads runtime settings/strategy."""
    global _trading_cycle_service
    with _trading_cycle_service_lock:
        _trading_cycle_service = None
