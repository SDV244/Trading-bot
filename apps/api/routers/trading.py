"""
Trading Endpoints

Endpoints for trading operations, positions, and orders.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from apps.api.security.auth import AuthUser, require_min_role
from packages.core.config import AuthRole

router = APIRouter()


class PositionResponse(BaseModel):
    """Current position response."""

    symbol: str
    side: str | None
    quantity: str
    avg_entry_price: str
    unrealized_pnl: str
    realized_pnl: str
    total_fees: str
    is_paper: bool


class OrderResponse(BaseModel):
    """Order response."""

    id: int
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: str
    price: str | None
    status: str
    is_paper: bool
    strategy_name: str
    signal_reason: str | None
    created_at: datetime


class FillResponse(BaseModel):
    """Fill response."""

    id: int
    order_id: int
    fill_id: str
    quantity: str
    price: str
    fee: str
    fee_asset: str
    is_paper: bool
    slippage_bps: float | None
    filled_at: datetime


class MetricsResponse(BaseModel):
    """Trading metrics response."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float | None
    total_pnl: str
    total_fees: str
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    profit_factor: float | None


class ConfigResponse(BaseModel):
    """Trading configuration response."""

    trading_pair: str
    timeframes: list[str]
    supported_strategies: list[str]
    live_mode: bool
    active_strategy: str
    require_data_ready: bool
    spot_position_mode: str
    paper_starting_equity: float
    advisor_interval_cycles: int
    min_cycle_interval_seconds: int
    reconciliation_interval_cycles: int
    reconciliation_warning_tolerance: float
    reconciliation_critical_tolerance: float
    grid_lookback_1h: int
    grid_atr_period_1h: int
    grid_levels: int
    grid_spacing_mode: str
    grid_min_spacing_bps: int
    grid_max_spacing_bps: int
    grid_trend_tilt: float
    grid_volatility_blend: float
    grid_take_profit_buffer: float
    grid_stop_loss_buffer: float
    grid_cooldown_seconds: int
    grid_auto_inventory_bootstrap: bool
    grid_bootstrap_fraction: float
    grid_enforce_fee_floor: bool
    grid_min_net_profit_bps: int
    grid_out_of_bounds_alert_cooldown_minutes: int
    grid_recenter_mode: str
    stop_loss_enabled: bool
    stop_loss_global_equity_pct: float
    stop_loss_max_drawdown_pct: float
    stop_loss_auto_close_positions: bool
    risk_per_trade: float
    max_daily_loss: float
    max_exposure: float
    fee_bps: int
    slippage_bps: int
    approval_timeout_hours: int
    approval_auto_approve_enabled: bool


class PaperCycleResponse(BaseModel):
    """Paper trading cycle execution response."""

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


class CloseAllPaperPositionsRequest(BaseModel):
    """Request payload to force-close all open paper positions."""

    reason: str = Field(default="manual_close_all_positions", min_length=3, max_length=200)


class PaperResetRequest(BaseModel):
    """Request payload to reset paper-account state/history."""

    reason: str = Field(default="manual_paper_reset", min_length=3, max_length=200)


class PaperResetResponse(BaseModel):
    """Reset summary response."""

    reset: bool
    reason: str
    deleted_orders: int
    deleted_fills: int
    deleted_positions: int
    deleted_equity_snapshots: int
    deleted_metrics_snapshots: int
    paper_starting_equity: float


class LiveOrderRequest(BaseModel):
    """Live market order request (explicitly gated)."""

    side: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: str = Field(..., min_length=1)
    ui_confirmed: bool
    reauthenticated: bool
    safety_acknowledged: bool
    client_order_id: str | None = Field(default=None, max_length=36)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    reason: str = Field(default="", description="Operator reason for live order")


class LiveOrderResponse(BaseModel):
    """Live order execution response."""

    accepted: bool
    reason: str
    order_id: str | None
    quantity: str | None
    price: str | None


class GridRecenterModeUpdateRequest(BaseModel):
    """Grid recenter mode update payload."""

    mode: str = Field(..., pattern="^(conservative|aggressive)$")
    reason: str = Field(default="dashboard_update")
    changed_by: str = Field(default="web_dashboard")


class GridRecenterModeUpdateResponse(BaseModel):
    """Grid recenter mode update response."""

    active_strategy: str
    mode: str
    applied_to_live_strategy: bool


class GridLevelPreviewResponse(BaseModel):
    """One projected smart-grid level."""

    level: int
    price: str
    distance_bps: float


class GridPreviewResponse(BaseModel):
    """Smart-grid level/trigger preview for operator UX."""

    symbol: str
    strategy: str
    last_price: str
    signal_side: str
    signal_reason: str
    confidence: float
    grid_center: str | None
    grid_upper: str | None
    grid_lower: str | None
    buy_trigger: str | None
    sell_trigger: str | None
    spacing_bps: float | None
    grid_step: str | None
    recentered: bool
    recenter_mode: str
    buy_levels: list[GridLevelPreviewResponse]
    sell_levels: list[GridLevelPreviewResponse]
    take_profit_trigger: str | None
    stop_loss_trigger: str | None
    position_quantity: str
    bootstrap_eligible: bool


async def _ensure_database() -> None:
    from packages.core.database.session import init_database

    await init_database()


@router.get("/config", response_model=ConfigResponse)
async def get_trading_config(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> ConfigResponse:
    """Get current trading configuration."""
    from packages.core.config import get_settings
    from packages.core.strategies import registry
    from packages.core.trading_cycle import get_trading_cycle_service

    settings = get_settings()
    service = get_trading_cycle_service()
    effective_recenter_mode = settings.trading.grid_recenter_mode
    if service.strategy.name == "smart_grid_ai":
        effective_recenter_mode = cast(str, getattr(service.strategy, "recenter_mode", effective_recenter_mode))
    return ConfigResponse(
        trading_pair=settings.trading.pair,
        timeframes=settings.trading.timeframe_list,
        supported_strategies=registry.list_names(),
        live_mode=settings.trading.live_mode,
        active_strategy=settings.trading.active_strategy,
        require_data_ready=settings.trading.require_data_ready,
        spot_position_mode=settings.trading.spot_position_mode,
        paper_starting_equity=settings.trading.paper_starting_equity,
        advisor_interval_cycles=settings.trading.advisor_interval_cycles,
        min_cycle_interval_seconds=settings.trading.min_cycle_interval_seconds,
        reconciliation_interval_cycles=settings.trading.reconciliation_interval_cycles,
        reconciliation_warning_tolerance=settings.trading.reconciliation_warning_tolerance,
        reconciliation_critical_tolerance=settings.trading.reconciliation_critical_tolerance,
        grid_lookback_1h=settings.trading.grid_lookback_1h,
        grid_atr_period_1h=settings.trading.grid_atr_period_1h,
        grid_levels=settings.trading.grid_levels,
        grid_spacing_mode=settings.trading.grid_spacing_mode,
        grid_min_spacing_bps=settings.trading.grid_min_spacing_bps,
        grid_max_spacing_bps=settings.trading.grid_max_spacing_bps,
        grid_trend_tilt=settings.trading.grid_trend_tilt,
        grid_volatility_blend=settings.trading.grid_volatility_blend,
        grid_take_profit_buffer=settings.trading.grid_take_profit_buffer,
        grid_stop_loss_buffer=settings.trading.grid_stop_loss_buffer,
        grid_cooldown_seconds=settings.trading.grid_cooldown_seconds,
        grid_auto_inventory_bootstrap=settings.trading.grid_auto_inventory_bootstrap,
        grid_bootstrap_fraction=settings.trading.grid_bootstrap_fraction,
        grid_enforce_fee_floor=settings.trading.grid_enforce_fee_floor,
        grid_min_net_profit_bps=settings.trading.grid_min_net_profit_bps,
        grid_out_of_bounds_alert_cooldown_minutes=settings.trading.grid_out_of_bounds_alert_cooldown_minutes,
        grid_recenter_mode=effective_recenter_mode,
        stop_loss_enabled=settings.trading.stop_loss_enabled,
        stop_loss_global_equity_pct=settings.trading.stop_loss_global_equity_pct,
        stop_loss_max_drawdown_pct=settings.trading.stop_loss_max_drawdown_pct,
        stop_loss_auto_close_positions=settings.trading.stop_loss_auto_close_positions,
        risk_per_trade=settings.risk.per_trade,
        max_daily_loss=settings.risk.max_daily_loss,
        max_exposure=settings.risk.max_exposure,
        fee_bps=settings.risk.fee_bps,
        slippage_bps=settings.risk.slippage_bps,
        approval_timeout_hours=settings.approval.timeout_hours,
        approval_auto_approve_enabled=settings.approval.auto_approve_enabled,
    )


@router.post("/config/recenter-mode", response_model=GridRecenterModeUpdateResponse)
async def set_grid_recenter_mode(
    payload: GridRecenterModeUpdateRequest,
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> GridRecenterModeUpdateResponse:
    """Update smart-grid recenter behavior (conservative/aggressive)."""
    from packages.core.audit import log_event
    from packages.core.config import get_settings
    from packages.core.database.session import get_session
    from packages.core.trading_cycle import get_trading_cycle_service

    await _ensure_database()
    service = get_trading_cycle_service()
    mode = service.set_grid_recenter_mode(payload.mode)
    get_settings().trading.grid_recenter_mode = mode
    applied = service.strategy.name == "smart_grid_ai"

    async with get_session() as session:
        await log_event(
            session,
            event_type="grid_recenter_mode_changed",
            event_category="config",
            summary=f"Grid recenter mode changed to {mode}",
            details={
                "mode": mode,
                "active_strategy": service.strategy.name,
                "applied_to_live_strategy": applied,
                "reason": payload.reason,
            },
            actor=payload.changed_by or "web_dashboard",
        )
        await session.commit()

    return GridRecenterModeUpdateResponse(
        active_strategy=service.strategy.name,
        mode=mode,
        applied_to_live_strategy=applied,
    )


@router.get("/position", response_model=PositionResponse)
async def get_current_position(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> PositionResponse:
    """Get current position for BTCUSDT."""
    from packages.core.config import get_settings
    from packages.core.database.models import Position
    from packages.core.database.session import get_session

    await _ensure_database()
    settings = get_settings()

    async with get_session() as session:
        result = await session.execute(
            select(Position).where(
                Position.symbol == settings.trading.pair,
                Position.is_paper.is_(True),
            )
        )
        position = result.scalar_one_or_none()

    if position is None:
        return PositionResponse(
            symbol=settings.trading.pair,
            side=None,
            quantity="0",
            avg_entry_price="0",
            unrealized_pnl="0",
            realized_pnl="0",
            total_fees="0",
            is_paper=True,
        )

    return PositionResponse(
        symbol=position.symbol,
        side=position.side,
        quantity=str(position.quantity),
        avg_entry_price=str(position.avg_entry_price),
        unrealized_pnl=str(position.unrealized_pnl),
        realized_pnl=str(position.realized_pnl),
        total_fees=str(position.total_fees),
        is_paper=position.is_paper,
    )


@router.get("/grid/preview", response_model=GridPreviewResponse | None)
async def get_grid_preview(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> GridPreviewResponse | None:
    """Get current smart-grid projected levels/triggers (if strategy supports it)."""
    from packages.core.database.session import get_session
    from packages.core.trading_cycle import get_trading_cycle_service

    await _ensure_database()
    service = get_trading_cycle_service()

    async with get_session() as session:
        preview = await service.get_grid_preview(session)

    if preview is None:
        return None

    return GridPreviewResponse(
        symbol=preview.symbol,
        strategy=preview.strategy,
        last_price=str(preview.last_price),
        signal_side=preview.signal_side,
        signal_reason=preview.signal_reason,
        confidence=preview.confidence,
        grid_center=str(preview.grid_center) if preview.grid_center is not None else None,
        grid_upper=str(preview.grid_upper) if preview.grid_upper is not None else None,
        grid_lower=str(preview.grid_lower) if preview.grid_lower is not None else None,
        buy_trigger=str(preview.buy_trigger) if preview.buy_trigger is not None else None,
        sell_trigger=str(preview.sell_trigger) if preview.sell_trigger is not None else None,
        spacing_bps=preview.spacing_bps,
        grid_step=str(preview.grid_step) if preview.grid_step is not None else None,
        recentered=preview.recentered,
        recenter_mode=preview.recenter_mode,
        buy_levels=[
            GridLevelPreviewResponse(
                level=level.level,
                price=str(level.price),
                distance_bps=float(level.distance_bps),
            )
            for level in preview.buy_levels
        ],
        sell_levels=[
            GridLevelPreviewResponse(
                level=level.level,
                price=str(level.price),
                distance_bps=float(level.distance_bps),
            )
            for level in preview.sell_levels
        ],
        take_profit_trigger=str(preview.take_profit_trigger) if preview.take_profit_trigger is not None else None,
        stop_loss_trigger=str(preview.stop_loss_trigger) if preview.stop_loss_trigger is not None else None,
        position_quantity=str(preview.position_quantity),
        bootstrap_eligible=preview.bootstrap_eligible,
    )


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Number of orders to return"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[OrderResponse]:
    """Get recent orders."""
    from packages.core.database.models import Order
    from packages.core.database.session import get_session

    await _ensure_database()

    async with get_session() as session:
        query = select(Order).where(Order.is_paper.is_(True)).order_by(Order.created_at.desc()).limit(limit)
        if status:
            query = query.where(Order.status == status.upper())

        result = await session.execute(query)
        orders = result.scalars().all()

    return [
        OrderResponse(
            id=o.id,
            client_order_id=o.client_order_id,
            symbol=o.symbol,
            side=o.side,
            order_type=o.order_type,
            quantity=str(o.quantity),
            price=str(o.price) if o.price is not None else None,
            status=o.status,
            is_paper=o.is_paper,
            strategy_name=o.strategy_name,
            signal_reason=o.signal_reason,
            created_at=o.created_at,
        )
        for o in orders
    ]


@router.get("/fills", response_model=list[FillResponse])
async def get_fills(
    limit: int = Query(50, ge=1, le=200, description="Number of fills to return"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[FillResponse]:
    """Get recent fills."""
    from packages.core.database.models import Fill
    from packages.core.database.session import get_session

    await _ensure_database()

    async with get_session() as session:
        result = await session.execute(
            select(Fill).where(Fill.is_paper.is_(True)).order_by(Fill.filled_at.desc()).limit(limit)
        )
        fills = result.scalars().all()

    return [
        FillResponse(
            id=f.id,
            order_id=f.order_id,
            fill_id=f.fill_id,
            quantity=str(f.quantity),
            price=str(f.price),
            fee=str(f.fee),
            fee_asset=f.fee_asset,
            is_paper=f.is_paper,
            slippage_bps=f.slippage_bps,
            filled_at=f.filled_at,
        )
        for f in fills
    ]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(_: AuthUser = Depends(require_min_role(AuthRole.VIEWER))) -> MetricsResponse:
    """Get current trading metrics."""
    from packages.core.database.models import MetricsSnapshot
    from packages.core.database.session import get_session

    await _ensure_database()

    async with get_session() as session:
        result = await session.execute(
            select(MetricsSnapshot)
            .where(MetricsSnapshot.is_paper.is_(True))
            .order_by(MetricsSnapshot.snapshot_at.desc())
            .limit(1)
        )
        metrics = result.scalars().first()

    if metrics is None:
        return MetricsResponse(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=None,
            total_pnl="0",
            total_fees="0",
            max_drawdown=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            profit_factor=None,
        )

    return MetricsResponse(
        total_trades=metrics.total_trades,
        winning_trades=metrics.winning_trades,
        losing_trades=metrics.losing_trades,
        win_rate=metrics.win_rate,
        total_pnl=str(metrics.total_pnl),
        total_fees=str(metrics.total_fees),
        max_drawdown=metrics.max_drawdown,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        profit_factor=metrics.profit_factor,
    )


@router.get("/equity/history")
async def get_equity_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[dict[str, Any]]:
    """Get equity history for charting."""
    from packages.core.database.models import EquitySnapshot
    from packages.core.database.session import get_session

    await _ensure_database()
    cutoff = datetime.now(UTC) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            select(EquitySnapshot)
            .where(
                EquitySnapshot.is_paper.is_(True),
                EquitySnapshot.snapshot_at >= cutoff,
            )
            .order_by(EquitySnapshot.snapshot_at.asc())
        )
        snapshots = result.scalars().all()

    return [
        {
            "timestamp": snapshot.snapshot_at.isoformat(),
            "equity": str(snapshot.equity),
            "available_balance": str(snapshot.available_balance),
            "unrealized_pnl": str(snapshot.unrealized_pnl),
        }
        for snapshot in snapshots
    ]


@router.post("/paper/cycle", response_model=PaperCycleResponse)
async def run_paper_cycle(
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> PaperCycleResponse:
    """Run one paper trading cycle."""
    from packages.core.database.session import get_session
    from packages.core.execution_lock import symbol_execution_lock
    from packages.core.trading_cycle import get_trading_cycle_service

    try:
        await _ensure_database()
        service = get_trading_cycle_service()
        async with symbol_execution_lock(service.symbol), get_session() as session:
            result = await service.run_once(session)
            return PaperCycleResponse(
                symbol=result.symbol,
                signal_side=result.signal_side,
                signal_reason=result.signal_reason,
                risk_action=result.risk_action,
                risk_reason=result.risk_reason,
                executed=result.executed,
                order_id=result.order_id,
                fill_id=result.fill_id,
                quantity=result.quantity,
                price=result.price,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paper cycle failed: {e}")


@router.post("/paper/close-all-positions", response_model=PaperCycleResponse)
async def close_all_paper_positions(
    request: CloseAllPaperPositionsRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> PaperCycleResponse:
    """Force-close any open paper position inventory at current market price."""
    from packages.core.audit import log_event
    from packages.core.database.session import get_session
    from packages.core.execution_lock import symbol_execution_lock
    from packages.core.trading_cycle import get_trading_cycle_service

    await _ensure_database()
    service = get_trading_cycle_service()
    try:
        async with symbol_execution_lock(service.symbol), get_session() as session:
            await log_event(
                session,
                event_type="paper_close_position_requested",
                event_category="trade",
                summary=f"Manual paper close-all requested: {request.reason}",
                details={"symbol": service.symbol, "reason": request.reason},
                actor=user.username,
            )
            result = await service.close_open_paper_position(
                session,
                reason=request.reason,
                actor=user.username,
            )
            return PaperCycleResponse(
                symbol=result.symbol,
                signal_side=result.signal_side,
                signal_reason=result.signal_reason,
                risk_action=result.risk_action,
                risk_reason=result.risk_reason,
                executed=result.executed,
                order_id=result.order_id,
                fill_id=result.fill_id,
                quantity=result.quantity,
                price=result.price,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Close-all paper positions failed: {e}")


@router.post("/paper/reset", response_model=PaperResetResponse)
async def reset_paper_account(
    request: PaperResetRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> PaperResetResponse:
    """Reset paper-account trade history, position state, and performance snapshots."""
    from packages.core.audit import log_event
    from packages.core.config import get_settings
    from packages.core.database.models import EquitySnapshot, Fill, MetricsSnapshot, Order, Position
    from packages.core.database.session import get_session
    from packages.core.state import get_state_manager
    from packages.core.trading_cycle import reset_trading_cycle_service

    await _ensure_database()
    state = get_state_manager()
    if state.current.state.value != "paused":
        raise HTTPException(
            status_code=409,
            detail="Paper reset requires system state PAUSED. Pause first, then reset.",
        )

    settings = get_settings()
    deleted_orders = 0
    deleted_fills = 0
    deleted_positions = 0
    deleted_equity_snapshots = 0
    deleted_metrics_snapshots = 0

    try:
        async with get_session() as session:
            fills_result = await session.execute(delete(Fill).where(Fill.is_paper.is_(True)))
            deleted_fills = int(fills_result.rowcount or 0)

            orders_result = await session.execute(delete(Order).where(Order.is_paper.is_(True)))
            deleted_orders = int(orders_result.rowcount or 0)

            positions_result = await session.execute(delete(Position).where(Position.is_paper.is_(True)))
            deleted_positions = int(positions_result.rowcount or 0)

            equity_result = await session.execute(
                delete(EquitySnapshot).where(EquitySnapshot.is_paper.is_(True))
            )
            deleted_equity_snapshots = int(equity_result.rowcount or 0)

            metrics_result = await session.execute(
                delete(MetricsSnapshot).where(MetricsSnapshot.is_paper.is_(True))
            )
            deleted_metrics_snapshots = int(metrics_result.rowcount or 0)

            await log_event(
                session,
                event_type="paper_account_reset",
                event_category="system",
                summary="Paper account reset completed",
                details={
                    "reason": request.reason,
                    "deleted_orders": deleted_orders,
                    "deleted_fills": deleted_fills,
                    "deleted_positions": deleted_positions,
                    "deleted_equity_snapshots": deleted_equity_snapshots,
                    "deleted_metrics_snapshots": deleted_metrics_snapshots,
                    "paper_starting_equity": settings.trading.paper_starting_equity,
                },
                actor=user.username,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paper reset failed: {e}")

    reset_trading_cycle_service()
    return PaperResetResponse(
        reset=True,
        reason=request.reason,
        deleted_orders=deleted_orders,
        deleted_fills=deleted_fills,
        deleted_positions=deleted_positions,
        deleted_equity_snapshots=deleted_equity_snapshots,
        deleted_metrics_snapshots=deleted_metrics_snapshots,
        paper_starting_equity=settings.trading.paper_starting_equity,
    )


@router.post("/live/order", response_model=LiveOrderResponse)
async def submit_live_order(
    request: LiveOrderRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.ADMIN)),
) -> LiveOrderResponse:
    """Submit one live market order with mandatory safety checklist."""
    from decimal import Decimal, InvalidOperation
    from uuid import uuid4

    from packages.core.audit import log_event, resolve_active_config_version
    from packages.core.config import Settings, get_settings
    from packages.core.database.models import Fill, Order, OrderAttempt, OrderAttemptStatus
    from packages.core.database.session import get_session
    from packages.core.execution import (
        LiveEngine,
        LiveEngineError,
        LiveSafetyChecklist,
        OrderRequest,
    )
    from packages.core.observability import increment_live_order
    from packages.core.state import get_state_manager

    await _ensure_database()
    settings = get_settings()
    if not settings.trading.live_mode:
        raise HTTPException(status_code=400, detail="live_mode is disabled")
    if not get_state_manager().can_trade:
        raise HTTPException(status_code=400, detail="system is not RUNNING")

    try:
        quantity = Decimal(request.quantity)
    except (InvalidOperation, ValueError):
        raise HTTPException(status_code=422, detail="Invalid quantity")
    side = cast(Literal["BUY", "SELL"], request.side)
    symbol = settings.trading.pair

    def _derive_idempotency_key(settings_obj: Settings) -> str:
        if request.idempotency_key:
            return request.idempotency_key.strip().lower()
        # Five-minute bucket prevents accidental duplicate clicks/retries
        # while still allowing new manual orders later with same payload.
        bucket = int(datetime.now(UTC).timestamp() // 300)
        material = "|".join(
            [
                symbol,
                side,
                format(quantity.normalize(), "f"),
                request.client_order_id or "",
                request.reason or "",
                user.username,
                str(bucket),
                str(settings_obj.live.recv_window_ms),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    idempotency_key = _derive_idempotency_key(settings)
    explicit_client_order_id = request.client_order_id or f"live_{uuid4().hex[:24]}"

    async with get_session() as session:
        config_version = await resolve_active_config_version(session)
        attempt_result = await session.execute(
            select(OrderAttempt).where(OrderAttempt.idempotency_key == idempotency_key)
        )
        attempt = attempt_result.scalar_one_or_none()

        if attempt is None:
            attempt = OrderAttempt(
                idempotency_key=idempotency_key,
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity,
                status=OrderAttemptStatus.PENDING.value,
                client_order_id=explicit_client_order_id,
                exchange_order_id=None,
                error_message=None,
            )
            session.add(attempt)
            await session.commit()
        else:
            explicit_client_order_id = attempt.client_order_id

        existing_order_result = await session.execute(
            select(Order).where(
                Order.client_order_id == explicit_client_order_id,
                Order.is_paper.is_(False),
            )
        )
        existing_order = existing_order_result.scalar_one_or_none()
        if existing_order is None and attempt.status == OrderAttemptStatus.CONFIRMED.value:
            return LiveOrderResponse(
                accepted=True,
                reason="idempotent_replay_confirmed_exchange_only",
                order_id=attempt.exchange_order_id,
                quantity=str(attempt.quantity),
                price=None,
            )
        if existing_order is not None and attempt.status == OrderAttemptStatus.CONFIRMED.value:
            existing_fill_result = await session.execute(
                select(Fill).where(Fill.order_id == existing_order.id).limit(1)
            )
            existing_fill = existing_fill_result.scalar_one_or_none()
            return LiveOrderResponse(
                accepted=existing_order.status == "FILLED",
                reason="idempotent_replay_confirmed",
                order_id=existing_order.exchange_order_id,
                quantity=str(existing_order.quantity),
                price=str(existing_fill.price) if existing_fill is not None else None,
            )

        engine = LiveEngine()
        try:
            result = await engine.execute_market_order(
                OrderRequest(symbol=symbol, side=side, quantity=quantity, order_type="MARKET"),
                checklist=LiveSafetyChecklist(
                    ui_confirmed=request.ui_confirmed,
                    reauthenticated=request.reauthenticated,
                    safety_acknowledged=request.safety_acknowledged,
                ),
                client_order_id=explicit_client_order_id,
            )
        except LiveEngineError as e:
            attempt.status = OrderAttemptStatus.FAILED.value
            attempt.error_message = str(e)
            attempt.last_checked_at = datetime.now(UTC)
            await session.commit()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            attempt.status = OrderAttemptStatus.FAILED.value
            attempt.error_message = str(e)
            attempt.last_checked_at = datetime.now(UTC)
            await session.commit()
            raise HTTPException(status_code=503, detail=f"Live order failed: {e}")

        persisted_order = existing_order
        if persisted_order is None:
            persisted_order = Order(
                client_order_id=result.client_order_id or explicit_client_order_id,
                exchange_order_id=result.order_id,
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=result.quantity or quantity,
                price=result.price,
                status="FILLED" if result.accepted else "REJECTED",
                is_paper=False,
                strategy_name="manual_live",
                signal_reason=request.reason or "manual_live_order",
                config_version=config_version,
            )
            session.add(persisted_order)
            await session.flush()

            if result.accepted and result.quantity is not None and result.price is not None:
                session.add(
                    Fill(
                        order_id=persisted_order.id,
                        fill_id=f"live_{persisted_order.id}_{result.order_id or 'na'}",
                        quantity=result.quantity,
                        price=result.price,
                        fee=Decimal("0"),
                        fee_asset="USDT",
                        is_paper=False,
                        slippage_bps=None,
                    )
                )

        attempt.status = (
            OrderAttemptStatus.CONFIRMED.value if result.accepted else OrderAttemptStatus.FAILED.value
        )
        attempt.exchange_order_id = result.order_id
        attempt.error_message = None if result.accepted else result.reason
        attempt.last_checked_at = datetime.now(UTC)

        await log_event(
            session,
            event_type="live_order_submitted",
            event_category="trade",
            summary=f"Live order {side} {request.quantity} {symbol}",
            details={
                "accepted": result.accepted,
                "reason": result.reason,
                "exchange_order_id": result.order_id,
                "idempotency_key": idempotency_key,
                "client_order_id": explicit_client_order_id,
            },
            inputs=request.model_dump(),
            actor=user.username,
            config_version=config_version,
        )
        increment_live_order(accepted=result.accepted, side=side)

        return LiveOrderResponse(
            accepted=result.accepted,
            reason=result.reason,
            order_id=result.order_id,
            quantity=str(result.quantity) if result.quantity is not None else None,
            price=str(result.price) if result.price is not None else None,
        )
