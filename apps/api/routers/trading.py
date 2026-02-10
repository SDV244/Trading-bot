"""
Trading Endpoints

Endpoints for trading operations, positions, and orders.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

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


class LiveOrderRequest(BaseModel):
    """Live market order request (explicitly gated)."""

    side: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: str = Field(..., min_length=1)
    ui_confirmed: bool
    reauthenticated: bool
    safety_acknowledged: bool
    client_order_id: str | None = Field(default=None, max_length=36)
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
    from packages.core.trading_cycle import get_trading_cycle_service

    try:
        await _ensure_database()
        service = get_trading_cycle_service()
        async with get_session() as session:
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


@router.post("/live/order", response_model=LiveOrderResponse)
async def submit_live_order(
    request: LiveOrderRequest,
    user: AuthUser = Depends(require_min_role(AuthRole.ADMIN)),
) -> LiveOrderResponse:
    """Submit one live market order with mandatory safety checklist."""
    from decimal import Decimal
    from uuid import uuid4

    from packages.core.audit import log_event
    from packages.core.config import get_settings
    from packages.core.database.models import Fill, Order
    from packages.core.database.session import get_session
    from packages.core.execution import (
        LiveEngine,
        LiveEngineError,
        LiveSafetyChecklist,
        OrderRequest,
    )
    from packages.core.state import get_state_manager

    await _ensure_database()
    settings = get_settings()
    if not settings.trading.live_mode:
        raise HTTPException(status_code=400, detail="live_mode is disabled")
    if not get_state_manager().can_trade:
        raise HTTPException(status_code=400, detail="system is not RUNNING")

    try:
        quantity = Decimal(request.quantity)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid quantity")
    side = cast(Literal["BUY", "SELL"], request.side)

    engine = LiveEngine()
    try:
        result = await engine.execute_market_order(
            OrderRequest(symbol=settings.trading.pair, side=side, quantity=quantity, order_type="MARKET"),
            checklist=LiveSafetyChecklist(
                ui_confirmed=request.ui_confirmed,
                reauthenticated=request.reauthenticated,
                safety_acknowledged=request.safety_acknowledged,
            ),
            client_order_id=request.client_order_id,
        )
    except LiveEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async with get_session() as session:
        order = Order(
            client_order_id=result.client_order_id or request.client_order_id or f"live_{uuid4().hex[:24]}",
            exchange_order_id=result.order_id,
            symbol=settings.trading.pair,
            side=side,
            order_type="MARKET",
            quantity=result.quantity or quantity,
            price=result.price,
            status="FILLED" if result.accepted else "REJECTED",
            is_paper=False,
            strategy_name="manual_live",
            signal_reason=request.reason or "manual_live_order",
            config_version=1,
        )
        session.add(order)
        await session.flush()

        if result.accepted and result.quantity is not None and result.price is not None:
            session.add(
                Fill(
                    order_id=order.id,
                    fill_id=f"live_{result.order_id or order.id}",
                    quantity=result.quantity,
                    price=result.price,
                    fee=Decimal("0"),
                    fee_asset="USDT",
                    is_paper=False,
                    slippage_bps=None,
                )
            )

        await log_event(
            session,
            event_type="live_order_submitted",
            event_category="trade",
            summary=f"Live order {side} {request.quantity} {settings.trading.pair}",
            details={
                "accepted": result.accepted,
                "reason": result.reason,
                "exchange_order_id": result.order_id,
            },
            inputs=request.model_dump(),
            actor=user.username,
        )

    return LiveOrderResponse(
        accepted=result.accepted,
        reason=result.reason,
        order_id=result.order_id,
        quantity=str(result.quantity) if result.quantity is not None else None,
        price=str(result.price) if result.price is not None else None,
    )
