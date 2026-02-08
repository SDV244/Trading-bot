"""
Trading Endpoints

Endpoints for trading operations, positions, and orders.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

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
    live_mode: bool
    risk_per_trade: float
    max_daily_loss: float
    max_exposure: float
    fee_bps: int
    slippage_bps: int
    approval_timeout_hours: int


@router.get("/config", response_model=ConfigResponse)
async def get_trading_config() -> ConfigResponse:
    """Get current trading configuration."""
    from packages.core.config import get_settings

    settings = get_settings()
    return ConfigResponse(
        trading_pair=settings.trading.pair,
        timeframes=settings.trading.timeframe_list,
        live_mode=settings.trading.live_mode,
        risk_per_trade=settings.risk.per_trade,
        max_daily_loss=settings.risk.max_daily_loss,
        max_exposure=settings.risk.max_exposure,
        fee_bps=settings.risk.fee_bps,
        slippage_bps=settings.risk.slippage_bps,
        approval_timeout_hours=settings.approval.timeout_hours,
    )


@router.get("/position", response_model=PositionResponse)
async def get_current_position() -> PositionResponse:
    """Get current position for BTCUSDT."""
    # TODO: Fetch from database
    return PositionResponse(
        symbol="BTCUSDT",
        side=None,
        quantity="0",
        avg_entry_price="0",
        unrealized_pnl="0",
        realized_pnl="0",
        total_fees="0",
        is_paper=True,
    )


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Number of orders to return"),
) -> list[OrderResponse]:
    """Get recent orders."""
    # TODO: Fetch from database
    return []


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get current trading metrics."""
    # TODO: Fetch from database
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


@router.get("/equity/history")
async def get_equity_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
) -> list[dict[str, Any]]:
    """Get equity history for charting."""
    # TODO: Fetch from database
    return []
