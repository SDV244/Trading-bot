"""
SQLAlchemy ORM Models for Trading Bot

All database tables for orders, fills, positions, metrics, and audit logging.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ApprovalStatus(str, Enum):
    """Approval status enumeration."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ============================================================================
# Configuration
# ============================================================================


class Config(Base):
    """Runtime configuration snapshots."""

    __tablename__ = "config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (Index("idx_config_active", "active"),)


# ============================================================================
# Market Data
# ============================================================================


class Candle(Base):
    """OHLCV candle data cache."""

    __tablename__ = "candles_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    quote_volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    trades_count: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_candles_symbol_tf_time", "symbol", "timeframe", "open_time", unique=True),
    )


# ============================================================================
# Orders and Trades
# ============================================================================


class Order(Base):
    """Order requests (paper and live)."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    exchange_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # MARKET, LIMIT
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)  # For limit orders
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=OrderStatus.PENDING)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_reason: Mapped[str] = mapped_column(Text, nullable=True)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order")

    __table_args__ = (
        Index("idx_orders_symbol_status", "symbol", "status"),
        Index("idx_orders_created", "created_at"),
    )


class Fill(Base):
    """Executed fills (simulated or live)."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("orders.id"), nullable=False)
    fill_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee_asset: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=True)
    filled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    __table_args__ = (Index("idx_fills_order", "order_id"),)


# ============================================================================
# Position Tracking
# ============================================================================


class Position(Base):
    """Current position state."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)  # None = no position
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    total_fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================================
# Equity and Metrics
# ============================================================================


class EquitySnapshot(Base):
    """Periodic equity tracking."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    available_balance: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("idx_equity_time", "snapshot_at"),)


class MetricsSnapshot(Base):
    """Strategy performance metrics."""

    __tablename__ = "metrics_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    total_fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_trade_pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("idx_metrics_strategy_time", "strategy_name", "snapshot_at"),)


# ============================================================================
# AI Approvals
# ============================================================================


class Approval(Base):
    """AI proposals with approval status."""

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    diff: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    expected_impact: Mapped[str] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=ApprovalStatus.PENDING)
    ttl_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_approvals_status", "status"),
        Index("idx_approvals_expires", "expires_at"),
    )


# ============================================================================
# Audit Log
# ============================================================================


class EventLog(Base):
    """Append-only audit trail."""

    __tablename__ = "events_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_category: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # trade, system, ai, etc.
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(100), nullable=False, default="system")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_events_type", "event_type"),
        Index("idx_events_category", "event_category"),
        Index("idx_events_time", "created_at"),
    )
