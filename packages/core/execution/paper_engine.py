"""Paper execution engine for simulated spot fills."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal
from uuid import uuid4

OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]


class PaperExecutionError(ValueError):
    """Raised when an order cannot be executed in paper mode."""


@dataclass(slots=True, frozen=True)
class OrderRequest:
    """Order request for paper execution."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = "MARKET"
    limit_price: Decimal | None = None


@dataclass(slots=True, frozen=True)
class PaperFill:
    """Simulated fill result."""

    fill_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    price: Decimal | None
    fee: Decimal
    notional: Decimal
    slippage_bps: int
    status: Literal["FILLED", "PARTIALLY_FILLED", "OPEN"]
    reason: str | None = None


class PaperEngine:
    """Spot-only paper execution engine."""

    def __init__(
        self,
        fee_bps: int = 10,
        slippage_bps: int = 5,
        min_notional: Decimal = Decimal("10"),
        lot_step_size: Decimal = Decimal("0.00001"),
        allowed_symbols: set[str] | None = None,
    ) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.min_notional = min_notional
        self.lot_step_size = lot_step_size
        self.allowed_symbols = allowed_symbols

    def execute_market_order(self, order: OrderRequest, last_price: Decimal) -> PaperFill:
        """Execute a simulated market order with configurable slippage."""
        self._validate_order(order)
        if order.order_type != "MARKET":
            msg = "execute_market_order only accepts MARKET orders"
            raise PaperExecutionError(msg)
        if last_price <= 0:
            msg = "last_price must be positive"
            raise PaperExecutionError(msg)

        slippage_multiplier = Decimal("1") + self._signed_slippage(order.side)
        fill_price = (last_price * slippage_multiplier).quantize(Decimal("0.01"))
        filled_quantity = order.quantity
        notional = fill_price * filled_quantity
        self._validate_notional(notional)
        fee = self._calculate_fee(notional)

        return PaperFill(
            fill_id=f"paper_{uuid4().hex[:16]}",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=Decimal("0"),
            price=fill_price,
            fee=fee,
            notional=notional,
            slippage_bps=self.slippage_bps,
            status="FILLED",
            reason="market_fill",
        )

    def execute_limit_order(
        self,
        order: OrderRequest,
        candle_low: Decimal,
        candle_high: Decimal,
        partial_fill_ratio: Decimal | None = None,
    ) -> PaperFill:
        """Execute a simulated limit order using candle range."""
        self._validate_order(order)
        if order.order_type != "LIMIT":
            msg = "execute_limit_order only accepts LIMIT orders"
            raise PaperExecutionError(msg)
        if order.limit_price is None:
            msg = "LIMIT orders require limit_price"
            raise PaperExecutionError(msg)
        if candle_low <= 0 or candle_high <= 0:
            msg = "candle_low and candle_high must be positive"
            raise PaperExecutionError(msg)
        if candle_high < candle_low:
            msg = "candle_high cannot be lower than candle_low"
            raise PaperExecutionError(msg)

        touched = self._limit_touched(
            side=order.side,
            limit_price=order.limit_price,
            candle_low=candle_low,
            candle_high=candle_high,
        )
        if not touched:
            return PaperFill(
                fill_id=f"paper_{uuid4().hex[:16]}",
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                requested_quantity=order.quantity,
                filled_quantity=Decimal("0"),
                remaining_quantity=order.quantity,
                price=None,
                fee=Decimal("0"),
                notional=Decimal("0"),
                slippage_bps=0,
                status="OPEN",
                reason="limit_not_touched",
            )

        fill_ratio = partial_fill_ratio if partial_fill_ratio is not None else Decimal("1")
        if fill_ratio <= 0 or fill_ratio > 1:
            msg = "partial_fill_ratio must be > 0 and <= 1"
            raise PaperExecutionError(msg)

        filled_quantity = (order.quantity * fill_ratio).quantize(self.lot_step_size, rounding=ROUND_DOWN)
        if filled_quantity <= 0:
            msg = "partial_fill_ratio produced zero quantity fill"
            raise PaperExecutionError(msg)

        remaining_quantity = order.quantity - filled_quantity
        notional = order.limit_price * filled_quantity
        self._validate_notional(notional)
        fee = self._calculate_fee(notional)
        status: Literal["FILLED", "PARTIALLY_FILLED"] = (
            "FILLED" if remaining_quantity == 0 else "PARTIALLY_FILLED"
        )

        return PaperFill(
            fill_id=f"paper_{uuid4().hex[:16]}",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            price=order.limit_price,
            fee=fee,
            notional=notional,
            slippage_bps=0,
            status=status,
            reason="limit_fill",
        )

    def _signed_slippage(self, side: OrderSide) -> Decimal:
        bps = Decimal(self.slippage_bps) / Decimal("10000")
        return bps if side == "BUY" else -bps

    def _calculate_fee(self, notional: Decimal) -> Decimal:
        return (notional * Decimal(self.fee_bps) / Decimal("10000")).quantize(Decimal("0.00000001"))

    def _validate_order(self, order: OrderRequest) -> None:
        if self.allowed_symbols is not None and order.symbol not in self.allowed_symbols:
            msg = f"Symbol {order.symbol} is not allowed in this paper engine instance"
            raise PaperExecutionError(msg)
        if order.quantity <= 0:
            msg = "quantity must be positive"
            raise PaperExecutionError(msg)
        if order.order_type == "LIMIT" and order.limit_price is None:
            msg = "LIMIT orders require limit_price"
            raise PaperExecutionError(msg)
        if order.order_type == "LIMIT" and order.limit_price is not None and order.limit_price <= 0:
            msg = "limit_price must be positive"
            raise PaperExecutionError(msg)

        # Quantity must be an exact multiple of lot step size.
        units = (order.quantity / self.lot_step_size).normalize()
        if units != units.to_integral_value():
            msg = f"quantity must align with lot_step_size={self.lot_step_size}"
            raise PaperExecutionError(msg)

    def _validate_notional(self, notional: Decimal) -> None:
        if notional < self.min_notional:
            msg = f"Order notional {notional} below minimum {self.min_notional}"
            raise PaperExecutionError(msg)

    @staticmethod
    def _limit_touched(
        side: OrderSide,
        limit_price: Decimal,
        candle_low: Decimal,
        candle_high: Decimal,
    ) -> bool:
        if side == "BUY":
            return candle_low <= limit_price
        return candle_high >= limit_price
