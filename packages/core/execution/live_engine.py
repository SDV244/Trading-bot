"""Live trading engine with strict go-live checks and adapter execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any
from uuid import uuid4

from packages.adapters.binance_live import get_binance_live_adapter
from packages.core.config import get_settings
from packages.core.execution.paper_engine import OrderRequest


class LiveEngineError(ValueError):
    """Raised when live execution preconditions fail."""


@dataclass(slots=True, frozen=True)
class LiveSafetyChecklist:
    """Mandatory checklist before live execution."""

    ui_confirmed: bool
    reauthenticated: bool
    safety_acknowledged: bool


@dataclass(slots=True, frozen=True)
class LiveOrderResult:
    """Result envelope for live execution."""

    accepted: bool
    reason: str
    order_id: str | None = None
    client_order_id: str | None = None
    quantity: Decimal | None = None
    price: Decimal | None = None
    raw: dict[str, Any] | None = None


class LiveEngine:
    """
    Live engine behind an explicit feature flag.

    This module enforces safety gates; exchange order placement should be
    integrated in a dedicated live adapter with hardened retry/idempotency.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.trading.live_mode and (
            not self.settings.binance.api_key or not self.settings.binance.api_secret
        ):
            raise LiveEngineError("TRADING_LIVE_MODE=true requires BINANCE_API_KEY and BINANCE_API_SECRET")

    async def execute_market_order(
        self,
        order: OrderRequest,
        checklist: LiveSafetyChecklist,
        client_order_id: str | None = None,
    ) -> LiveOrderResult:
        self._validate_preconditions(order, checklist)
        normalized_quantity = await self._validate_symbol_filters(order.symbol, order.quantity)
        effective_client_order_id = client_order_id or self._generate_client_order_id()
        adapter = get_binance_live_adapter()
        try:
            response = await adapter.place_market_order(
                symbol=order.symbol,
                side=order.side,
                quantity=normalized_quantity,
                new_client_order_id=effective_client_order_id,
            )
        except Exception:
            recovered = await adapter.query_order(
                symbol=order.symbol,
                client_order_id=effective_client_order_id,
            )
            if recovered is None:
                raise
            response = recovered
        status = str(response.get("status", "")).upper()
        executed_qty = Decimal(str(response.get("executedQty", "0")))
        cumm_quote_qty = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        avg_price = (cumm_quote_qty / executed_qty) if executed_qty > 0 else None
        accepted = status in {"FILLED", "PARTIALLY_FILLED", "NEW"}

        return LiveOrderResult(
            accepted=accepted,
            reason=f"exchange_status_{status or 'UNKNOWN'}",
            order_id=str(response.get("orderId")) if response.get("orderId") is not None else None,
            client_order_id=(
                str(response.get("clientOrderId")) if response.get("clientOrderId") else effective_client_order_id
            ),
            quantity=executed_qty if executed_qty > 0 else normalized_quantity,
            price=avg_price,
            raw=response,
        )

    def _validate_preconditions(self, order: OrderRequest, checklist: LiveSafetyChecklist) -> None:
        if not self.settings.trading.live_mode:
            raise LiveEngineError("live_mode is disabled; cannot execute live orders")
        if order.symbol != "BTCUSDT":
            raise LiveEngineError("Only BTCUSDT is supported in v1")
        if order.order_type != "MARKET":
            raise LiveEngineError("Only MARKET orders are supported in current live gate")
        if order.quantity <= 0:
            raise LiveEngineError("quantity must be positive")
        if not checklist.ui_confirmed:
            raise LiveEngineError("UI confirmation is required")
        if not checklist.reauthenticated:
            raise LiveEngineError("Re-authentication is required")
        if not checklist.safety_acknowledged:
            raise LiveEngineError("Safety checklist acknowledgement is required")

    async def _validate_symbol_filters(self, symbol: str, quantity: Decimal) -> Decimal:
        info = await get_binance_live_adapter().get_exchange_filters(symbol)
        filters = {f.get("filterType"): f for f in info.get("filters", [])}

        normalized_quantity = quantity
        lot = filters.get("LOT_SIZE")
        if lot is not None:
            min_qty = Decimal(str(lot.get("minQty", "0")))
            step_size = Decimal(str(lot.get("stepSize", "0.00000001")))
            if step_size > 0:
                step_units = (quantity / step_size).to_integral_value(rounding=ROUND_DOWN)
                normalized_quantity = (step_units * step_size).quantize(step_size, rounding=ROUND_DOWN)
            if normalized_quantity < min_qty:
                raise LiveEngineError(f"quantity below minQty ({min_qty})")
            if normalized_quantity <= 0:
                raise LiveEngineError("quantity rounds down to zero after LOT_SIZE precision normalization")

        # MIN_NOTIONAL is exchange-enforced and depends on execution price.
        # We keep authoritative validation at order placement response.
        return normalized_quantity

    @staticmethod
    def _generate_client_order_id() -> str:
        return f"live_{uuid4().hex[:24]}"
