"""Risk engine for signal-level trade gating and sizing."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from packages.core.strategies.base import Signal

RiskAction = Literal["ALLOW", "HOLD", "REJECT"]


@dataclass(slots=True, frozen=True)
class RiskConfig:
    """Risk engine parameters."""

    risk_per_trade: Decimal = Decimal("0.005")
    max_daily_loss: Decimal = Decimal("0.02")
    max_exposure: Decimal = Decimal("0.25")
    fee_bps: int = 10
    slippage_bps: int = 5
    min_notional: Decimal = Decimal("10")
    lot_step_size: Decimal = Decimal("0.00001")


@dataclass(slots=True, frozen=True)
class RiskInput:
    """Risk evaluation inputs."""

    equity: Decimal
    daily_realized_pnl: Decimal
    current_exposure_notional: Decimal
    price: Decimal


@dataclass(slots=True, frozen=True)
class RiskDecision:
    """Risk decision output."""

    action: RiskAction
    allowed: bool
    quantity: Decimal
    notional: Decimal
    estimated_fee: Decimal
    estimated_slippage_cost: Decimal
    reason: str


class RiskEngine:
    """Evaluate whether a strategy signal can be executed safely."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def evaluate(self, signal: Signal, risk_input: RiskInput) -> RiskDecision:
        """Evaluate signal and compute safe order sizing."""
        if signal.side == "HOLD":
            return self._hold("signal_hold")

        if risk_input.equity <= 0:
            return self._reject("invalid_equity")
        if risk_input.price <= 0:
            return self._reject("invalid_price")

        if self._daily_loss_exceeded(risk_input):
            return self._hold("max_daily_loss_exceeded")

        is_sell = signal.side == "SELL"
        max_notional = self.config.max_exposure * risk_input.equity
        available_notional = max_notional - risk_input.current_exposure_notional
        if not is_sell and available_notional <= 0:
            return self._hold("max_exposure_reached")
        if is_sell and risk_input.current_exposure_notional <= 0:
            return self._hold("no_exposure_to_reduce")

        risk_budget_notional = self.config.risk_per_trade * risk_input.equity
        if is_sell:
            # SELL orders reduce exposure and should not be blocked by max_exposure
            # guard. Cap size by current exposure when known.
            trade_notional = max(risk_budget_notional, self.config.min_notional)
            trade_notional = min(
                trade_notional,
                max(risk_input.current_exposure_notional, Decimal("0")),
            )
        else:
            trade_notional = min(risk_budget_notional, available_notional)
        if trade_notional < self.config.min_notional:
            return self._reject("below_min_notional")

        quantity = (trade_notional / risk_input.price).quantize(
            self.config.lot_step_size,
            rounding=ROUND_DOWN,
        )
        if quantity <= 0:
            return self._reject("quantity_rounded_to_zero")

        final_notional = quantity * risk_input.price
        if final_notional < self.config.min_notional:
            return self._reject("final_notional_below_minimum")

        fee = (final_notional * Decimal(self.config.fee_bps) / Decimal("10000")).quantize(
            Decimal("0.00000001")
        )
        slippage = (
            final_notional * Decimal(self.config.slippage_bps) / Decimal("10000")
        ).quantize(Decimal("0.00000001"))

        return RiskDecision(
            action="ALLOW",
            allowed=True,
            quantity=quantity,
            notional=final_notional,
            estimated_fee=fee,
            estimated_slippage_cost=slippage,
            reason="risk_checks_passed",
        )

    def _daily_loss_exceeded(self, risk_input: RiskInput) -> bool:
        if risk_input.daily_realized_pnl >= 0:
            return False
        daily_loss = abs(risk_input.daily_realized_pnl)
        return (daily_loss / risk_input.equity) >= self.config.max_daily_loss

    @staticmethod
    def _hold(reason: str) -> RiskDecision:
        return RiskDecision(
            action="HOLD",
            allowed=False,
            quantity=Decimal("0"),
            notional=Decimal("0"),
            estimated_fee=Decimal("0"),
            estimated_slippage_cost=Decimal("0"),
            reason=reason,
        )

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(
            action="REJECT",
            allowed=False,
            quantity=Decimal("0"),
            notional=Decimal("0"),
            estimated_fee=Decimal("0"),
            estimated_slippage_cost=Decimal("0"),
            reason=reason,
        )
