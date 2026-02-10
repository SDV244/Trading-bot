"""Global stop-loss guard for catastrophic risk containment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class StopLossType(str, Enum):
    """Stop-loss trigger categories."""

    GLOBAL_EQUITY = "global_equity"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass(slots=True, frozen=True)
class GlobalStopLossConfig:
    """Configurable global stop-loss limits."""

    enabled: bool = True
    global_equity_pct: Decimal = Decimal("0.15")
    max_drawdown_pct: Decimal = Decimal("0.20")
    auto_close_positions: bool = True


@dataclass(slots=True, frozen=True)
class StopLossDecision:
    """Decision returned by global stop-loss check."""

    triggered: bool
    trigger_type: StopLossType | None
    reason: str
    current_equity: Decimal
    starting_equity: Decimal
    peak_equity: Decimal
    loss_pct: Decimal
    drawdown_pct: Decimal


class GlobalStopLossGuard:
    """Evaluate hard global loss limits for each trading cycle."""

    def __init__(self, config: GlobalStopLossConfig | None = None) -> None:
        self.config = config or GlobalStopLossConfig()

    def evaluate(
        self,
        *,
        current_equity: Decimal,
        starting_equity: Decimal,
        peak_equity: Decimal,
    ) -> StopLossDecision:
        if not self.config.enabled:
            return StopLossDecision(
                triggered=False,
                trigger_type=None,
                reason="stop_loss_disabled",
                current_equity=current_equity,
                starting_equity=starting_equity,
                peak_equity=peak_equity,
                loss_pct=Decimal("0"),
                drawdown_pct=Decimal("0"),
            )
        if starting_equity <= 0:
            return StopLossDecision(
                triggered=False,
                trigger_type=None,
                reason="invalid_starting_equity",
                current_equity=current_equity,
                starting_equity=starting_equity,
                peak_equity=peak_equity,
                loss_pct=Decimal("0"),
                drawdown_pct=Decimal("0"),
            )

        effective_peak = max(peak_equity, starting_equity, current_equity)
        loss_pct = max(Decimal("0"), (starting_equity - current_equity) / starting_equity)
        drawdown_pct = Decimal("0")
        if effective_peak > 0:
            drawdown_pct = max(Decimal("0"), (effective_peak - current_equity) / effective_peak)

        if loss_pct >= self.config.global_equity_pct:
            return StopLossDecision(
                triggered=True,
                trigger_type=StopLossType.GLOBAL_EQUITY,
                reason="stop_loss_global_equity_triggered",
                current_equity=current_equity,
                starting_equity=starting_equity,
                peak_equity=effective_peak,
                loss_pct=loss_pct,
                drawdown_pct=drawdown_pct,
            )

        if drawdown_pct >= self.config.max_drawdown_pct:
            return StopLossDecision(
                triggered=True,
                trigger_type=StopLossType.MAX_DRAWDOWN,
                reason="stop_loss_drawdown_triggered",
                current_equity=current_equity,
                starting_equity=starting_equity,
                peak_equity=effective_peak,
                loss_pct=loss_pct,
                drawdown_pct=drawdown_pct,
            )

        return StopLossDecision(
            triggered=False,
            trigger_type=None,
            reason="stop_loss_checks_passed",
            current_equity=current_equity,
            starting_equity=starting_equity,
            peak_equity=effective_peak,
            loss_pct=loss_pct,
            drawdown_pct=drawdown_pct,
        )
