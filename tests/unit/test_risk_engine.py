"""Tests for risk engine."""

from decimal import Decimal

from packages.core.risk.engine import RiskConfig, RiskEngine, RiskInput
from packages.core.strategies.base import Signal


def test_hold_signal_short_circuits() -> None:
    engine = RiskEngine()
    decision = engine.evaluate(
        Signal(side="HOLD", confidence=0.0, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("0"),
            current_exposure_notional=Decimal("0"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "HOLD"
    assert not decision.allowed
    assert decision.reason == "signal_hold"


def test_reject_when_daily_loss_exceeded() -> None:
    engine = RiskEngine(RiskConfig(max_daily_loss=Decimal("0.02")))
    decision = engine.evaluate(
        Signal(side="BUY", confidence=0.8, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("-250"),
            current_exposure_notional=Decimal("0"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "HOLD"
    assert decision.reason == "max_daily_loss_exceeded"


def test_hold_when_max_exposure_reached() -> None:
    engine = RiskEngine(RiskConfig(max_exposure=Decimal("0.25")))
    decision = engine.evaluate(
        Signal(side="BUY", confidence=0.9, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("0"),
            current_exposure_notional=Decimal("2500"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "HOLD"
    assert decision.reason == "max_exposure_reached"


def test_sell_not_blocked_by_max_exposure_limit() -> None:
    engine = RiskEngine(RiskConfig(max_exposure=Decimal("0.25")))
    decision = engine.evaluate(
        Signal(side="SELL", confidence=0.9, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("0"),
            current_exposure_notional=Decimal("2500"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "ALLOW"
    assert decision.allowed is True
    assert decision.reason == "risk_checks_passed"
    assert decision.quantity > 0


def test_allow_valid_signal_and_compute_size() -> None:
    engine = RiskEngine(
        RiskConfig(
            risk_per_trade=Decimal("0.01"),
            max_exposure=Decimal("0.25"),
            fee_bps=10,
            slippage_bps=5,
            lot_step_size=Decimal("0.00001"),
        )
    )
    decision = engine.evaluate(
        Signal(side="BUY", confidence=0.75, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("100"),
            current_exposure_notional=Decimal("500"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "ALLOW"
    assert decision.allowed
    assert decision.quantity > 0
    assert decision.notional >= Decimal("10")
    assert decision.estimated_fee > 0
    assert decision.estimated_slippage_cost > 0


def test_reject_below_min_notional() -> None:
    engine = RiskEngine(
        RiskConfig(
            risk_per_trade=Decimal("0.0001"),
            min_notional=Decimal("10"),
        )
    )
    decision = engine.evaluate(
        Signal(side="BUY", confidence=0.75, reason="test", indicators={}),
        RiskInput(
            equity=Decimal("10000"),
            daily_realized_pnl=Decimal("0"),
            current_exposure_notional=Decimal("0"),
            price=Decimal("50000"),
        ),
    )
    assert decision.action == "REJECT"
    assert decision.reason == "below_min_notional"
