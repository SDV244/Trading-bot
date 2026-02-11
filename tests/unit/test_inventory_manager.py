"""Tests for inventory manager decisions."""

from decimal import Decimal

from packages.core.inventory_manager import InventoryManager


def test_profit_level_triggers_partial_reduce() -> None:
    manager = InventoryManager()
    decision = manager.evaluate(
        current_price=Decimal("103"),
        avg_entry=Decimal("100"),
        current_quantity=Decimal("1.0"),
        hours_in_position=2,
    )
    assert decision.action in {"REDUCE_50", "REDUCE_ALL"}
    assert decision.urgency >= 4
    assert decision.reduce_fraction >= 0.5


def test_time_stop_triggers_exit() -> None:
    manager = InventoryManager(time_stop_hours=24, min_profit_for_time_stop=0.01)
    decision = manager.evaluate(
        current_price=Decimal("100.2"),
        avg_entry=Decimal("100"),
        current_quantity=Decimal("0.7"),
        hours_in_position=30,
    )
    assert decision.action == "REDUCE_ALL"
    assert "time_stop" in decision.reason


def test_volatility_aware_evaluation_and_stats() -> None:
    manager = InventoryManager()
    decision = manager.evaluate_with_volatility(
        current_price=Decimal("101.8"),
        avg_entry=Decimal("100"),
        current_quantity=Decimal("1.0"),
        hours_in_position=6,
        volatility_percentile=0.9,
    )
    stats = manager.get_statistics()
    assert decision.action in {"HOLD", "REDUCE_25", "REDUCE_50", "REDUCE_ALL"}
    assert stats["decisions_total"] >= 1
