"""Tests for metrics calculator."""

from packages.core.metrics import MetricsCalculator


def test_metrics_calculation_core_values() -> None:
    calc = MetricsCalculator()
    metrics = calc.calculate(
        trade_pnls=[10, -5, 8, -2, 6],
        equity_curve=[1000, 1010, 1005, 1013, 1011, 1017],
        fees_paid=3.5,
        exposures_pct=[0.1, 0.2, 0.15, 0.22, 0.18, 0.16],
    )

    assert metrics.pnl == 17
    assert metrics.returns > 0
    assert metrics.max_drawdown >= 0
    assert 0 <= metrics.win_rate <= 1
    assert metrics.profit_factor > 1
    assert metrics.fees_paid == 3.5
    assert metrics.sharpe_ratio is not None


def test_metrics_empty_equity_curve() -> None:
    calc = MetricsCalculator()
    metrics = calc.calculate(
        trade_pnls=[],
        equity_curve=[],
        fees_paid=0.0,
        exposures_pct=[],
    )
    assert metrics.pnl == 0.0
    assert metrics.composite_score == 0.0
