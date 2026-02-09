"""Tests for walk-forward evaluator."""

import pytest

from packages.research.walk_forward import WalkForwardEvaluator


def test_walk_forward_generates_multiple_folds() -> None:
    evaluator = WalkForwardEvaluator()
    equity = [1000 + i * 2 for i in range(120)]
    pnl = [1 if i % 2 == 0 else -0.5 for i in range(120)]
    exposure = [0.2 for _ in range(120)]

    result = evaluator.evaluate(
        equity_series=equity,
        trade_pnl_series=pnl,
        exposure_series=exposure,
        train_size=40,
        test_size=20,
        step=10,
    )

    assert len(result.folds) >= 2
    assert result.mean_composite_score != 0
    assert result.max_drawdown_worst_fold >= 0


def test_walk_forward_rejects_invalid_windows() -> None:
    evaluator = WalkForwardEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            equity_series=[1000, 1001, 1002],
            trade_pnl_series=[1, 1, 1],
            exposure_series=[0.1, 0.1, 0.1],
            train_size=0,
            test_size=1,
            step=1,
        )
