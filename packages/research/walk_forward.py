"""Walk-forward strategy evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from packages.core.metrics import MetricsCalculator, PerformanceMetrics

StrategyEvaluator = Callable[[list[float]], list[float]]


@dataclass(slots=True, frozen=True)
class WalkForwardFold:
    """Single fold evaluation result."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: PerformanceMetrics


@dataclass(slots=True, frozen=True)
class WalkForwardResult:
    """Walk-forward aggregate result."""

    folds: list[WalkForwardFold]
    mean_composite_score: float
    max_drawdown_worst_fold: float
    stable: bool


class WalkForwardEvaluator:
    """Rolling train/test walk-forward evaluator."""

    def __init__(self) -> None:
        self.metrics = MetricsCalculator()

    def evaluate(
        self,
        equity_series: list[float],
        trade_pnl_series: list[float],
        exposure_series: list[float],
        *,
        train_size: int,
        test_size: int,
        step: int,
    ) -> WalkForwardResult:
        if train_size <= 0 or test_size <= 0 or step <= 0:
            raise ValueError("train_size, test_size and step must be positive")
        if len(equity_series) < (train_size + test_size):
            raise ValueError("Not enough data for one fold")

        folds: list[WalkForwardFold] = []
        start = 0
        while (start + train_size + test_size) <= len(equity_series):
            train_start = start
            train_end = start + train_size
            test_start = train_end
            test_end = test_start + test_size

            test_equity = equity_series[test_start:test_end]
            test_trades = trade_pnl_series[test_start:test_end]
            test_exposure = exposure_series[test_start:test_end]
            metrics = self.metrics.calculate(
                trade_pnls=test_trades,
                equity_curve=test_equity,
                fees_paid=0.0,
                exposures_pct=test_exposure,
            )
            folds.append(
                WalkForwardFold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    metrics=metrics,
                )
            )
            start += step

        if not folds:
            raise ValueError("No folds generated; adjust window sizes")

        mean_score = sum(f.metrics.composite_score for f in folds) / len(folds)
        worst_drawdown = max(f.metrics.max_drawdown for f in folds)
        stable = worst_drawdown <= 0.10 and all(f.metrics.composite_score > -1.0 for f in folds)
        return WalkForwardResult(
            folds=folds,
            mean_composite_score=mean_score,
            max_drawdown_worst_fold=worst_drawdown,
            stable=stable,
        )
