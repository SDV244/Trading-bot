"""Trading metrics calculator."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(slots=True, frozen=True)
class PerformanceMetrics:
    """Computed performance metrics."""

    pnl: float
    returns: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    fees_paid: float
    avg_exposure_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    stability_penalty: float
    composite_score: float


class MetricsCalculator:
    """Calculate portfolio and strategy performance metrics."""

    def calculate(
        self,
        *,
        trade_pnls: list[float],
        equity_curve: list[float],
        fees_paid: float,
        exposures_pct: list[float],
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        if not equity_curve:
            return PerformanceMetrics(
                pnl=0.0,
                returns=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                fees_paid=fees_paid,
                avg_exposure_pct=0.0,
                sharpe_ratio=None,
                sortino_ratio=None,
                stability_penalty=0.0,
                composite_score=0.0,
            )

        pnl = sum(trade_pnls)
        initial = equity_curve[0]
        final = equity_curve[-1]
        returns = 0.0 if initial == 0 else (final - initial) / initial
        drawdown = _max_drawdown(equity_curve)
        win_rate = _win_rate(trade_pnls)
        profit_factor = _profit_factor(trade_pnls)
        avg_exposure = sum(exposures_pct) / len(exposures_pct) if exposures_pct else 0.0

        periodic_returns = _periodic_returns(equity_curve)
        sharpe = _sharpe(periodic_returns, risk_free_rate)
        sortino = _sortino(periodic_returns, risk_free_rate)
        stability_penalty = _stability_penalty(periodic_returns)

        # Sortino-first ranking with drawdown penalty and stability adjustment.
        sortino_component = sortino if sortino is not None else 0.0
        sharpe_component = sharpe if sharpe is not None else 0.0
        composite = (
            (sortino_component * 0.55)
            + (sharpe_component * 0.25)
            + (returns * 0.20)
            - (drawdown * 0.35)
            - stability_penalty
        )

        return PerformanceMetrics(
            pnl=pnl,
            returns=returns,
            max_drawdown=drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            fees_paid=fees_paid,
            avg_exposure_pct=avg_exposure,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            stability_penalty=stability_penalty,
            composite_score=composite,
        )


def _periodic_returns(equity_curve: list[float]) -> list[float]:
    if len(equity_curve) < 2:
        return []
    out: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        curr = equity_curve[i]
        if prev == 0:
            out.append(0.0)
        else:
            out.append((curr - prev) / prev)
    return out


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)


def _profit_factor(trade_pnls: list[float]) -> float:
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _sharpe(returns: list[float], risk_free_rate: float) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    adjusted = mean - risk_free_rate
    variance = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    std = sqrt(variance)
    if std == 0:
        return None
    return (adjusted / std) * sqrt(252)


def _sortino(returns: list[float], risk_free_rate: float) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    adjusted = mean - risk_free_rate
    downside = [min(0.0, x) for x in returns]
    downside_variance = sum(x**2 for x in downside) / len(downside)
    downside_dev = sqrt(downside_variance)
    if downside_dev == 0:
        return None
    return (adjusted / downside_dev) * sqrt(252)


def _stability_penalty(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    return sqrt(variance) * 0.1
