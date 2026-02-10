"""AI advisor for generating proposal candidates."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.database.models import EquitySnapshot, Fill, MetricsSnapshot

if TYPE_CHECKING:
    from packages.core.ai.drl_optimizer import DRLOptimizer


@dataclass(slots=True, frozen=True)
class AIProposal:
    """AI proposal payload."""

    title: str
    proposal_type: str
    description: str
    diff: dict[str, Any]
    expected_impact: str
    evidence: dict[str, Any]
    confidence: float
    ttl_hours: int = 2


class AIAdvisor:
    """Rule-guided advisor producing human-approval proposals."""

    def __init__(self, *, ttl_hours: int = 2) -> None:
        self.ttl_hours = ttl_hours
        self.optimizer: DRLOptimizer | None = None

    async def generate_proposals(self, session: AsyncSession) -> list[AIProposal]:
        """
        Generate proposal candidates.

        Production-safe behavior:
        - Suggest only parameter changes.
        - Never execute changes directly.
        """
        proposals: list[AIProposal] = []
        latest_metrics = await self._latest_metrics(session)
        if latest_metrics is None:
            return proposals

        if latest_metrics.max_drawdown >= 0.08:
            proposals.append(
                AIProposal(
                    title="Reduce risk_per_trade due to elevated drawdown",
                    proposal_type="risk_tuning",
                    description=(
                        "Recent drawdown breached 8%. Suggest lowering risk per trade to contain volatility."
                    ),
                    diff={"risk": {"per_trade": 0.0035}},
                    expected_impact="Lower downside volatility and improve drawdown profile.",
                    evidence={
                        "max_drawdown": latest_metrics.max_drawdown,
                        "total_trades": latest_metrics.total_trades,
                        "total_pnl": float(latest_metrics.total_pnl),
                    },
                    confidence=0.82,
                    ttl_hours=self.ttl_hours,
                )
            )

        slippage_spike = await self._recent_slippage_spike(session)
        if slippage_spike is not None:
            proposals.append(
                AIProposal(
                    title="Increase slippage_bps safety margin",
                    proposal_type="execution_tuning",
                    description="Observed sustained slippage above configured expectation.",
                    diff={"risk": {"slippage_bps": max(5, int(slippage_spike + 2))}},
                    expected_impact="Improve fill realism and reduce optimistic paper fill assumptions.",
                    evidence={"avg_slippage_bps_24h": slippage_spike},
                    confidence=0.73,
                    ttl_hours=self.ttl_hours,
                )
            )

        grid_tuning = self._grid_tuning_proposal(latest_metrics)
        if grid_tuning is not None:
            proposals.append(grid_tuning)

        drl = await self._drl_proposal(session)
        if drl is not None:
            proposals.append(
                AIProposal(
                    title=drl.title,
                    proposal_type="drl_risk_tuning",
                    description="PPO optimizer suggests dynamic risk adjustment.",
                    diff=drl.diff,
                    expected_impact=drl.expected_impact,
                    evidence=drl.evidence,
                    confidence=drl.confidence,
                    ttl_hours=self.ttl_hours,
                )
            )

        return proposals

    async def _latest_metrics(self, session: AsyncSession) -> MetricsSnapshot | None:
        result = await session.execute(
            select(MetricsSnapshot)
            .where(MetricsSnapshot.is_paper.is_(True))
            .order_by(MetricsSnapshot.snapshot_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def _recent_slippage_spike(self, session: AsyncSession) -> float | None:
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        result = await session.execute(
            select(func.avg(Fill.slippage_bps))
            .where(Fill.is_paper.is_(True), Fill.filled_at >= cutoff, Fill.slippage_bps.is_not(None))
        )
        avg_slippage = result.scalar_one_or_none()
        if avg_slippage is None:
            return None
        slippage = float(avg_slippage)
        return slippage if slippage >= 5.0 else None

    def _grid_tuning_proposal(self, latest_metrics: MetricsSnapshot) -> AIProposal | None:
        if latest_metrics.strategy_name != "smart_grid_ai":
            return None

        win_rate = latest_metrics.win_rate if latest_metrics.win_rate is not None else 0.0
        total_trades = latest_metrics.total_trades
        max_drawdown = latest_metrics.max_drawdown

        if total_trades < 20:
            return None

        if max_drawdown >= 0.08:
            return AIProposal(
                title="Grid risk-off tuning due to elevated drawdown",
                proposal_type="grid_tuning",
                description=(
                    "Drawdown is elevated for smart_grid_ai. Suggest widening spacing "
                    "and reducing active grid levels to lower churn."
                ),
                diff={
                    "trading": {
                        "grid_min_spacing_bps": 35,
                        "grid_max_spacing_bps": 280,
                        "grid_levels": 5,
                    }
                },
                expected_impact="Lower turnover and reduce drawdown pressure in volatile phases.",
                evidence={
                    "strategy_name": latest_metrics.strategy_name,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                },
                confidence=0.78,
                ttl_hours=self.ttl_hours,
            )

        if win_rate < 0.42:
            return AIProposal(
                title="Grid spacing increase for low win-rate conditions",
                proposal_type="grid_tuning",
                description=(
                    "Win rate dropped below threshold. Suggest wider spacing to avoid overtrading."
                ),
                diff={
                    "trading": {
                        "grid_min_spacing_bps": 32,
                        "grid_max_spacing_bps": 260,
                        "grid_volatility_blend": 0.85,
                    }
                },
                expected_impact="Fewer low-quality grid flips and improved net expectancy after fees.",
                evidence={
                    "strategy_name": latest_metrics.strategy_name,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                },
                confidence=0.71,
                ttl_hours=self.ttl_hours,
            )

        if win_rate > 0.60 and max_drawdown < 0.06:
            return AIProposal(
                title="Grid efficiency tuning in stable regime",
                proposal_type="grid_tuning",
                description=(
                    "Performance is stable. Suggest modestly tighter grid spacing to capture "
                    "more mean-reverting moves."
                ),
                diff={
                    "trading": {
                        "grid_min_spacing_bps": 22,
                        "grid_max_spacing_bps": 200,
                        "grid_levels": 7,
                    }
                },
                expected_impact="Higher opportunity capture while maintaining acceptable drawdown.",
                evidence={
                    "strategy_name": latest_metrics.strategy_name,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                },
                confidence=0.66,
                ttl_hours=self.ttl_hours,
            )

        return None

    async def _drl_proposal(self, session: AsyncSession) -> Any | None:
        from packages.core.ai.drl_optimizer import DRLOptimizer

        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.is_paper.is_(True))
            .order_by(EquitySnapshot.snapshot_at.asc())
            .limit(400)
        )
        snapshots = list(result.scalars().all())
        if len(snapshots) < 40:
            return None

        equity = [float(s.equity) for s in snapshots]
        rewards: list[float] = []
        for i in range(1, len(equity)):
            prev = equity[i - 1]
            curr = equity[i]
            rewards.append((curr - prev) / prev if prev else 0.0)
        exposures = [0.2 for _ in equity]
        try:
            if self.optimizer is None:
                self.optimizer = DRLOptimizer()
            self.optimizer.train(rewards, timesteps=512)
            if not self.optimizer.walk_forward_validate(
                equity_series=equity,
                pnl_series=rewards + [0.0],
                exposure_series=exposures,
            ):
                return None
            return self.optimizer.propose(
                base_risk_per_trade=0.005,
                reward_series=rewards,
                equity_curve=equity,
            )
        except Exception:
            # Optimizer should never break the main advisor path.
            return None


_advisor: AIAdvisor | None = None
_advisor_lock = threading.Lock()


def get_ai_advisor() -> AIAdvisor:
    """Get or create singleton advisor."""
    global _advisor
    if _advisor is None:
        with _advisor_lock:
            if _advisor is None:
                _advisor = AIAdvisor()
    return _advisor
