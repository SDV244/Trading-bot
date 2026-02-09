"""AI advisor for generating proposal candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.ai.drl_optimizer import DRLOptimizer
from packages.core.database.models import EquitySnapshot, Fill, MetricsSnapshot


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
        self.optimizer = DRLOptimizer()

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

    async def _drl_proposal(self, session: AsyncSession) -> Any | None:
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


def get_ai_advisor() -> AIAdvisor:
    """Get or create singleton advisor."""
    global _advisor
    if _advisor is None:
        _advisor = AIAdvisor()
    return _advisor
