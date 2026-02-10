"""AI advisor for generating proposal candidates."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.ai.llm_client import get_llm_advisor_client
from packages.core.config import get_settings
from packages.core.database.models import EquitySnapshot, Fill, MetricsSnapshot
from packages.core.strategies import registry

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
    """LLM-first advisor producing human-approval proposals with safety filters."""

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
        latest_metrics = await self._latest_metrics(session)
        if latest_metrics is None:
            return []

        settings = get_settings()
        llm_settings = settings.llm
        llm_proposals: list[AIProposal] = []
        if llm_settings.enabled:
            llm_proposals = await self._llm_proposals(session, latest_metrics)
            if llm_proposals and llm_settings.prefer_llm:
                return llm_proposals[: llm_settings.max_proposals]
            if not llm_proposals and not llm_settings.fallback_to_rules:
                return []

        proposals: list[AIProposal] = list(llm_proposals)

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

        if llm_settings.enabled:
            return proposals[: max(1, llm_settings.max_proposals)]
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

    async def _llm_proposals(
        self,
        session: AsyncSession,
        latest_metrics: MetricsSnapshot,
    ) -> list[AIProposal]:
        settings = get_settings()
        context = await self._build_llm_context(session, latest_metrics)
        try:
            raw = await self._generate_llm_raw_proposals(context=context)
        except Exception:
            return []
        return self._sanitize_llm_proposals(raw, settings.approval.timeout_hours)

    async def _build_llm_context(
        self,
        session: AsyncSession,
        latest_metrics: MetricsSnapshot,
    ) -> dict[str, Any]:
        settings = get_settings()
        recent_metrics_result = await session.execute(
            select(MetricsSnapshot)
            .where(MetricsSnapshot.is_paper.is_(True))
            .order_by(MetricsSnapshot.snapshot_at.desc())
            .limit(8)
        )
        recent_metrics = list(recent_metrics_result.scalars().all())

        fills_24h_result = await session.execute(
            select(func.count(Fill.id), func.avg(Fill.slippage_bps))
            .where(
                Fill.is_paper.is_(True),
                Fill.filled_at >= datetime.now(UTC) - timedelta(hours=24),
            )
        )
        fills_24h_count, avg_slippage_24h = fills_24h_result.one()

        equity_result = await session.execute(
            select(EquitySnapshot.equity)
            .where(EquitySnapshot.is_paper.is_(True))
            .order_by(EquitySnapshot.snapshot_at.desc())
            .limit(60)
        )
        equity_values = [float(value) for value in reversed(list(equity_result.scalars().all()))]
        equity_return = 0.0
        if len(equity_values) >= 2 and equity_values[0] > 0:
            equity_return = (equity_values[-1] - equity_values[0]) / equity_values[0]

        return {
            "symbol": settings.trading.pair,
            "timeframes": settings.trading.timeframe_list,
            "active_strategy": settings.trading.active_strategy,
            "supported_strategies": registry.list_names(),
            "risk_settings": {
                "per_trade": settings.risk.per_trade,
                "max_daily_loss": settings.risk.max_daily_loss,
                "max_exposure": settings.risk.max_exposure,
                "fee_bps": settings.risk.fee_bps,
                "slippage_bps": settings.risk.slippage_bps,
            },
            "trading_settings": {
                "grid_levels": settings.trading.grid_levels,
                "grid_spacing_mode": settings.trading.grid_spacing_mode,
                "grid_min_spacing_bps": settings.trading.grid_min_spacing_bps,
                "grid_max_spacing_bps": settings.trading.grid_max_spacing_bps,
                "grid_volatility_blend": settings.trading.grid_volatility_blend,
                "grid_trend_tilt": settings.trading.grid_trend_tilt,
                "grid_take_profit_buffer": settings.trading.grid_take_profit_buffer,
                "grid_stop_loss_buffer": settings.trading.grid_stop_loss_buffer,
                "grid_recenter_mode": settings.trading.grid_recenter_mode,
            },
            "latest_metrics": {
                "strategy_name": latest_metrics.strategy_name,
                "total_trades": latest_metrics.total_trades,
                "winning_trades": latest_metrics.winning_trades,
                "losing_trades": latest_metrics.losing_trades,
                "total_pnl": float(latest_metrics.total_pnl),
                "total_fees": float(latest_metrics.total_fees),
                "max_drawdown": latest_metrics.max_drawdown,
                "win_rate": latest_metrics.win_rate,
                "sharpe_ratio": latest_metrics.sharpe_ratio,
                "sortino_ratio": latest_metrics.sortino_ratio,
                "profit_factor": latest_metrics.profit_factor,
            },
            "recent_metrics": [
                {
                    "snapshot_at": metric.snapshot_at.isoformat(),
                    "total_trades": metric.total_trades,
                    "total_pnl": float(metric.total_pnl),
                    "max_drawdown": metric.max_drawdown,
                    "win_rate": metric.win_rate,
                    "sharpe_ratio": metric.sharpe_ratio,
                }
                for metric in recent_metrics
            ],
            "recent_activity": {
                "fills_24h_count": int(fills_24h_count or 0),
                "avg_slippage_bps_24h": float(avg_slippage_24h or 0.0),
                "equity_points": len(equity_values),
                "equity_return_window": equity_return,
            },
        }

    async def _generate_llm_raw_proposals(self, *, context: dict[str, Any]) -> list[dict[str, Any]]:
        return await get_llm_advisor_client().generate_proposals(context)

    def llm_status(self) -> dict[str, Any]:
        status = get_llm_advisor_client().status()
        return {
            "enabled": status.enabled,
            "provider": status.provider,
            "model": status.model,
            "configured": status.configured,
            "base_url": status.base_url,
            "prefer_llm": status.prefer_llm,
            "fallback_to_rules": status.fallback_to_rules,
            "last_error": status.last_error,
        }

    async def test_llm_connection(self) -> dict[str, Any]:
        return await get_llm_advisor_client().test_connection()

    def _sanitize_llm_proposals(
        self,
        raw: list[dict[str, Any]],
        approval_timeout_hours: int,
    ) -> list[AIProposal]:
        settings = get_settings()
        max_items = max(1, settings.llm.max_proposals)
        min_conf = settings.llm.min_confidence
        allowed_types = {
            "risk_tuning",
            "execution_tuning",
            "grid_tuning",
            "strategy_switch",
            "anomaly_alert",
            "drl_risk_tuning",
        }
        sanitized: list[AIProposal] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            proposal_type = str(item.get("proposal_type", "")).strip().lower()
            if proposal_type not in allowed_types:
                continue
            diff = self._sanitize_diff(item.get("diff", {}))
            if not diff:
                continue
            confidence = _safe_confidence(item.get("confidence", 0.0))
            if confidence < min_conf:
                continue
            ttl_raw = item.get("ttl_hours", self.ttl_hours)
            ttl = int(ttl_raw) if isinstance(ttl_raw, int | float) else self.ttl_hours
            ttl = max(1, min(ttl, approval_timeout_hours))
            title = str(item.get("title", "")).strip()[:180]
            description = str(item.get("description", "")).strip()[:800]
            expected_impact = str(item.get("expected_impact", "")).strip()[:400]
            evidence = item.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {"raw_evidence": str(evidence)[:500]}
            if not title or not description:
                continue
            sanitized.append(
                AIProposal(
                    title=title,
                    proposal_type=proposal_type,
                    description=description,
                    diff=diff,
                    expected_impact=expected_impact or "LLM-suggested config adjustment.",
                    evidence=evidence,
                    confidence=confidence,
                    ttl_hours=ttl,
                )
            )
            if len(sanitized) >= max_items:
                break
        return sanitized

    def _sanitize_diff(self, diff: Any) -> dict[str, Any]:
        if not isinstance(diff, dict):
            return {}
        allowed: dict[str, set[str]] = {
            "risk": {"per_trade", "max_daily_loss", "max_exposure", "fee_bps", "slippage_bps"},
            "trading": {
                "active_strategy",
                "grid_lookback_1h",
                "grid_atr_period_1h",
                "grid_levels",
                "grid_spacing_mode",
                "grid_min_spacing_bps",
                "grid_max_spacing_bps",
                "grid_trend_tilt",
                "grid_volatility_blend",
                "grid_take_profit_buffer",
                "grid_stop_loss_buffer",
                "grid_cooldown_seconds",
                "grid_bootstrap_fraction",
                "grid_min_net_profit_bps",
                "grid_recenter_mode",
                "advisor_interval_cycles",
                "stop_loss_global_equity_pct",
                "stop_loss_max_drawdown_pct",
            },
        }
        valid_strategies = set(registry.list_names())
        cleaned: dict[str, dict[str, Any]] = {}
        for section, values in diff.items():
            if section not in allowed or not isinstance(values, dict):
                continue
            section_clean: dict[str, Any] = {}
            for key, value in values.items():
                if key not in allowed[section]:
                    continue
                if section == "trading" and key == "active_strategy":
                    if not isinstance(value, str) or value not in valid_strategies:
                        continue
                    section_clean[key] = value
                    continue
                if isinstance(value, bool | int | str):
                    section_clean[key] = value
                    continue
                if isinstance(value, float) and isfinite(value):
                    section_clean[key] = value
            if section_clean:
                cleaned[section] = section_clean
        return cleaned

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


def _safe_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))
