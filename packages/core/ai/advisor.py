"""AI advisor for generating proposal candidates."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.adapters.binance_spot import get_binance_adapter
from packages.core.ai.llm_client import get_llm_advisor_client
from packages.core.ai.multi_agent import MultiAgentCoordinator
from packages.core.alternative_data import get_alternative_data_aggregator
from packages.core.config import get_settings
from packages.core.database.models import (
    Approval,
    ApprovalStatus,
    Candle,
    EquitySnapshot,
    EventLog,
    Fill,
    MetricsSnapshot,
    Order,
)
from packages.core.market_regime import MarketRegimeDetector
from packages.core.order_book import OrderBookAnalyzer
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
        rule_based: list[AIProposal] = []

        if latest_metrics.max_drawdown >= 0.08:
            rule_based.append(
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
            rule_based.append(
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
            rule_based.append(grid_tuning)

        strategy_recommendations = await self._strategy_improvement_proposals(
            session=session,
            latest_metrics=latest_metrics,
        )
        rule_based.extend(strategy_recommendations)

        drl = await self._drl_proposal(session)
        if drl is not None:
            rule_based.append(
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

        llm_proposals: list[AIProposal] = []
        if llm_settings.enabled:
            if settings.multiagent.enabled:
                llm_proposals = await self._multi_agent_proposals(session, latest_metrics)
            if not llm_proposals:
                llm_proposals = await self._llm_proposals(session, latest_metrics)
            if not llm_proposals and not llm_settings.fallback_to_rules:
                return []

        if llm_settings.enabled and llm_settings.prefer_llm:
            proposals = [*llm_proposals, *rule_based]
        else:
            proposals = [*rule_based, *llm_proposals]
        proposals = self._dedupe_proposals(proposals)
        proposals = await self._prune_recent_duplicates(session, proposals)

        if llm_settings.enabled:
            max_items = max(1, min(llm_settings.max_proposals, settings.multiagent.max_proposals))
            return proposals[:max_items]
        return proposals[: max(1, settings.multiagent.max_proposals)]

    async def analyze_strategy(self, session: AsyncSession) -> dict[str, Any]:
        """Return strategy diagnostics and AI recommendations for improvement."""
        settings = get_settings()
        latest_metrics = await self._latest_metrics(session)
        if latest_metrics is None:
            return {
                "generated_at": datetime.now(UTC).isoformat(),
                "active_strategy": settings.trading.active_strategy,
                "symbol": settings.trading.pair,
                "latest_metrics": None,
                "strategy_insights": {},
                "hold_diagnostics": {},
                "regime_analysis": {},
                "recommendations": [],
            }

        strategy_insights = await self._analyze_strategy_insights(session)
        hold_diagnostics = await self._strategy_hold_diagnostics(session)
        regime_analysis = await self._get_regime_context(session, symbol=settings.trading.pair)
        recommendations = await self._strategy_improvement_proposals(
            session=session,
            latest_metrics=latest_metrics,
            strategy_insights=strategy_insights,
            hold_diagnostics=hold_diagnostics,
            regime_analysis=regime_analysis,
        )
        recommendations = await self._prune_recent_duplicates(session, self._dedupe_proposals(recommendations))
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "active_strategy": settings.trading.active_strategy,
            "symbol": settings.trading.pair,
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
            "strategy_insights": strategy_insights,
            "hold_diagnostics": hold_diagnostics,
            "regime_analysis": regime_analysis,
            "recommendations": [self._proposal_payload(item) for item in recommendations],
        }

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

    async def _multi_agent_proposals(
        self,
        session: AsyncSession,
        latest_metrics: MetricsSnapshot,
    ) -> list[AIProposal]:
        settings = get_settings()
        context = await self._build_llm_context(session, latest_metrics)
        coordinator = MultiAgentCoordinator()
        try:
            proposals = await coordinator.generate_proposals(
                context=context,
                llm_client=get_llm_advisor_client(),
            )
        except Exception:
            return []

        out: list[AIProposal] = []
        for proposal in proposals:
            diff = self._sanitize_diff(proposal.diff)
            if not diff:
                continue
            confidence = _safe_confidence(proposal.confidence)
            if confidence < settings.llm.min_confidence:
                continue
            out.append(
                AIProposal(
                    title=f"[{proposal.agent_name}] {proposal.title}",
                    proposal_type=proposal.proposal_type,
                    description=proposal.description,
                    diff=diff,
                    expected_impact=proposal.expected_impact,
                    evidence={
                        **proposal.evidence,
                        "agent_name": proposal.agent_name,
                        "agent_priority": proposal.priority,
                        "agent_reasoning": proposal.reasoning,
                    },
                    confidence=confidence,
                    ttl_hours=min(self.ttl_hours, settings.approval.timeout_hours),
                )
            )
        max_items = max(1, min(settings.llm.max_proposals, settings.multiagent.max_proposals))
        return out[:max_items]

    async def _build_llm_context(
        self,
        session: AsyncSession,
        latest_metrics: MetricsSnapshot,
    ) -> dict[str, Any]:
        settings = get_settings()
        symbol = settings.trading.pair
        recent_metrics_result = await session.execute(
            select(MetricsSnapshot)
            .where(MetricsSnapshot.is_paper.is_(True))
            .order_by(MetricsSnapshot.snapshot_at.desc())
            .limit(30)
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

        trade_quality = await self._analyze_trade_quality(session)
        regime_analysis = await self._get_regime_context(session, symbol=symbol)
        market_context = await self._get_market_context(symbol=symbol)
        order_book_context = await self._get_order_book_context(symbol=symbol)
        strategy_insights = await self._analyze_strategy_insights(session)
        hold_diagnostics = await self._strategy_hold_diagnostics(session)
        risk_metrics = self._build_risk_metrics(
            equity_values=equity_values,
            latest_metrics=latest_metrics,
            trade_quality=trade_quality,
        )
        execution_metrics = {
            "fills_24h": int(fills_24h_count or 0),
            "avg_slippage_bps_24h": float(avg_slippage_24h or 0.0),
            "expected_slippage_bps": settings.risk.slippage_bps,
            "fill_success_rate": 1.0 if int(fills_24h_count or 0) > 0 else 0.0,
        }

        return {
            "symbol": symbol,
            "timeframes": settings.trading.timeframe_list,
            "active_strategy": settings.trading.active_strategy,
            "supported_strategies": registry.list_names(),
            "risk_settings": {
                "per_trade": settings.risk.per_trade,
                "max_daily_loss": settings.risk.max_daily_loss,
                "max_exposure": settings.risk.max_exposure,
                "fee_bps": settings.risk.fee_bps,
                "slippage_bps": settings.risk.slippage_bps,
                "min_per_trade": settings.risk.min_per_trade,
                "max_per_trade": settings.risk.max_per_trade,
                "dynamic_sizing_enabled": settings.risk.dynamic_sizing_enabled,
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
                "regime_adaptation_enabled": settings.trading.regime_adaptation_enabled,
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
            "trade_quality": trade_quality,
            "regime_analysis": regime_analysis,
            "market_context": market_context,
            "order_book": order_book_context,
            "risk_metrics": risk_metrics,
            "execution_metrics": execution_metrics,
            "strategy_insights": strategy_insights,
            "hold_diagnostics": hold_diagnostics,
        }

    async def _analyze_trade_quality(self, session: AsyncSession) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(days=30)
        result = await session.execute(
            select(Order.side, Fill.price, Fill.quantity, Fill.fee, Fill.filled_at)
            .join(Fill, Fill.order_id == Order.id)
            .where(
                Fill.is_paper.is_(True),
                Fill.filled_at >= cutoff,
            )
            .order_by(Fill.filled_at.asc())
            .limit(1000)
        )
        rows = list(result.all())
        if not rows:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "current_streak": 0,
                "max_loss_streak": 0,
                "avg_duration_hrs": 0.0,
            }

        pnl_points: list[float] = []
        for side, price, quantity, fee, _filled_at in rows:
            notional = Decimal(str(price)) * Decimal(str(quantity))
            signed = notional if side == "SELL" else -notional
            pnl_points.append(float(signed - Decimal(str(fee))))

        wins = [value for value in pnl_points if value > 0]
        losses = [value for value in pnl_points if value < 0]

        max_loss_streak = 0
        current_loss_streak = 0
        for value in pnl_points:
            if value < 0:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0

        return {
            "total_trades": len(pnl_points),
            "winners": len(wins),
            "losers": len(losses),
            "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "current_streak": current_loss_streak,
            "max_loss_streak": max_loss_streak,
            "avg_duration_hrs": 1.0,
        }

    def _build_risk_metrics(
        self,
        *,
        equity_values: list[float],
        latest_metrics: MetricsSnapshot,
        trade_quality: dict[str, Any],
    ) -> dict[str, Any]:
        if len(equity_values) < 2:
            return {
                "current_drawdown": float(latest_metrics.max_drawdown),
                "max_drawdown": float(latest_metrics.max_drawdown),
                "drawdown_days": 0,
                "recovery_factor": 0.0,
                "var_95": 0.0,
                "var_99": 0.0,
                "cvar_95": 0.0,
                "current_exposure_pct": 0.0,
                "max_exposure_pct": get_settings().risk.max_exposure,
                "exposure_utilization": 0.0,
                "realized_vol_30d": 0.0,
                "vol_percentile": 50.0,
                "vol_regime": "normal",
            }

        returns: list[float] = []
        for i in range(1, len(equity_values)):
            prev = equity_values[i - 1]
            curr = equity_values[i]
            returns.append((curr - prev) / prev if prev > 0 else 0.0)

        sorted_returns = sorted(returns)
        idx95 = max(0, int(len(sorted_returns) * 0.05) - 1)
        idx99 = max(0, int(len(sorted_returns) * 0.01) - 1)
        var95 = sorted_returns[idx95] if sorted_returns else 0.0
        var99 = sorted_returns[idx99] if sorted_returns else 0.0
        tail95 = [ret for ret in returns if ret <= var95]
        cvar95 = (sum(tail95) / len(tail95)) if tail95 else var95

        peak = max(equity_values)
        current = equity_values[-1]
        current_drawdown = ((peak - current) / peak) if peak > 0 else 0.0

        vol = float(np.std(returns[-30:])) if len(returns) >= 2 else 0.0
        vol_percentile = min(100.0, max(0.0, (vol / 0.05) * 100))
        vol_regime = "high" if vol_percentile > 70 else ("low" if vol_percentile < 30 else "normal")

        max_exposure = get_settings().risk.max_exposure
        losers = int(trade_quality.get("losers", 0) or 0)
        total = int(trade_quality.get("total_trades", 0) or 0)
        exposure_utilization = min(1.0, max(0.0, (losers / total) if total > 0 else 0.0))

        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": float(latest_metrics.max_drawdown),
            "drawdown_days": int(max(0, len(returns) // 24)),
            "recovery_factor": float((latest_metrics.total_pnl or 0) / Decimal(str(max(1e-9, latest_metrics.max_drawdown)))),
            "var_95": var95 * current,
            "var_99": var99 * current,
            "cvar_95": cvar95 * current,
            "current_exposure_pct": exposure_utilization * max_exposure,
            "max_exposure_pct": max_exposure,
            "exposure_utilization": exposure_utilization,
            "realized_vol_30d": vol,
            "vol_percentile": vol_percentile,
            "vol_regime": vol_regime,
        }

    async def _get_regime_context(self, session: AsyncSession, *, symbol: str) -> dict[str, Any]:
        timeframe = get_settings().trading.timeframe_list[0]
        result = await session.execute(
            select(Candle)
            .where(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
            )
            .order_by(Candle.open_time.desc())
            .limit(240)
        )
        candles = list(reversed(result.scalars().all()))
        if len(candles) < 40:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "trend_strength": 0.0,
                "volatility_percentile": 0.5,
                "breakout_probability": 0.0,
                "transitions": 0,
                "hours_in_regime": 0,
            }

        detector = MarketRegimeDetector()
        analysis = detector.detect_regime(
            closes=[Decimal(str(c.close)) for c in candles],
            volumes=[Decimal(str(c.volume)) for c in candles],
            highs=[Decimal(str(c.high)) for c in candles],
            lows=[Decimal(str(c.low)) for c in candles],
        )

        transitions = 0
        previous = None
        step = 12
        for idx in range(60, len(candles), step):
            window = candles[max(0, idx - 120) : idx]
            if len(window) < 40:
                continue
            sampled = detector.detect_regime(
                closes=[Decimal(str(c.close)) for c in window],
                volumes=[Decimal(str(c.volume)) for c in window],
                highs=[Decimal(str(c.high)) for c in window],
                lows=[Decimal(str(c.low)) for c in window],
            ).regime.value
            if previous is not None and sampled != previous:
                transitions += 1
            previous = sampled

        return {
            "regime": analysis.regime.value,
            "confidence": analysis.confidence,
            "trend_strength": analysis.trend_strength,
            "volatility_percentile": analysis.volatility_percentile,
            "breakout_probability": analysis.breakout_probability,
            "transitions": transitions,
            "hours_in_regime": step,
        }

    async def _get_market_context(self, *, symbol: str) -> dict[str, Any]:
        try:
            context = await get_alternative_data_aggregator().build_market_context(symbol)
            return context.to_dict()
        except Exception:
            return {
                "symbol": symbol,
                "last_price": 0.0,
                "change_24h": 0.0,
                "volume_24h": 0.0,
                "fear_greed": 50,
                "funding_rate": 0.0,
            }

    async def _get_order_book_context(self, *, symbol: str) -> dict[str, Any]:
        try:
            payload = await get_binance_adapter().get_order_book(symbol=symbol, limit=20)
            snapshot = OrderBookAnalyzer.from_binance_depth(symbol=symbol, payload=payload)
            if snapshot is None:
                return {}
            return snapshot.to_dict()
        except Exception:
            return {}

    async def _analyze_strategy_insights(self, session: AsyncSession) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(days=30)
        recenter_result = await session.execute(
            select(func.count(EventLog.id))
            .where(
                EventLog.created_at >= since,
                EventLog.event_type.like("grid_recenter%"),
            )
        )
        sl_hits_result = await session.execute(
            select(func.count(EventLog.id))
            .where(
                EventLog.created_at >= since,
                EventLog.event_type == "global_stop_loss_triggered",
            )
        )
        tp_hits_result = await session.execute(
            select(func.count(Order.id))
            .where(
                Order.created_at >= since,
                Order.signal_reason == "grid_take_profit_buffer_hit",
            )
        )
        fills_result = await session.execute(
            select(func.count(Fill.id))
            .where(
                Fill.is_paper.is_(True),
                Fill.filled_at >= since,
            )
        )
        fill_count = int(fills_result.scalar_one_or_none() or 0)
        return {
            "grid_fill_rate": min(1.0, fill_count / 500.0),
            "avg_reversion_time": 1.0,
            "recenter_count_30d": int(recenter_result.scalar_one_or_none() or 0),
            "sl_hits": int(sl_hits_result.scalar_one_or_none() or 0),
            "tp_hits": int(tp_hits_result.scalar_one_or_none() or 0),
        }

    async def _strategy_hold_diagnostics(self, session: AsyncSession) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(hours=24)
        signal_reason = func.json_extract(EventLog.details, "$.signal_reason")
        hold_result = await session.execute(
            select(signal_reason, func.count(EventLog.id))
            .where(
                EventLog.event_type == "risk_hold",
                EventLog.created_at >= since,
            )
            .group_by(signal_reason)
        )
        reason_counts: dict[str, int] = {}
        hold_total = 0
        for reason, count in hold_result.all():
            key = str(reason or "unknown")
            value = int(count or 0)
            reason_counts[key] = value
            hold_total += value

        order_result = await session.execute(
            select(func.count(Order.id)).where(Order.created_at >= since)
        )
        order_count = int(order_result.scalar_one_or_none() or 0)
        total_cycles = hold_total + order_count
        hold_rate = float(hold_total / total_cycles) if total_cycles > 0 else 0.0
        return {
            "window_hours": 24,
            "hold_count_24h": hold_total,
            "order_count_24h": order_count,
            "hold_rate_24h": hold_rate,
            "reason_counts_24h": reason_counts,
        }

    async def _strategy_improvement_proposals(
        self,
        *,
        session: AsyncSession,
        latest_metrics: MetricsSnapshot,
        strategy_insights: dict[str, Any] | None = None,
        hold_diagnostics: dict[str, Any] | None = None,
        regime_analysis: dict[str, Any] | None = None,
    ) -> list[AIProposal]:
        settings = get_settings()
        active_strategy = settings.trading.active_strategy
        symbol = settings.trading.pair

        insights = strategy_insights or await self._analyze_strategy_insights(session)
        holds = hold_diagnostics or await self._strategy_hold_diagnostics(session)
        regime = regime_analysis or await self._get_regime_context(session, symbol=symbol)

        reason_counts = holds.get("reason_counts_24h", {})
        if not isinstance(reason_counts, dict):
            reason_counts = {}
        hold_rate = float(holds.get("hold_rate_24h", 0.0) or 0.0)
        grid_wait_holds = int(reason_counts.get("grid_wait_inside_band", 0) or 0)
        recenter_count = int(insights.get("recenter_count_30d", 0) or 0)
        regime_name = str(regime.get("regime", "unknown"))
        regime_confidence = float(regime.get("confidence", 0.0) or 0.0)
        trend_strength = float(regime.get("trend_strength", 0.0) or 0.0)
        latest_win_rate = float(latest_metrics.win_rate or 0.0)
        latest_drawdown = float(latest_metrics.max_drawdown or 0.0)

        recommendations: list[AIProposal] = []
        if active_strategy == "smart_grid_ai":
            if hold_rate >= 0.82 and grid_wait_holds >= 45:
                tighter_min = max(12, settings.trading.grid_min_spacing_bps - 8)
                tighter_max = max(tighter_min + 40, settings.trading.grid_max_spacing_bps - 30)
                shorter_cooldown = max(30, settings.trading.grid_cooldown_seconds - 30)
                recommendations.append(
                    AIProposal(
                        title="Increase smart-grid participation in low-activity regime",
                        proposal_type="grid_tuning",
                        description=(
                            "High hold-rate suggests the strategy is waiting too often inside the active band. "
                            "Tighten spacing and reduce cooldown modestly to increase trade participation."
                        ),
                        diff={
                            "trading": {
                                "grid_min_spacing_bps": tighter_min,
                                "grid_max_spacing_bps": tighter_max,
                                "grid_cooldown_seconds": shorter_cooldown,
                            }
                        },
                        expected_impact=(
                            "Higher fill probability with bounded increase in turnover."
                        ),
                        evidence={
                            "hold_rate_24h": hold_rate,
                            "grid_wait_holds_24h": grid_wait_holds,
                            "order_count_24h": int(holds.get("order_count_24h", 0) or 0),
                        },
                        confidence=0.68,
                        ttl_hours=self.ttl_hours,
                    )
                )

            if recenter_count >= 100:
                wider_min = min(500, settings.trading.grid_min_spacing_bps + 10)
                wider_max = min(900, settings.trading.grid_max_spacing_bps + 60)
                recenter_mode = (
                    "conservative"
                    if settings.trading.grid_recenter_mode == "aggressive"
                    else settings.trading.grid_recenter_mode
                )
                recommendations.append(
                    AIProposal(
                        title="Reduce recenter churn in smart-grid",
                        proposal_type="grid_tuning",
                        description=(
                            "Frequent recenter events indicate unstable band placement. "
                            "Widen spacing and reduce recenter aggressiveness."
                        ),
                        diff={
                            "trading": {
                                "grid_min_spacing_bps": wider_min,
                                "grid_max_spacing_bps": wider_max,
                                "grid_recenter_mode": recenter_mode,
                            }
                        },
                        expected_impact="Lower recenter churn and more stable grid behavior.",
                        evidence={
                            "recenter_count_30d": recenter_count,
                            "regime": regime_name,
                            "regime_confidence": regime_confidence,
                        },
                        confidence=0.66,
                        ttl_hours=self.ttl_hours,
                    )
                )

            if (
                regime_name in {"trending_bullish", "trending_bearish"}
                and regime_confidence >= 0.65
                and trend_strength >= 0.55
            ):
                recommendations.append(
                    AIProposal(
                        title="Switch to trend strategy for persistent trend regime",
                        proposal_type="strategy_switch",
                        description=(
                            "Detected persistent trend regime with high confidence. "
                            "A trend strategy is likely better aligned than neutral grid."
                        ),
                        diff={"trading": {"active_strategy": "trend_ema_fast"}},
                        expected_impact="Improved alignment with directional market phases.",
                        evidence={
                            "regime": regime_name,
                            "regime_confidence": regime_confidence,
                            "trend_strength": trend_strength,
                            "active_strategy": active_strategy,
                        },
                        confidence=0.64,
                        ttl_hours=self.ttl_hours,
                    )
                )
        elif regime_name in {"ranging_tight", "ranging_wide"} and regime_confidence >= 0.60:
            recommendations.append(
                AIProposal(
                    title="Switch to smart-grid for ranging regime",
                    proposal_type="strategy_switch",
                    description=(
                        "Current strategy is not optimal for a ranging market. "
                        "Switching to smart-grid can better capture mean reversion."
                    ),
                    diff={"trading": {"active_strategy": "smart_grid_ai"}},
                    expected_impact="Better fit between strategy profile and current market regime.",
                    evidence={
                        "regime": regime_name,
                        "regime_confidence": regime_confidence,
                        "active_strategy": active_strategy,
                    },
                    confidence=0.63,
                    ttl_hours=self.ttl_hours,
                )
            )
        elif latest_drawdown >= 0.10 and latest_win_rate < 0.45:
            recommendations.append(
                AIProposal(
                    title="Switch to smart-grid after weak trend performance",
                    proposal_type="strategy_switch",
                    description=(
                        "Current strategy is underperforming with high drawdown and low win-rate. "
                        "Switching to smart-grid can reduce directional dependence."
                    ),
                    diff={"trading": {"active_strategy": "smart_grid_ai"}},
                    expected_impact="Improved robustness during mixed or low-trend conditions.",
                    evidence={
                        "max_drawdown": latest_drawdown,
                        "win_rate": latest_win_rate,
                        "active_strategy": active_strategy,
                    },
                    confidence=0.61,
                    ttl_hours=self.ttl_hours,
                )
            )

        return self._dedupe_proposals(recommendations)

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
            "fallback_enabled": status.fallback_enabled,
            "fallback_provider": status.fallback_provider,
            "fallback_model": status.fallback_model,
            "fallback_configured": status.fallback_configured,
            "last_provider_used": status.last_provider_used,
            "last_model_used": status.last_model_used,
            "last_used_at": status.last_used_at,
            "last_fallback_used": status.last_fallback_used,
            "last_error": status.last_error,
        }

    def multiagent_status(self) -> dict[str, Any]:
        settings = get_settings()
        return {
            "enabled": settings.multiagent.enabled,
            "max_proposals": settings.multiagent.max_proposals,
            "min_confidence": settings.multiagent.min_confidence,
            "meta_agent_enabled": settings.multiagent.meta_agent_enabled,
            "strategy_agent_enabled": settings.multiagent.strategy_agent_enabled,
            "risk_agent_enabled": settings.multiagent.risk_agent_enabled,
            "market_agent_enabled": settings.multiagent.market_agent_enabled,
            "execution_agent_enabled": settings.multiagent.execution_agent_enabled,
            "sentiment_agent_enabled": settings.multiagent.sentiment_agent_enabled,
        }

    async def test_llm_connection(self) -> dict[str, Any]:
        return await get_llm_advisor_client().test_connection()

    async def test_multiagent_connection(self) -> dict[str, Any]:
        settings = get_settings()
        if not settings.multiagent.enabled:
            return {
                "ok": False,
                "enabled": False,
                "message": "Multi-agent is disabled (MULTIAGENT_ENABLED=false)",
                "agents_used": [],
                "proposal_count": 0,
            }
        context = {
            "active_strategy": settings.trading.active_strategy,
            "latest_metrics": {
                "strategy_name": settings.trading.active_strategy,
                "total_trades": 100,
                "winning_trades": 54,
                "losing_trades": 46,
                "total_pnl": 80.0,
                "max_drawdown": 0.06,
                "win_rate": 0.54,
                "sharpe_ratio": 1.2,
            },
            "risk_settings": {
                "per_trade": settings.risk.per_trade,
                "max_daily_loss": settings.risk.max_daily_loss,
                "max_exposure": settings.risk.max_exposure,
                "slippage_bps": settings.risk.slippage_bps,
            },
            "trading_settings": {
                "grid_levels": settings.trading.grid_levels,
                "grid_min_spacing_bps": settings.trading.grid_min_spacing_bps,
                "grid_max_spacing_bps": settings.trading.grid_max_spacing_bps,
            },
            "market_context": {"fear_greed": 50, "funding_rate": 0.0001},
            "order_book": {"spread_bps": 3.0},
            "execution_metrics": {
                "avg_slippage_bps_24h": float(settings.risk.slippage_bps),
                "expected_slippage_bps": float(settings.risk.slippage_bps),
            },
            "regime_analysis": {"regime": "ranging_tight", "confidence": 0.65},
            "strategy_insights": {"grid_fill_rate": 0.6},
        }
        coordinator = MultiAgentCoordinator()
        proposals = await coordinator.generate_proposals(
            context=context,
            llm_client=get_llm_advisor_client(),
        )
        return {
            "ok": True,
            "enabled": True,
            "message": "Multi-agent coordinator responded",
            "agents_used": [agent.name for agent in coordinator.agents],
            "proposal_count": len(proposals),
        }

    @staticmethod
    def _proposal_payload(proposal: AIProposal) -> dict[str, Any]:
        return {
            "title": proposal.title,
            "proposal_type": proposal.proposal_type,
            "description": proposal.description,
            "diff": proposal.diff,
            "expected_impact": proposal.expected_impact,
            "evidence": proposal.evidence,
            "confidence": proposal.confidence,
            "ttl_hours": proposal.ttl_hours,
        }

    @staticmethod
    def _dedupe_proposals(proposals: list[AIProposal]) -> list[AIProposal]:
        seen: set[tuple[str, str]] = set()
        unique: list[AIProposal] = []
        for proposal in proposals:
            key = (proposal.proposal_type, proposal.title.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(proposal)
        return unique

    async def _prune_recent_duplicates(
        self,
        session: AsyncSession,
        proposals: list[AIProposal],
        *,
        lookback_hours: int = 12,
    ) -> list[AIProposal]:
        if not proposals:
            return []
        cutoff = datetime.now(UTC) - timedelta(hours=max(1, lookback_hours))
        result = await session.execute(
            select(Approval.title)
            .where(
                Approval.created_at >= cutoff,
                Approval.status.in_([ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value]),
            )
            .limit(500)
        )
        recent_titles = {str(title).strip().lower() for title in result.scalars().all()}
        return [proposal for proposal in proposals if proposal.title.strip().lower() not in recent_titles]

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
            "risk": {
                "per_trade",
                "min_per_trade",
                "max_per_trade",
                "max_daily_loss",
                "max_exposure",
                "fee_bps",
                "slippage_bps",
                "dynamic_sizing_enabled",
                "confidence_sizing_enabled",
                "regime_sizing_enabled",
                "drawdown_scaling_enabled",
            },
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
                "regime_adaptation_enabled",
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
