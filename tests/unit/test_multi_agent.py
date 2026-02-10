"""Tests for multi-agent coordinator behavior."""

import pytest

from packages.core.ai.multi_agent import AgentProposal, MetaAgent, MultiAgentCoordinator
from packages.core.config import reload_settings


def test_meta_agent_dedupes_and_ranks() -> None:
    proposals = [
        AgentProposal(
            agent_name="strategy_agent",
            title="A",
            proposal_type="grid_tuning",
            description="A",
            diff={"trading": {"grid_levels": 6}},
            expected_impact="A",
            evidence={},
            confidence=0.7,
            priority=3,
            reasoning="A",
        ),
        AgentProposal(
            agent_name="risk_agent",
            title="B",
            proposal_type="grid_tuning",
            description="B",
            diff={"trading": {"grid_levels": 6}},
            expected_impact="B",
            evidence={},
            confidence=0.9,
            priority=4,
            reasoning="B",
        ),
    ]
    ranked = MetaAgent.rank_and_merge(proposals, max_items=5)
    assert len(ranked) == 1
    assert ranked[0].title == "B"


@pytest.mark.asyncio
async def test_multiagent_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULTIAGENT_ENABLED", "false")
    monkeypatch.setenv("LLM_ENABLED", "false")
    reload_settings()
    coordinator = MultiAgentCoordinator()
    result = await coordinator.generate_proposals(context={}, llm_client=object())
    assert result == []

