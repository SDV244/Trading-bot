"""Tests for DRL optimizer."""

from typing import Any

from packages.core.ai.drl_optimizer import DRLOptimizer


class _FakePPO:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.learned = False

    def learn(self, total_timesteps: int) -> None:
        self.learned = total_timesteps > 0

    def predict(self, _obs: Any, deterministic: bool = True) -> tuple[int, Any]:
        return (2 if deterministic else 1, None)


def test_drl_optimizer_train_and_propose(monkeypatch: Any) -> None:
    monkeypatch.setattr("packages.core.ai.drl_optimizer.PPO", _FakePPO)
    optimizer = DRLOptimizer()
    rewards = [0.001, -0.0005, 0.002, 0.0015] * 20
    optimizer.train(reward_series=rewards, timesteps=256)
    proposal = optimizer.propose(
        base_risk_per_trade=0.005,
        reward_series=rewards,
        equity_curve=[10000 + i * 5 for i in range(len(rewards) + 1)],
    )
    assert proposal is not None
    assert proposal.diff["risk"]["per_trade"] > 0


def test_walk_forward_validate_returns_bool(monkeypatch: Any) -> None:
    monkeypatch.setattr("packages.core.ai.drl_optimizer.PPO", _FakePPO)
    optimizer = DRLOptimizer()
    result = optimizer.walk_forward_validate(
        equity_series=[10000 + i * 2 for i in range(120)],
        pnl_series=[0.1 for _ in range(120)],
        exposure_series=[0.2 for _ in range(120)],
    )
    assert isinstance(result, bool)
