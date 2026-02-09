"""Deep RL optimizer (PPO) for strategy parameter suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO

from packages.core.metrics import MetricsCalculator
from packages.research.walk_forward import WalkForwardEvaluator


@dataclass(slots=True, frozen=True)
class DRLProposal:
    """Optimizer proposal output."""

    title: str
    diff: dict[str, Any]
    expected_impact: str
    evidence: dict[str, Any]
    confidence: float


class _ParamTuningEnv(Env[np.ndarray, int]):
    """Minimal environment where actions adjust risk_per_trade multiplier."""

    metadata = {"render_modes": []}

    def __init__(self, rewards: list[float]) -> None:
        super().__init__()
        self.rewards = rewards or [0.0]
        self.idx = 0
        self.scale = 1.0
        self.action_space = spaces.Discrete(3)  # type: ignore[assignment]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, 0.5], dtype=np.float32),
            high=np.array([5.0, 2.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        self.idx = 0
        self.scale = 1.0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action == 0:
            self.scale = max(0.5, self.scale - 0.05)
        elif action == 2:
            self.scale = min(2.0, self.scale + 0.05)

        reward = float(self.rewards[self.idx]) * self.scale
        self.idx += 1
        terminated = self.idx >= len(self.rewards)
        return self._obs(), reward, terminated, False, {"scale": self.scale}

    def _obs(self) -> np.ndarray:
        current_reward = float(self.rewards[min(self.idx, len(self.rewards) - 1)])
        return np.array([current_reward, self.scale], dtype=np.float32)


class DRLOptimizer:
    """PPO-based optimizer producing parameter tuning proposals."""

    def __init__(self) -> None:
        self.model: PPO | None = None
        self.last_scale: float = 1.0

    def train(
        self,
        reward_series: list[float],
        *,
        timesteps: int = 1_000,
        learning_rate: float = 3e-4,
    ) -> None:
        env = _ParamTuningEnv(reward_series)
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=min(128, max(16, len(reward_series))),
            batch_size=16,
            verbose=0,
        )
        self.model.learn(total_timesteps=max(256, timesteps))

    def propose(
        self,
        *,
        base_risk_per_trade: float,
        reward_series: list[float],
        equity_curve: list[float],
    ) -> DRLProposal | None:
        if self.model is None or not reward_series:
            return None

        env = _ParamTuningEnv(reward_series)
        obs, _ = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(int(action))
        scale = float(info.get("scale", 1.0))
        self.last_scale = scale

        suggested = round(base_risk_per_trade * scale, 6)
        metrics = MetricsCalculator().calculate(
            trade_pnls=reward_series,
            equity_curve=equity_curve if equity_curve else [10_000.0, 10_000.0],
            fees_paid=0.0,
            exposures_pct=[0.2 for _ in reward_series] or [0.0],
        )
        confidence = float(min(0.9, max(0.55, abs(scale - 1.0) + 0.55)))

        return DRLProposal(
            title="DRL risk_per_trade adjustment proposal",
            diff={"risk": {"per_trade": suggested}},
            expected_impact="Tune risk allocation based on learned reward dynamics.",
            evidence={
                "suggested_scale": scale,
                "composite_score": metrics.composite_score,
                "max_drawdown": metrics.max_drawdown,
            },
            confidence=confidence,
        )

    def walk_forward_validate(
        self,
        *,
        equity_series: list[float],
        pnl_series: list[float],
        exposure_series: list[float],
    ) -> bool:
        evaluator = WalkForwardEvaluator()
        result = evaluator.evaluate(
            equity_series=equity_series,
            trade_pnl_series=pnl_series,
            exposure_series=exposure_series,
            train_size=max(20, len(equity_series) // 3),
            test_size=max(10, len(equity_series) // 6),
            step=max(5, len(equity_series) // 12),
        )
        return result.stable
