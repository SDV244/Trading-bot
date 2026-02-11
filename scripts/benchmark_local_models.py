"""Benchmark local Ollama models for trading-advisor JSON workload."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODELS = [
    "llama3.2:3b-instruct-q4",
    "llama3.2:1b",
    "qwen2.5:1.5b",
]
RUNS = 4
TIMEOUT_S = 180

SYSTEM_PROMPT = (
    "You are a senior crypto spot-grid risk advisor. Output strict JSON only. "
    "Never propose disabling safety controls."
)

SCHEMA = {
    "proposals": [
        {
            "title": "string",
            "proposal_type": "risk_tuning|execution_tuning|grid_tuning|strategy_switch|anomaly_alert",
            "description": "string",
            "diff": {
                "risk": {
                    "per_trade": "number",
                    "max_daily_loss": "number",
                    "max_exposure": "number",
                    "fee_bps": "integer",
                    "slippage_bps": "integer",
                },
                "trading": {
                    "active_strategy": "string",
                    "grid_levels": "integer",
                    "grid_min_spacing_bps": "integer",
                    "grid_max_spacing_bps": "integer",
                    "grid_volatility_blend": "number",
                    "grid_trend_tilt": "number",
                },
            },
            "expected_impact": "string",
            "evidence": {"key": "value"},
            "confidence": "0.0-1.0",
            "ttl_hours": "integer",
        }
    ]
}

CONTEXT = {
    "symbol": "BTCUSDT",
    "active_strategy": "smart_grid_ai",
    "latest_metrics": {
        "total_trades": 182,
        "win_rate": 0.54,
        "max_drawdown": 0.072,
        "total_pnl": 143.2,
        "profit_factor": 1.38,
    },
    "risk_settings": {
        "per_trade": 0.015,
        "max_daily_loss": 0.025,
        "max_exposure": 0.35,
        "fee_bps": 10,
        "slippage_bps": 5,
    },
    "trading_settings": {
        "grid_levels": 7,
        "grid_min_spacing_bps": 70,
        "grid_max_spacing_bps": 360,
        "grid_volatility_blend": 0.85,
        "grid_trend_tilt": 1.35,
    },
    "regime_analysis": {
        "regime": "ranging_low_vol",
        "trend_strength": 0.21,
        "atr_1h": 245.0,
    },
}

USER_PROMPT = (
    "Return ONLY a JSON object following this schema. No markdown.\n"
    f"Schema:\n{json.dumps(SCHEMA)}\n\n"
    f"Context:\n{json.dumps(CONTEXT)}"
)


@dataclass
class RunResult:
    ok: bool
    latency_ms: int
    proposals: int
    quality: float
    error: str | None


def extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no json object")
    data = json.loads(clean[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("not object")
    return data


def score_quality(payload: dict[str, Any]) -> tuple[int, float]:
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        return 0, 0.0
    if not proposals:
        return 0, 0.2
    required = [
        "title",
        "proposal_type",
        "description",
        "diff",
        "expected_impact",
        "evidence",
        "confidence",
    ]
    scores: list[float] = []
    valid_count = 0
    for proposal in proposals[:3]:
        if not isinstance(proposal, dict):
            scores.append(0.0)
            continue
        fields = sum(1 for key in required if key in proposal)
        base = fields / len(required)
        conf_ok = isinstance(proposal.get("confidence"), (float, int, str))
        diff_ok = isinstance(proposal.get("diff"), dict)
        bonus = (0.1 if conf_ok else 0.0) + (0.1 if diff_ok else 0.0)
        score = min(1.0, base + bonus)
        if score >= 0.7:
            valid_count += 1
        scores.append(score)
    return valid_count, sum(scores) / len(scores)


async def run_once(client: httpx.AsyncClient, model: str) -> RunResult:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "options": {"temperature": 0.15},
    }
    started = time.perf_counter()
    try:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        content = ((response_json.get("message") or {}).get("content")) or ""
        parsed = extract_json(content)
        valid_count, quality = score_quality(parsed)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            ok=True,
            latency_ms=latency_ms,
            proposals=valid_count,
            quality=quality,
            error=None,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            ok=False,
            latency_ms=latency_ms,
            proposals=0,
            quality=0.0,
            error=exc.__class__.__name__,
        )


async def bench_model(model: str) -> dict[str, Any]:
    timeout = httpx.Timeout(TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        runs: list[RunResult] = []
        for _ in range(RUNS):
            runs.append(await run_once(client, model))

    ok_runs = [run for run in runs if run.ok]
    success_rate = len(ok_runs) / len(runs)
    latencies = [run.latency_ms for run in ok_runs]
    qualities = [run.quality for run in ok_runs]
    proposals = [run.proposals for run in ok_runs]

    p95 = max(latencies) if latencies else TIMEOUT_S * 1000
    p50 = int(statistics.median(latencies)) if latencies else TIMEOUT_S * 1000
    avg_quality = statistics.mean(qualities) if qualities else 0.0
    avg_valid_proposals = statistics.mean(proposals) if proposals else 0.0

    # Composite: reliability 50%, quality 30%, speed 20%
    speed_score = max(0.0, 1.0 - (p95 / (TIMEOUT_S * 1000)))
    score = (0.5 * success_rate) + (0.3 * avg_quality) + (0.2 * speed_score)

    return {
        "model": model,
        "success_rate": round(success_rate, 3),
        "p50_ms": p50,
        "p95_ms": int(p95),
        "avg_quality": round(avg_quality, 3),
        "avg_valid_proposals": round(avg_valid_proposals, 3),
        "score": round(score, 4),
        "runs": [
            {
                "ok": run.ok,
                "latency_ms": run.latency_ms,
                "quality": round(run.quality, 3),
                "valid_proposals": run.proposals,
                "error": run.error,
            }
            for run in runs
        ],
    }


async def main() -> None:
    results = []
    for model in MODELS:
        results.append(await bench_model(model))
    results.sort(key=lambda item: item["score"], reverse=True)
    payload = {"best_model": results[0]["model"], "results": results}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
