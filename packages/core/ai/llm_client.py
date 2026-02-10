"""Provider-agnostic LLM client for proposal generation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from packages.core.config import LLMSettings, get_settings


class LLMClientError(RuntimeError):
    """Raised when LLM provider calls fail."""


@dataclass(slots=True, frozen=True)
class LLMStatus:
    """Runtime status of configured LLM provider."""

    enabled: bool
    provider: str
    model: str
    configured: bool
    base_url: str | None
    prefer_llm: bool
    fallback_to_rules: bool
    last_error: str | None


class LLMAdvisorClient:
    """Calls external LLM providers and returns proposal JSON."""

    def __init__(self) -> None:
        self._last_error: str | None = None

    def status(self) -> LLMStatus:
        settings = get_settings().llm
        return LLMStatus(
            enabled=settings.enabled,
            provider=settings.provider,
            model=settings.model,
            configured=self._is_configured(settings),
            base_url=settings.base_url or None,
            prefer_llm=settings.prefer_llm,
            fallback_to_rules=settings.fallback_to_rules,
            last_error=self._last_error,
        )

    async def test_connection(self) -> dict[str, Any]:
        """Ping configured provider with a small structured request."""
        settings = get_settings().llm
        if not settings.enabled:
            return {
                "ok": False,
                "provider": settings.provider,
                "model": settings.model,
                "latency_ms": None,
                "message": "LLM is disabled (LLM_ENABLED=false)",
            }
        if not self._is_configured(settings):
            return {
                "ok": False,
                "provider": settings.provider,
                "model": settings.model,
                "latency_ms": None,
                "message": "LLM is enabled but not fully configured",
            }

        payload = {
            "symbol": "BTCUSDT",
            "active_strategy": "smart_grid_ai",
            "latest_metrics": {
                "total_trades": 100,
                "win_rate": 0.52,
                "max_drawdown": 0.06,
                "total_pnl": 124.0,
            },
            "instructions": "Return one safe proposal or empty proposals list.",
        }
        started = time.perf_counter()
        try:
            proposals = await self.generate_proposals(payload)
        except Exception as exc:
            return {
                "ok": False,
                "provider": settings.provider,
                "model": settings.model,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "message": f"LLM test failed: {exc}",
            }

        return {
            "ok": True,
            "provider": settings.provider,
            "model": settings.model,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "message": "LLM provider responded successfully",
            "raw_proposals_count": len(proposals),
        }

    async def generate_proposals(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate raw proposal dicts from the configured LLM provider."""
        settings = get_settings().llm
        if not settings.enabled:
            return []
        if not self._is_configured(settings):
            self._last_error = "LLM enabled but missing required configuration"
            return []

        prompt = self._build_user_prompt(context)
        body = self._build_request_payload(settings, prompt)
        url = self._resolve_url(settings)
        headers = self._build_headers(settings)

        last_exc: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            if attempt > 0:
                await self._sleep_backoff(attempt)
            try:
                text = await self._execute_request(
                    url=url,
                    headers=headers,
                    payload=body,
                    timeout_seconds=settings.timeout_seconds,
                )
                parsed = _extract_json_payload(text)
                proposals = parsed.get("proposals", [])
                if not isinstance(proposals, list):
                    raise LLMClientError("LLM response missing proposals list")
                self._last_error = None
                return [item for item in proposals if isinstance(item, dict)]
            except Exception as exc:
                last_exc = exc
                self._last_error = str(exc)
                continue

        if last_exc is None:
            return []
        raise LLMClientError(str(last_exc))

    @staticmethod
    async def _sleep_backoff(attempt: int) -> None:
        import asyncio

        await asyncio.sleep(min(3.0, 0.35 * attempt))

    @staticmethod
    async def _execute_request(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> str:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return _extract_text_response(data)

    @staticmethod
    def _is_configured(settings: LLMSettings) -> bool:
        if not settings.model.strip():
            return False
        if settings.provider == "ollama":
            return True
        return bool(settings.api_key.strip())

    @staticmethod
    def _build_user_prompt(context: dict[str, Any]) -> str:
        schema = {
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
                            "grid_take_profit_buffer": "number",
                            "grid_stop_loss_buffer": "number",
                            "grid_recenter_mode": "aggressive|conservative",
                        },
                    },
                    "expected_impact": "string",
                    "evidence": {"key": "value"},
                    "confidence": "0.0-1.0",
                    "ttl_hours": "integer",
                }
            ]
        }
        return (
            "Return ONLY a JSON object following this schema.\n"
            "No markdown, no prose, no code fences.\n"
            f"Schema:\n{json.dumps(schema)}\n\n"
            "Context:\n"
            f"{json.dumps(context, default=str)}"
        )

    def _resolve_url(self, settings: LLMSettings) -> str:
        provider = settings.provider
        if provider == "openai":
            return settings.base_url.strip() or "https://api.openai.com/v1/chat/completions"
        if provider == "anthropic":
            return settings.base_url.strip() or "https://api.anthropic.com/v1/messages"
        if provider == "gemini":
            base = settings.base_url.strip() or (
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.model}:generateContent"
            )
            if settings.api_key.strip() and "key=" not in base:
                separator = "&" if "?" in base else "?"
                return f"{base}{separator}{urlencode({'key': settings.api_key.strip()})}"
            return base
        if provider == "ollama":
            return settings.base_url.strip() or "http://127.0.0.1:11434/api/chat"
        raise LLMClientError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _build_headers(settings: LLMSettings) -> dict[str, str]:
        provider = settings.provider
        if provider == "openai":
            return {
                "Authorization": f"Bearer {settings.api_key.strip()}",
                "Content-Type": "application/json",
            }
        if provider == "anthropic":
            return {
                "x-api-key": settings.api_key.strip(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        if provider == "gemini":
            return {"Content-Type": "application/json"}
        if provider == "ollama":
            return {"Content-Type": "application/json"}
        return {"Content-Type": "application/json"}

    @staticmethod
    def _build_request_payload(settings: LLMSettings, user_prompt: str) -> dict[str, Any]:
        system_prompt = settings.system_prompt.strip() or (
            "You are a crypto trading assistant. Return strict JSON proposals only."
        )
        provider = settings.provider
        if provider == "openai":
            return {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": settings.temperature,
                "max_tokens": settings.max_output_tokens,
                "response_format": {"type": "json_object"},
            }
        if provider == "anthropic":
            return {
                "model": settings.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": settings.temperature,
                "max_tokens": settings.max_output_tokens,
            }
        if provider == "gemini":
            return {
                "generationConfig": {
                    "temperature": settings.temperature,
                    "maxOutputTokens": settings.max_output_tokens,
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                    }
                ],
            }
        if provider == "ollama":
            return {
                "model": settings.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": settings.temperature,
                },
            }
        raise LLMClientError(f"Unsupported LLM provider: {provider}")


def _extract_text_response(response_json: dict[str, Any]) -> str:
    if "choices" in response_json:
        choices = response_json.get("choices", [])
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    if "content" in response_json and isinstance(response_json["content"], list):
        blocks = response_json["content"]
        if blocks and isinstance(blocks[0], dict):
            text = blocks[0].get("text", "")
            if isinstance(text, str):
                return text
    if "candidates" in response_json:
        candidates = response_json.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if isinstance(parts, list) and parts and isinstance(parts[0], dict):
                text = parts[0].get("text", "")
                if isinstance(text, str):
                    return text
    if "message" in response_json and isinstance(response_json["message"], dict):
        content = response_json["message"].get("content", "")
        if isinstance(content, str):
            return content
    if "output_text" in response_json and isinstance(response_json["output_text"], str):
        return response_json["output_text"]
    raise LLMClientError("Unable to parse text content from provider response")


def _extract_json_payload(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMClientError("LLM returned non-JSON content")
    candidate = clean[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM returned non-object JSON")
    return parsed


_llm_client: LLMAdvisorClient | None = None
_llm_client_lock = threading.Lock()


def get_llm_advisor_client() -> LLMAdvisorClient:
    """Get singleton LLM advisor client."""
    global _llm_client
    if _llm_client is None:
        with _llm_client_lock:
            if _llm_client is None:
                _llm_client = LLMAdvisorClient()
    return _llm_client

