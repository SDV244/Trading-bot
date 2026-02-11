"""Unit tests for LLM client safety behavior."""

from __future__ import annotations

import pytest

from packages.core.ai.llm_client import LLMAdvisorClient, LLMClientError
from packages.core.config import reload_settings


@pytest.mark.asyncio
async def test_llm_client_redacts_key_from_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider errors must not leak API keys in runtime status."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_API_KEY", "DUMMY_KEY")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    reload_settings()

    async def _fail_request(**_kwargs):
        raise RuntimeError(
            "Client error '429 Too Many Requests' for url "
            "'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            "?key=THIS_SHOULD_NOT_LEAK'"
        )

    monkeypatch.setattr(LLMAdvisorClient, "_execute_request", staticmethod(_fail_request))
    client = LLMAdvisorClient()

    with pytest.raises(LLMClientError):
        await client.generate_structured(context={"symbol": "BTCUSDT"})

    status = client.status()
    assert status.last_error is not None
    assert "THIS_SHOULD_NOT_LEAK" not in status.last_error
    assert "key=REDACTED" in status.last_error


@pytest.mark.asyncio
async def test_llm_client_falls_back_to_ollama_when_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary Gemini failures should fail over to configured Ollama fallback."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_API_KEY", "DUMMY_KEY")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "http://127.0.0.1:11434/api/chat")
    reload_settings()

    async def _execute_request(**kwargs):
        url = str(kwargs.get("url", ""))
        if "generativelanguage.googleapis.com" in url:
            raise RuntimeError("Client error '429 Too Many Requests' for url '?key=TOP_SECRET'")
        return '{"proposals":[{"title":"Fallback","proposal_type":"risk_tuning","description":"ok","diff":{"risk":{"per_trade":0.01}},"expected_impact":"ok","evidence":{},"confidence":0.8,"ttl_hours":2}]}'

    monkeypatch.setattr(LLMAdvisorClient, "_execute_request", staticmethod(_execute_request))
    client = LLMAdvisorClient()
    proposals = await client.generate_structured(context={"symbol": "BTCUSDT"})

    assert len(proposals) == 1
    assert proposals[0]["title"] == "Fallback"
    status = client.status()
    assert status.fallback_enabled is True
    assert status.fallback_provider == "ollama"
    assert status.last_error is None
