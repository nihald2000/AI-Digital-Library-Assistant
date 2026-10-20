"""Offline tests for the free-tier LLM provider chain."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from services import llm_providers
from services.llm_service import LLMService


CFG = SimpleNamespace(  # minimal config surface for the resolver
    LLM_PROVIDER_CHAIN="gemini,groq,openrouter,nebius,openai,mistral",
    GEMINI_API_KEY="gk",
    GEMINI_BASE_URL="https://gemini.test/v1/",
    GEMINI_MODEL="gemini-2.5-flash",
    GEMINI_FAST_MODEL="gemini-2.5-flash-lite",
    GROQ_API_KEY="grk",
    GROQ_BASE_URL="https://groq.test/v1",
    GROQ_MODEL="openai/gpt-oss-120b",
    GROQ_FAST_MODEL="openai/gpt-oss-20b",
    OPENROUTER_API_KEY=None,  # no openrouter key -> filtered out
    OPENROUTER_BASE_URL="https://openrouter.test/v1",
    OPENROUTER_MODEL="openai/gpt-oss-120b:free",
    OPENROUTER_FAST_MODEL="openai/gpt-oss-20b:free",
    NEBIUS_API_KEY="nk",
    NEBIUS_BASE_URL="https://nebius.test/v1/",
    NEBIUS_MODEL="meta-llama/Llama-3.3-70B-Instruct",
    OPENAI_API_KEY=None,
    OPENAI_MODEL="gpt-5.1-chat-latest",
    FAST_MODEL="gpt-5-mini",
    MISTRAL_API_KEY=None,
    MISTRAL_MODEL="mistral-large-2407",
)


def test_resolver_order_and_key_filtering():
    legs = llm_providers.resolve_llm_providers(CFG)
    names = [l["name"] for l in legs]
    # user-mandated order: Gemini first, Groq backup; openrouter dropped (no key)
    assert names == ["gemini", "groq", "nebius"]
    assert legs[0]["model"] == "gemini-2.5-flash"
    assert all(l["api_key"] for l in legs)


def test_user_empty_chain():
    cfg = SimpleNamespace(**{**CFG.__dict__, "GEMINI_API_KEY": None,
                             "GROQ_API_KEY": None, "NEBIUS_API_KEY": None,
                             "MISTRAL_API_KEY": None})
    assert llm_providers.resolve_llm_providers(cfg) == []


def test_leg_for_model_prefixes():
    assert llm_providers.leg_for_model("groq/openai/gpt-oss-120b") == "groq"
    assert llm_providers.leg_for_model("gemini/gemini-2.5-flash") == "gemini"
    assert llm_providers.leg_for_model("openrouter/x:free") == "openrouter"
    assert llm_providers.leg_for_model("meta-llama/Llama-3.3-70B-Instruct") == "nebius"
    assert llm_providers.leg_for_model("gpt-5.1-chat-latest") == "openai"
    assert llm_providers.leg_for_model("mistral-large-2407") == "mistral"
    assert llm_providers.leg_for_model("totally-unknown") is None


def test_pick_llm_provider_skips_noncompat():
    leg = llm_providers.pick_llm_provider(CFG)
    assert leg["name"] == "gemini"  # first openai-comat leg in order


class FakeExc(Exception):
    def __init__(self, status_code=None, text="boom"):
        super().__init__(text)
        self.status_code = status_code


def test_permanent_error_classification():
    assert llm_providers.permanent_llm_error(FakeExc(401)) is True
    assert llm_providers.permanent_llm_error(FakeExc(402)) is True  # like Nebius 402
    assert llm_providers.permanent_llm_error(FakeExc(429)) is True
    assert llm_providers.permanent_llm_error(
        FakeExc(None, "credit_balance_exhausted")) is True
    assert llm_providers.permanent_llm_error(FakeExc(500, "rate limit skip")) is False


def _svc_with_stubbed_legs(monkeypatch, legs: List[Dict[str, str]]) -> tuple:
    """LLMService whose chain and per-leg generation are fully stubbed."""
    monkeypatch.setattr(llm_providers, "resolve_llm_providers", lambda *a, **k: legs)
    calls: List[str] = []

    async def fake_gen(self, leg, prompt, model_name, max_tokens, temperature):
        leg_name = leg["name"]
        calls.append((leg_name, model_name))
        if leg_name == "fake-fail":
            raise FakeExc(None, "credit_balance_exhausted - no credits")
        return f"ok-via-{leg_name}:{model_name}"

    monkeypatch.setattr(LLMService, "_generate_on_leg", fake_gen)
    svc = LLMService.__new__(LLMService)  # skip SDK client init / real config
    svc.config = CFG
    svc._disabled = {}
    svc._method_stub_calls = calls
    return svc, calls


def test_generate_text_fails_over_after_quota_error(monkeypatch):
    legs = [
        {"name": "fake-fail", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m1", "fast_model": "m1f"},
        {"name": "fake-win", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m2", "fast_model": "m2f"},
    ]
    svc, calls = _svc_with_stubbed_legs(monkeypatch, legs)

    out = asyncio.run(svc.generate_text("hello", model="auto", max_tokens=10))
    assert out == "ok-via-fake-win:m2"
    assert calls == [("fake-fail", "m1"), ("fake-win", "m2")]
    assert "fake-fail" in svc._disabled  # permanently dropped for the session


def test_disabled_leg_skipped_within_session(monkeypatch):
    legs = [
        {"name": "fake-fail", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m1", "fast_model": "m1f"},
        {"name": "fake-win", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m2", "fast_model": "m2f"},
    ]
    svc, calls = _svc_with_stubbed_legs(monkeypatch, legs)
    svc._disabled["fake-fail"] = "already dead"

    out = asyncio.run(svc.generate_text("hello"))
    assert out == "ok-via-fake-win:m2"
    assert calls == [("fake-win", "m2")]


def test_all_legs_dead_raises_with_details(monkeypatch):
    legs = [
        {"name": "a", "kind": "openai-compat", "api_key": "x", "base_url": "x",
         "model": "ma", "fast_model": "ma"},
    ]
    async def fake_gen(self, leg, prompt, model_name, max_tokens, temperature):
        raise FakeExc(None, "credit_balance_exhausted")

    monkeypatch.setattr(llm_providers, "resolve_llm_providers", lambda *a, **k: legs)
    monkeypatch.setattr(LLMService, "_generate_on_leg", fake_gen)
    svc = LLMService.__new__(LLMService)
    svc.config = CFG
    svc._disabled = {}

    with pytest.raises(ValueError, match="All LLM providers failed"):
        asyncio.run(svc.generate_text("hello"))


def test_explicit_model_routes_and_strips_prefix(monkeypatch):
    legs = [
        {"name": "gemini", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "gemini-2.5-flash", "fast_model": "lite"},
    ]
    svc, calls = _svc_with_stubbed_legs(monkeypatch, legs)
    out = asyncio.run(svc.generate_text("hi", model="gemini/gemini-2.5-pro"))
    assert out == "ok-via-gemini:gemini-2.5-pro"
    assert calls == [("gemini", "gemini-2.5-pro")]


def test_fast_model_uses_each_legs_fast_variant(monkeypatch):
    legs = [
        {"name": "fake-fail", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m1", "fast_model": "m1f"},
        {"name": "fake-win", "kind": "openai-compat", "api_key": "x",
         "base_url": "https://x", "model": "m2", "fast_model": "m2f"},
    ]
    svc, calls = _svc_with_stubbed_legs(monkeypatch, legs)
    out = asyncio.run(svc.generate_text("hi", model="fast"))
    assert out == "ok-via-fake-win:m2f"
