"""Free-tier LLM provider chain.

Every leg here is OpenAI-compatible REST (Gemini's compat endpoint, Groq,
OpenRouter) except Mistral, which keeps its own SDK. First usable leg serves
requests; a leg that answers with a permanent error (quota exhausted, bad
key) is skipped for the rest of the process.
"""
import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def resolve_llm_providers(cfg: Any = None) -> List[Dict[str, str]]:
    """Ordered list of configured legs: [{name, api_key, base_url, model, fast_model, kind}]."""
    cfg = cfg or config.config
    chain = [s.strip().lower() for s in (cfg.LLM_PROVIDER_CHAIN or "").split(",") if s.strip()]
    registry: Dict[str, Dict[str, str]] = {
        "gemini": {
            "name": "gemini",
            "kind": "openai-compat",
            "api_key": cfg.GEMINI_API_KEY or "",
            "base_url": cfg.GEMINI_BASE_URL,
            "model": cfg.GEMINI_MODEL,
            "fast_model": cfg.GEMINI_FAST_MODEL,
        },
        "groq": {
            "name": "groq",
            "kind": "openai-compat",
            "api_key": cfg.GROQ_API_KEY or "",
            "base_url": cfg.GROQ_BASE_URL,
            "model": cfg.GROQ_MODEL,
            "fast_model": cfg.GROQ_FAST_MODEL,
        },
        "openrouter": {
            "name": "openrouter",
            "kind": "openai-compat",
            "api_key": cfg.OPENROUTER_API_KEY or "",
            "base_url": cfg.OPENROUTER_BASE_URL,
            "model": cfg.OPENROUTER_MODEL,
            "fast_model": cfg.OPENROUTER_FAST_MODEL,
        },
        "nebius": {
            "name": "nebius",
            "kind": "openai-compat",
            "api_key": cfg.NEBIUS_API_KEY or "",
            "base_url": cfg.NEBIUS_BASE_URL,
            "model": cfg.NEBIUS_MODEL,
            "fast_model": cfg.NEBIUS_MODEL,
        },
        "openai": {
            "name": "openai",
            "kind": "openai-compat",
            "api_key": cfg.OPENAI_API_KEY or "",
            "base_url": "https://api.openai.com/v1",
            "model": cfg.OPENAI_MODEL,
            "fast_model": cfg.FAST_MODEL,
        },
        # Mistral uses its own SDK - handled specially in LLMService.
        "mistral": {
            "name": "mistral",
            "kind": "sdk-mistral",
            "api_key": cfg.MISTRAL_API_KEY or "",
            "base_url": "",
            "model": cfg.MISTRAL_MODEL,
            "fast_model": cfg.MISTRAL_MODEL,
        },
    }
    legs: List[Dict[str, str]] = []
    for name in chain:
        spec = registry.get(name)
        if not spec:
            logger.warning("Unknown provider '%s' in LLM_PROVIDER_CHAIN - ignored", name)
            continue
        if not spec["api_key"]:
            continue
        legs.append(dict(spec))
    if not legs:
        logger.warning("LLM_PROVIDER_CHAIN resolved to ZERO usable legs - check API keys")
    return legs


# Prefixes used for explicit model selection, e.g. "groq/openai/gpt-oss-120b".
PROVIDER_PREFIXES = {
    "gemini": ("gemini/", "google/"),
    "groq": ("groq/",),
    "openrouter": ("openrouter/",),
    "nebius": ("nebius/", "meta-llama/"),
    "openai": ("openai/", "gpt-", "o1-", "o3-"),
    "mistral": ("mistral",),
}


def leg_for_model(model: str) -> Optional[str]:
    """Leg name that owns an explicit model string (e.g. 'groq/openai/gpt-oss-120b')."""
    lowered = model.lower()
    for leg, prefixes in PROVIDER_PREFIXES.items():
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return leg
    return None


def pick_llm_provider(cfg: Any = None) -> Optional[Dict[str, str]]:
    """First OpenAI-compatible leg with a key (for LlamaIndex Settings.llm)."""
    for leg in resolve_llm_providers(cfg):
        if leg["kind"] == "openai-compat":
            return leg
    return None


def permanent_llm_error(exc: Exception) -> bool:
    """True for rejections that never succeed on retry (quota/auth)."""
    status = str(getattr(exc, "status_code", "") or getattr(exc, "http_status", "") or "")
    if status in ("401", "402", "403", "429"):
        return True
    from services.embedding_service import _QUOTA_MARKERS

    text = str(exc)
    return any(marker in text for marker in _QUOTA_MARKERS)
