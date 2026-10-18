"""Embedding-leg resilience: permanent OpenAI failures switch to the local model."""
import asyncio

import pytest

import config
import services.embedding_service as es
from services.embedding_service import EmbeddingService, OpenAIEmbeddingsUnavailable


@pytest.fixture()
def openai_legged_service(monkeypatch):
    """EmbeddingService forced onto the OpenAI leg (as if a dead key existed)."""
    monkeypatch.setattr(config.config, "OPENAI_API_KEY", "sk-dead-key-test")
    svc = EmbeddingService()
    assert svc.is_openai_model
    return svc


def test_permanent_quota_failure_falls_back_local(openai_legged_service, monkeypatch):
    calls = {"n": 0}

    async def _quota_bomb(self, index_to_text, batch_size):
        calls["n"] += 1
        raise OpenAIEmbeddingsUnavailable("429 credit_balance_exhausted")

    monkeypatch.setattr(EmbeddingService, "_embed_openai", _quota_bomb)
    out = asyncio.run(openai_legged_service.generate_embeddings(["quantum computing paper"]))
    assert calls["n"] == 1, "should not retry a quota error five times"
    assert not openai_legged_service.is_openai_model
    assert openai_legged_service.model_name == config.config.EMBEDDING_LOCAL_MODEL
    assert len(out[0]) > 0


def test_transient_errors_still_graceful(openai_legged_service):
    # non-quota exceptions keep the openai leg + partial results behavior
    assert not EmbeddingService._permanent_openai_failure(RuntimeError("some timeout"))

    class FakeAuthError(Exception):
        status_code = 401
        body = {"code": "invalid_api_key"}

    class FakeQuotaError(Exception):
        status_code = 429
        body = {"code": "insufficient_quota"}

    assert EmbeddingService._permanent_openai_failure(FakeAuthError("bad"))
    assert EmbeddingService._permanent_openai_failure(FakeQuotaError("dead"))
    assert EmbeddingService._permanent_openai_failure(
        es.OpenAIEmbeddingsUnavailable("429 credit_balance_exhausted")
    )
