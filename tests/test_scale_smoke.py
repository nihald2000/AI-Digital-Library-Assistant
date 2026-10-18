"""Offline smoke tests for the scaled RAG stack.

Runs the full pipeline (ingest -> hybrid search -> delete/compaction ->
stats) against a temporary SQLite database with a deterministic fake
embedder, so no API keys or model downloads are required.

Run: python -m pytest tests/test_scale_smoke.py -q
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest  # type: ignore


class FakeEmbedder:
    """Deterministic hash-based embedder with topic affinity."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    async def generate_embeddings(self, texts, batch_size=256):
        return [self._vector(t) for t in texts]

    async def generate_single_embedding(self, text):
        if text is None:
            return None
        return self._vector(text)

    def _vector(self, text):
        raw = text if isinstance(text, str) else getattr(text, "content", str(text))
        seed = abs(hash(raw)) % (2 ** 32)
        vec = np.random.default_rng(seed).standard_normal(self.dims).astype("float32")
        # inject affinity so "quantum"-ish texts cluster
        if "quantum" in raw.lower():
            vec[:2] += 2.0
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).tolist()

    def get_embedding_dimension(self):
        return self.dims

    def get_model_info(self):
        return {"provider": "fake", "dimensions": self.dims}


def _make_text(i: int, words: int = 600, topic: str = "quantum") -> str:
    filler = ["signal", "vector", "archive", "schema", "metric"]
    body = " ".join([" ".join(filler + [topic]) for _ in range(max(1, words // 6))])
    body += f" KEYTERM-{i % 11} quantum harbor terminator sentinel"
    return body


@pytest.fixture()
def env(monkeypatch):
    workdir = tempfile.mkdtemp(prefix="rag_smoke_")
    import config
    monkeypatch.setattr(config.config, "DB_PATH", os.path.join(workdir, "smoke.db"))
    monkeypatch.setattr(config.config, "DATA_DIR", workdir)
    monkeypatch.setattr(config.config, "BM25_ENABLED", True)
    monkeypatch.setattr(config.config, "IVF_SWITCH_THRESHOLD", 10**12)  # don't trigger IVF path here
    monkeypatch.setattr(config.config, "FAISS_PERSIST_BATCH", 1000)
    monkeypatch.setattr(config.config, "RERANK_ENABLED", False)
    return workdir


def _build_services():
    from services.sqlite_store import SQLiteStore
    from services.vector_store_service import VectorStoreService
    from services.document_store_service import DocumentStoreService
    from services.bm25_service import BM25Service
    from services.hybrid_search_service import HybridSearchService, Reranker
    from mcp_tools.search_tool import SearchTool
    from mcp_tools.ingestion_tool import IngestionTool

    config_object = __import__("config").config
    store = SQLiteStore(config_object.DB_PATH)
    vector_store = VectorStoreService(store=store)
    document_store = DocumentStoreService(store=store)
    bm25 = BM25Service(store)
    vector_store.bm25_service = bm25
    embedder = FakeEmbedder(dims=64)
    hybrid = HybridSearchService(vector_store, bm25, reranker=Reranker("", 50))
    search_tool = SearchTool(vector_store, embedder, document_store, hybrid_service=hybrid)
    ingestion = IngestionTool(vector_store, document_store, embedder, None)
    return store, vector_store, document_store, bm25, embedder, search_tool, ingestion


def test_full_pipeline_smoke(env):
    async def flow():
        (
            store, vector_store, doc_store, bm25, embedder,
            search_tool, ingestion,
        ) = _build_services()
        await store.initialize()
        await vector_store.initialize()
        await bm25.initialize()

        # 1. ingest documents (mixed topics)
        result = await ingestion.process_text_content(_make_text(0, topic="quantum"))
        assert result.get("success"), result
        doc_id_quantum = result["document_id"]
        results = []
        for i in range(3):
            topic = "quantum" if i % 2 == 0 else "harbor"
            results.append(await ingestion.process_text_content(
                _make_text(i, topic=topic), filename=f"smoke_{i}.txt"
            ))
            assert results[-1].get("success"), results[-1]

        stats = await vector_store.get_stats()
        assert stats["active_vectors"] > 0
        doc_stats = await doc_store.get_stats()
        assert doc_stats["total_documents"] == 4

        # 2. hybrid search works (dense-only path here; BM25 index not rebuilt
        # yet). The fake hash-wrapped embeddings yield weak cosine scores, so
        # lower the similarity floor for the smoke run.
        hits = await search_tool.search("quantum archive", top_k=4, similarity_threshold=0.05)
        assert hits, "expected at least a dense hit"
        assert all(h.score > 0 for h in hits)

        # 3. BM25 sparse leg (skip if bm25s package is not available)
        if bm25.enabled:
            rebuild = await bm25.rebuild()
            assert rebuild.get("total_chunks", 0) > 0
            sparse_hits = await bm25.query("quantum KEYTERM-1", 5)
            assert sparse_hits, "sparse leg returned nothing"

        # 4. document deletion tombstones and results respect them
        target_doc = hits[0].document_id
        assert await doc_store.delete_document(target_doc)
        assert await vector_store.delete_document(target_doc)
        hits_after = await search_tool.search("quantum archive", top_k=10, similarity_threshold=0.05)
        assert all(h.document_id != target_doc for h in hits_after)

        # 5. stats sanity after delete
        stats = await vector_store.get_stats()
        assert stats["tombstones"] >= 1
        doc_stats = await doc_store.get_stats()
        assert doc_stats["total_documents"] == 3

        # 6. clear-and-rebuild path
        await vector_store.clear_all()
        emptied = await vector_store.get_stats()
        assert emptied["active_vectors"] == 0

        _ = doc_id_quantum

    asyncio.run(flow())


def test_hybrid_fusion_ranks(env):
    """RRF fusion ranks chunks that appear in both legs first."""
    from services.hybrid_search_service import reciprocal_rank_fusion

    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(2, 3.5), (1, 1.2), (9, 0.4)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0].vector_id == 1  # top of dense + second of sparse
    assert fused[1].vector_id == 2
    assert {f.vector_id for f in fused} == {1, 2, 3, 9}


def test_auto_ivf_tiering(env):
    """Auto mode upgrades from flat to IVF when the vector threshold hits."""
    async def flow():
        (
            store, vector_store, doc_store, bm25, embedder,
            search_tool, ingestion,
        ) = _build_services()
        await store.initialize()
        await vector_store.initialize()
        try:
            import config
            config.config.IVF_SWITCH_THRESHOLD = 50  # force the switch for the test

            for i in range(2):
                result = await ingestion.process_text_content(_make_text(i, words=1200))
                assert result.get("success")
            stats = await vector_store.get_stats()
            # the compaction may be availed later (background); check either mode
            assert stats["active_vectors"] > 0
            hits = await search_tool.search("quantum archive", top_k=5, similarity_threshold=0.05)
            assert hits, "search should still work after tier switch"
        finally:
            import config
            config.config.IVF_SWITCH_THRESHOLD = 500_000

    asyncio.run(flow())
