"""Offline tests for the persistent ingestion queue (SQLite-backed worker).

Run: python -m pytest tests/test_ingest_queue.py -q
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic hash-based embedder (no API calls)."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    async def generate_embeddings(self, texts, batch_size=256, is_query=False):
        return [self._vector(t) for t in texts]

    async def generate_single_embedding(self, text, is_query=False):
        if text is None:
            return None
        return self._vector(text)

    def _vector(self, text):
        raw = text if isinstance(text, str) else getattr(text, "content", str(text))
        seed = abs(hash(raw)) % (2 ** 32)
        vec = np.random.default_rng(seed).standard_normal(self.dims).astype("float32")
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).tolist()

    def get_embedding_dimension(self):
        return self.dims

    def get_model_info(self):
        return {"provider": "fake", "dimensions": self.dims}


@pytest.fixture()
def env(monkeypatch):
    workdir = tempfile.mkdtemp(prefix="queue_test_")
    import config

    monkeypatch.setattr(config.config, "DB_PATH", os.path.join(workdir, "queue.db"))
    monkeypatch.setattr(config.config, "DATA_DIR", workdir)
    monkeypatch.setattr(config.config, "IVF_SWITCH_THRESHOLD", 10**12)
    monkeypatch.setattr(config.config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config.config, "INGEST_QUEUE_POLL_SECONDS", 0.2)
    return workdir


def _build(workdir):
    from services.sqlite_store import SQLiteStore
    from services.vector_store_service import VectorStoreService
    from services.document_store_service import DocumentStoreService
    from services.bm25_service import BM25Service
    from mcp_tools.ingestion_tool import IngestionTool
    from services.ingestion_queue_service import IngestionQueueService

    import config as _config

    store = SQLiteStore(os.path.join(workdir, "queue2.db"))
    vector_store = VectorStoreService(store=store)
    doc_store = DocumentStoreService(store=store)
    bm25 = BM25Service(store)
    vector_store.bm25_service = bm25
    ingestion = IngestionTool(vector_store, doc_store, FakeEmbedder(dims=64), None)
    queue = IngestionQueueService(ingestion_tool=ingestion, store=store)
    return store, doc_store, queue, _config.config


def _seed_directory(workdir: str, count: int = 2) -> str:
    files_dir = Path(workdir) / "seed_docs"
    files_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (files_dir / f"note_{i}.txt").write_text(
            f"Document number {i} about vector retention windows and archival policy.", "utf-8"
        )
    return str(files_dir)


def test_directory_job_end_to_end(env):
    async def flow():
        workdir = env
        store, doc_store, queue, cfg = _build(workdir)
        await store.initialize()
        await queue.initialize()

        target = _seed_directory(workdir, count=2)
        result = await queue.enqueue("directory", target, concurrency=2)
        assert result["success"]
        job_id = result["job_id"]

        job = await queue.get_job(job_id)
        assert job["status"] == "pending"

        claimed = await queue.claim_next()
        assert claimed is not None and claimed["id"] == job_id
        assert claimed["status"] == "running"

        # atomic claim must not hand out the same job twice
        assert await queue.claim_next() is None

        await queue.process_claimed(claimed)
        finished = await queue.get_job(job_id)
        assert finished["status"] == "done", finished.get("error")
        assert "result_summary" in (finished.get("extra") or {})

        stats = await doc_store.get_stats()
        assert stats["total_documents"] == 2
        assert await queue.claim_next() is None

    asyncio.run(flow())


def test_stale_running_jobs_requeued_on_boot(env):
    """A crash mid-job leaves status='running'; worker boot must requeue them."""
    async def flow():
        store, doc_store, queue, cfg = _build(env)
        await store.initialize()
        await queue.initialize()

        target = _seed_directory(workdir := env, count=1)
        result = await queue.enqueue("directory", target)
        job_id = result["job_id"]

        # simulate the crash: job claimed (running), then the process died
        claimed = await queue.claim_next()
        assert claimed["status"] == "running"

        requeued = await queue.recover_stale_running()
        assert requeued == 1
        job = await queue.get_job(job_id)
        assert job["status"] == "pending"

        # idempotent: a second sweep with no running rows is a no-op
        assert await queue.recover_stale_running() == 0

        # and the recovered job completes normally
        claimed_again = await queue.claim_next()
        await queue.process_claimed(claimed_again)
        finished = await queue.get_job(job_id)
        assert finished["status"] == "done"

    asyncio.run(flow())


def test_failed_job_records_error(env):
    async def flow():
        store, doc_store, queue, cfg = _build(env)
        await store.initialize()
        await queue.initialize()

        result = await queue.enqueue("directory", os.path.join(env, "does_not_exist"))
        job_id = result["job_id"]
        claimed = await queue.claim_next()
        assert claimed["id"] == job_id
        await queue.process_claimed(claimed)
        finished = await queue.get_job(job_id)
        assert finished["status"] == "failed"
        assert "does not exist" in finished["error"]

    asyncio.run(flow())


def test_worker_loop_drains_queue(env):
    async def flow():
        workdir = env
        store, doc_store, queue, cfg = _build(workdir)
        await store.initialize()
        await queue.initialize()

        target = _seed_directory(workdir, count=2)
        await queue.enqueue("directory", target, concurrency=2)
        await queue.enqueue("document", os.path.join(target, "note_0.txt"))

        # run the real worker loop until the queue is drained (bounded wait)
        worker = asyncio.create_task(queue.run_worker_loop())
        try:
            for _ in range(100):  # ~20s budget
                await asyncio.sleep(0.2)
                jobs = await queue.list_jobs(10)
                if all(j["status"] in ("done", "failed") for j in jobs):
                    break
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        jobs = await queue.list_jobs(10)
        statuses = {j["status"] for j in jobs}
        assert statuses <= {"done", "failed"}
        assert any(j["status"] == "done" for j in jobs)
        stats = await doc_store.get_stats()
        assert stats["total_documents"] >= 2

    asyncio.run(flow())
