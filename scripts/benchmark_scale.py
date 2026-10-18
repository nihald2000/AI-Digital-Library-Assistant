"""Scaling benchmark for the RAG pipeline.

Benchmarks ingestion throughput and hybrid search latency against a synthetic
corpus at configurable scale. Uses a deterministic hash-based fake embedder
(``--embedder fake``) so the run exercises the full SQLite/FAISS/BM25 stack
without network or model downloads - CI friendly.

Usage:
    python scripts/benchmark_scale.py --docs 1000 --embedder fake
    python scripts/benchmark_scale.py --docs 20000 --embedder fake --search 200
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

logging.basicConfig(level=logging.WARNING)


class FakeEmbedder:
    """Deterministic, no-network embedder for scale testing."""

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    async def generate_embeddings(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    async def generate_single_embedding(self, text) -> List[float]:
        return self._vector(text)

    def _vector(self, text) -> List[float]:
        if hasattr(text, "content"):
            text = text.content
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        vec = rng.standard_normal(self.dimensions).astype("float32")
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def get_embedding_dimension(self):
        return self.dimensions

    def get_model_info(self):
        return {"provider": "fake", "dimensions": self.dimensions}


def make_document(index: int, words: int = 1500) -> str:
    """A synthetic 1500-word document with topical vocabulary per doc."""
    topics = ["quantum", "library", "ledger", "beetle", "harbor", "orbit", "canyon", "ember", "orchid"]
    sentences = []
    topic = topics[index % len(topics)]
    filler = [
        "quantum", "library", "ledger", "beetle", "harbor", "orbit", "canyon", "ember",
        "orchid", "signal", "metric", "protocol", "archive", "vector", "citation",
        "theorem", "archive", "schema", "vector", "journal", "ledger",
    ]
    rng = random.Random(index)
    for _ in range(words * 6):
        i = rng.randrange(0, len(filler))
        sentences.append(filler[i])
        if rng.random() < 0.02:
            sentences.append(topic)
        if rng.random() < 0.05:
            sentences.append(".")
    body = " ".join(sentences)
    # guarantee keyword hits for BM25
    body += f" KEYWORDTOKEN-{index % 17} "
    return body


async def run(args: argparse.Namespace) -> None:
    import tempfile
    import config
    from services.sqlite_store import SQLiteStore
    from services.vector_store_service import VectorStoreService
    from services.document_store_service import DocumentStoreService
    from services.bm25_service import BM25Service
    from services.hybrid_search_service import HybridSearchService, Reranker, FusionResult
    from core.models import Document, DocumentType

    workdir = tempfile.mkdtemp(prefix="rag_bench_")
    db_path = os.path.join(workdir, "bench.db")
    config.config.DB_PATH = db_path
    config.config.DATA_DIR = workdir
    config.config.IVF_SWITCH_THRESHOLD = args.ivf_threshold
    config.config.FAISS_PERSIST_BATCH = args.persist_batch
    config.config.BM25_ENABLED = not args.no_bm25

    embedder = FakeEmbedder(dimensions=args.dims)

    store = SQLiteStore(db_path)
    vector_store = VectorStoreService(store=store)
    document_store = DocumentStoreService(store=store)
    bm25 = BM25Service(store)
    vector_store.bm25_service = bm25
    hybrid = HybridSearchService(vector_store, bm25, reranker=Reranker("", 50))
    await store.initialize()
    await vector_store.initialize()
    await bm25.initialize()

    # monkey-patch the fake embedder into the pipeline
    from core.chunker import TextChunker

    from mcp_tools.ingestion_tool import IngestionTool

    tool = IngestionTool(vector_store, document_store, embedder, None)

    print(f"[bench] corpus={args.docs} docs (~{args.words} words each) | dims={args.dims} | "
          f"ivf_threshold={args.ivf_threshold} | bm25={'on' if config.config.BM25_ENABLED else 'off'}")

    ingest_start = time.perf_counter()
    per_doc_stats = []
    chunker = TextChunker()

    from datetime import datetime

    def build_doc(i: int) -> Document:
        content = make_document(i, words=args.words)
        return Document(
            id=f"doc_{i}", filename=f"bench_{i}.txt", content=content,
            doc_type=DocumentType.TEXT, file_size=len(content.encode()),
            created_at=datetime.utcnow(),
        )

    # concurrency-bounded ingest (mirrors the production pipeline)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def ingest_one(i: int):
        async with semaphore:
            doc = build_doc(i)
            await document_store.store_document(doc)
            chunks = chunker.optimize_chunks_for_embedding(
                chunker.chunk_document(doc.id, doc.content, method="recursive")
            )
            if not chunks:
                return
            embeds = await embedder.generate_embeddings([c.content for c in chunks])
            for chunk, emb in zip(chunks, embeds):
                chunk.embedding = emb
            batch = 512
            for start in range(0, len(chunks), batch):
                await vector_store.add_chunks(chunks[start:start + batch])
            bm25.mark_chunks_added(len(chunks))
            await document_store.update_document_metadata(doc.id, {"chunk_count": len(chunks)})

    tasks = [asyncio.create_task(ingest_one(i)) for i in range(args.docs)]
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if done % max(1, args.docs // 10) == 0:
            elapsed = time.perf_counter() - ingest_start
            print(f"[bench] {done}/{args.docs} docs | {elapsed:.1f}s elapsed")

    ingest_elapsed = time.perf_counter() - ingest_start
    total_chunks = (await vector_store.get_stats())["active_vectors"]
    print(f"[bench] ingest: {ingest_elapsed:.1f}s total, "
          f"{args.docs / max(ingest_elapsed, 1e-9):.0f} docs/s, "
          f"{total_chunks / max(ingest_elapsed, 1e-9):.0f} chunks/s")

    await vector_store.persist(force=True)
    await bm25.rebuild()

    # ------------------------------------------------------------ search bench
    from mcp_tools.search_tool import SearchTool

    search_tool = SearchTool(vector_store, embedder, document_store, hybrid_service=hybrid)

    latencies: List[float] = []
    result_counts: List[int] = []
    for i in range(args.search):
        query = f"topic {['quantum', 'library', 'ledger'][i % 3]} KEYWORDTOKEN-{i % 17}"
        started = time.perf_counter()
        results = await search_tool.search(query, top_k=10, similarity_threshold=0.0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        result_counts.append(len(results))

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    print(f"[bench] hybrid search p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms "
          f"(n={args.search}, avg_hits={statistics.mean(result_counts):.1f})")

    exact_start = time.perf_counter()
    results = await search_tool.search("KEYWORDTOKEN-7", top_k=5, similarity_threshold=0.0)
    exact_ms = (time.perf_counter() - exact_start) * 1000
    print(f"[bench] single query spot-check: {exact_ms:.1f}ms -> {len(results)} results")
    if results:
        print(f"[bench] top result doc: {results[0].document_id} score={results[0].score:.3f}")
    print(f"[bench] workdir={workdir} (SQLite db {os.path.getsize(db_path) // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=1000)
    parser.add_argument("--words", type=int, default=1500)
    parser.add_argument("--dims", type=int, default=512)
    parser.add_argument("--ivf-threshold", type=int, default=250_000)
    parser.add_argument("--persist-batch", type=int, default=25_000)
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--search", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
