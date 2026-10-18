"""Migration: legacy per-file storage -> scaled SQLite/FAISS/BM25 stack.

Usage:
    python scripts/migrate_to_scaled.py [--data-dir ./data] [--re-embed]

Handles two legacy layouts:

1. Legacy FAISS index (``data/vector_store/<name>.index``) + metadata JSON
   (``<name>_metadata.json``): embeddings are copied verbatim into SQLite - no
   re-embedding cost, the fastest path.
2. JSON document metadata + content files (``data/documents/{metadata,content}``):
   documents are re-chunked and re-embedded with the currently configured
   embedding model (only with --re-embed; otherwise documents are stored and
   marked for backfill).

After migration the scripts/benchmark_scale.py smoke benchmark or
library_stats MCP tool can verify the migrated library.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("migrate")


async def migrate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    config.config.DB_PATH = str(Path(args.db))
    config.config.DATA_DIR = str(data_dir)

    from services.sqlite_store import SQLiteStore, dumps_json
    from services.vector_store_service import VectorStoreService
    from services.document_store_service import DocumentStoreService
    from services.bm25_service import BM25Service
    from core.models import Document, DocumentType

    store = SQLiteStore(config.config.DB_PATH)
    vector_store = VectorStoreService(store=store, index_name=config.config.INDEX_NAME)
    document_store = DocumentStoreService(store=store)
    await store.initialize()
    await vector_store.initialize()  # creates empty tiers, no legacy import

    legacy_vector_dir = data_dir / "vector_store"
    legacy_doc_dir = data_dir / "documents"

    migrated_docs = migrated_chunks = 0

    # ------------------------------------------------------------- path 1
    legacy_index: Optional[Any] = None
    legacy_meta: Dict[str, Any] = {}
    import faiss  # migration always needs faiss

    index_path = legacy_vector_dir / f"{config.config.INDEX_NAME}.index"
    meta_path = legacy_vector_dir / f"{config.config.INDEX_NAME}_metadata.json"
    if index_path.exists() and meta_path.exists():
        logger.info("Legacy FAISS index found: %s", index_path)
        legacy_index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as fh:
            legacy_meta = json.load(fh)
        # metadata maps str(faiss_idx) -> chunk dict
        items = sorted(((int(k), v) for k, v in legacy_meta.items()))
        dim = legacy_index.d
        logger.info("Legacy index: %s vectors, dim=%s", legacy_index.ntotal, dim)
        batch_metas = []
        batch_vecs = []
        for faiss_idx, chunk in items:
            try:
                vector = legacy_index.reconstruct(int(faiss_idx)).tolist()
            except Exception:
                logger.warning("Skipping chunk %s - vector not reconstructable", faiss_idx)
                continue
            batch_metas.append(chunk)
            batch_vecs.append(vector)
            if len(batch_metas) >= 5000:
                migrated_chunks += await _copy_legacy_chunks(
                    vector_store, document_store, batch_metas, batch_vecs
                )
                batch_metas, batch_vecs = [], []
        if batch_metas:
            migrated_chunks += await _copy_legacy_chunks(
                vector_store, document_store, batch_metas, batch_vecs
            )
        migrated_docs = migrated_chunks  # one legacy chunk bundle per doc approximated below

    # ------------------------------------------------------------- path 2
    doc_meta_dir = legacy_doc_dir / "metadata"
    if doc_meta_dir.exists():
        doc_ids_seen = set()
        for meta_file in sorted(doc_meta_dir.glob("*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                logger.warning("Skipping unreadable metadata file %s", meta_file)
                continue
            doc_id = meta["id"]
            content_file = legacy_doc_dir / "content" / f"{doc_id}.txt"
            content = content_file.read_text(encoding="utf-8") if content_file.exists() else ""
            doc = Document(
                id=doc_id,
                filename=meta.get("filename", doc_id),
                content=content,
                doc_type=DocumentType(meta.get("doc_type", "txt")),
                file_size=meta.get("file_size", len(content.encode("utf-8"))),
                created_at=datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat())),
                metadata=meta.get("metadata", {}) or {},
                tags=meta.get("tags", []) or [],
                summary=meta.get("summary"),
                category=meta.get("category"),
                language=meta.get("language"),
            )
            await document_store.store_document(doc)
            migrated_docs += 1
            doc_ids_seen.add(doc_id)
            if args.re_embed and content:
                result = await _reembed_document(content, doc, vector_store)
                migrated_chunks += result
            elif content:
                # mark for backfill so ops can reindex without API keys
                await document_store.update_document_metadata(
                    doc_id, {"metadata": {"needs_backfill": True}}
                )
        _ = doc_ids_seen

    bm25 = BM25Service(store)
    await bm25.initialize()
    await bm25.rebuild()
    await vector_store.persist(force=True)

    logger.info(
        "Migration complete: %s documents, %s chunks (BM25 index frozen; will lazy-rebuild)",
        migrated_docs, migrated_chunks,
    )
    logger.info("Run `library_stats` via the MCP server or scripts/benchmark_scale.py to verify.")


async def _copy_legacy_chunks(
    vector_store, document_store, chunk_dicts: List[Dict[str, Any]], vectors: List[List[float]]
) -> int:
    from core.models import Chunk

    class _Chunk:  # lightweight stand-in with the attributes add_chunks reads
        def __init__(self, record: Dict[str, Any], embedding: List[float]):
            self.id = record.get("chunk_id")
            self.document_id = record.get("document_id")
            self.content = record.get("content", "")
            self.chunk_index = record.get("chunk_index", 0)
            self.start_pos = record.get("start_pos", 0)
            self.end_pos = record.get("end_pos", 0)
            self.metadata = record.get("metadata", {}) or {}
            self.embedding = embedding

    unique_docs = {c.get("document_id") for c in chunk_dicts}
    for doc_id in unique_docs:
        if doc_id and not await document_store.get_document(doc_id):
            # legacy bundles may not have document records - synthesize one
            from core.models import Document, DocumentType
            from datetime import datetime
            content_parts = [c.get("content", "") for c in chunk_dicts if c.get("document_id") == doc_id]
            await document_store.store_document(Document(
                id=doc_id,
                filename=f"migrated_{doc_id}.txt",
                content="\n".join(content_parts),
                doc_type=DocumentType.TEXT,
                file_size=sum(len(p) for p in content_parts),
                created_at=datetime.utcnow(),
                metadata={"migrated": True},
            ))
    chunks = [_Chunk(record, emb) for record, emb in zip(chunk_dicts, vectors)]
    ok = await vector_store.add_chunks(chunks)
    return len(chunks) if ok else 0


async def _reembed_document(content: str, doc, vector_store) -> int:
    from core.chunker import TextChunker
    from services.embedding_service import EmbeddingService

    chunker = TextChunker()
    chunks = chunker.optimize_chunks_for_embedding(
        chunker.chunk_document(doc.id, content, method="recursive")
    )
    if not chunks:
        return 0
    embedder = EmbeddingService()
    embeddings = await embedder.generate_embeddings([c.content for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    if await vector_store.add_chunks(chunks):
        return len(chunks)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./data", help="Legacy data directory")
    parser.add_argument("--db", default=None, help="Target SQLite DB path (default config.DB_PATH)")
    parser.add_argument("--re-embed", action="store_true", help="Re-embed documents from raw content")
    args = parser.parse_args()
    if args.db:
        args.db = str(Path(args.db))
    else:
        args.db = config.config.DB_PATH
    asyncio.run(migrate(args))


if __name__ == "__main__":
    main()
