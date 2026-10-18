"""Re-embed the entire library with the currently configured embedding model.

Run this whenever you switch EMBEDDING_MODEL (e.g. from OpenAI
text-embedding-3-small to a local model such as harrier-oss-v1-0.6b). It:

1. Backs up data/library.db (unless --no-backup).
2. Re-encodes every active chunk with the new model (no query prompt - the
   document side is always encoded plain).
3. Rewrites chunk_vectors BLOBs in batches.
4. Rebuilds the FAISS index from SQLite (compact(force=True)) under the new
   embedding dimension and persists it.

Usage (from the project root):
    python scripts/reembed_library.py            # confirm + backup + run
    python scripts/reembed_library.py --dry-run  # count + dimension check only
    python scripts/reembed_library.py --yes --no-backup
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from services.bootstrap import get_services  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reembed")

SELECT_BATCH = 2000


async def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed library with the configured embedding model")
    parser.add_argument("--batch-size", type=int, default=64, help="encode batch size (default 64)")
    parser.add_argument("--db-batch", type=int, default=2000, help="chunks fetched per DB page (default 2000)")
    parser.add_argument("--dry-run", action="store_true", help="count chunks + report dimensions only")
    parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    parser.add_argument("--no-backup", action="store_true", help="skip the library.db backup")
    args = parser.parse_args()

    cfg = config.config
    db_path = cfg.DB_PATH

    services = get_services()
    store = services.store
    embedding = services.embedding_service
    vector_store = services.vector_store
    await vector_store.initialize()

    total_row = await store.fetch_one("SELECT COUNT(*) AS c FROM chunks WHERE deleted = 0")
    total = int(total_row["c"]) if total_row else 0
    logger.info(
        "Model: %s (%s, dims=%s) | active chunks: %s | FAISS dimension: %s",
        embedding.model_name,
        "openai-api" if embedding.is_openai_model else "local",
        embedding.get_embedding_dimension(),
        total,
        vector_store._dimension,
    )
    if total == 0:
        logger.info("Nothing to re-embed.")
        return 0
    if args.dry_run:
        logger.info("Dry run - no changes made.")
        return 0

    if not args.yes:
        answer = input(f"Re-embed {total} chunks with '{embedding.model_name}'? [y/N] ").strip().lower()
        if answer != "y":
            logger.info("Aborted by user.")
            return 1

    if not args.no_backup and os.path.exists(db_path):
        backup = f"{db_path}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(db_path, backup)
        logger.info("Backup written: %s", backup)

    done = 0
    mismatches = 0
    last_id = 0
    while True:
        rows: List[Tuple[int, str]] = []
        page = await store.fetch_all(
            "SELECT c.vector_id AS vector_id, c.content AS content FROM chunks c"
            " WHERE c.vector_id > ? AND c.deleted = 0 ORDER BY c.vector_id LIMIT ?",
            (last_id, args.db_batch),
        )
        if not page:
            break
        for row in page:
            rows.append((row["vector_id"], row["content"]))
            last_id = row["vector_id"]

        texts = [content for _, content in rows]
        embeddings = await embedding.generate_embeddings(texts, batch_size=args.batch_size)
        updates = []
        for (vector_id, _), vec in zip(rows, embeddings):
            arr = np.asarray(vec, dtype="float32")
            updates.append((sqlite3.Binary(arr.tobytes()), vector_id))
        await store.execute_many("UPDATE chunk_vectors SET embedding = ? WHERE vector_id = ?", updates)
        done += len(rows)
        logger.info("Re-embedded %s/%s chunks", done, total)

    if done == 0:
        logger.warning("No vectors updated - aborting before index rebuild.")
        return 1

    new_dim = int(np.asarray(
        (await store.fetch_one(
            "SELECT embedding AS e FROM chunk_vectors LIMIT 1"))["e"],
        dtype="float32",
    ).size)
    if new_dim != vector_store._dimension:
        mismatches = new_dim
        logger.warning("Dimension changed %s -> %s; resizing FAISS index", vector_store._dimension, new_dim)
        vector_store._dimension = new_dim

    logger.info("Rebuilding FAISS index from SQLite ...")
    await vector_store.compact(force=True)
    await vector_store.persist(force=True)
    stats = await vector_store.get_stats()
    logger.info("Done. chunks=%s new_dim=%s faiss=%s totals=%s", done, new_dim, mismatches, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
