"""Sparse (BM25) retrieval leg backed by bm25s for hybrid search.

Why: dense embeddings understand *meaning* but miss *exact tokens* - product
codes, regulation numbers, error strings. BM25 rescues those. bm25s in-memory
indexing scales alongside our FAISS tier without an extra retrieval service.

Lifecycle: the on-disk index is rebuilt from SQLite chunks on a threshold
(``BM25_REBUILD_THRESHOLD`` new chunks) or at startup, keeping memory bounded;
it runs in an executor so the event loop is never blocked.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.sqlite_store import SQLiteStore
import config

logger = logging.getLogger(__name__)

try:
    import bm25s
    BM25_AVAILABLE = True
except Exception:  # pragma: no cover
    bm25s = None
    BM25_AVAILABLE = False


def _get_stemmer():
    """Prefer PyStemmer when installed; otherwise disable stemming."""
    try:
        import Stemmer  # optional PyStemmer extra
        return Stemmer.Stemmer("english")
    except Exception:
        return None


class BM25Service:
    def __init__(self, store: SQLiteStore):
        self.store = store
        self.config = config.config
        self._index: Optional[Any] = None
        self._doc_ids: List[int] = []
        self._rebuilding = False
        self._unsynced = 0
        self._disabled_reason: Optional[str] = None

        self._index_path = Path(self.config.DATA_DIR) / "bm25"
        self._index_path.mkdir(parents=True, exist_ok=True)

        if self.config.BM25_ENABLED and not BM25_AVAILABLE:
            self._disabled_reason = "bm25s package not installed"
            logger.warning("BM25_ENABLED=true but bm25s is missing - lexical leg disabled")

    @property
    def enabled(self) -> bool:
        return BM25_AVAILABLE and self.config.BM25_ENABLED

    def __repr__(self) -> str:  # helpful in logs
        return f"<BM25Service enabled={self.enabled} docs={len(self._doc_ids)}>"

    # ------------------------------------------------------------------ boot
    async def initialize(self) -> None:
        if not self.enabled or self._disabled_reason:
            return
        row = await self.store.fetch_one("SELECT COUNT(1) AS c FROM chunks WHERE deleted=0")
        count = int(row["c"]) if row else 0
        if count > self.config.BM25_MAX_CHUNKS:
            self._disabled_reason = (
                f"corpus ({count} chunks) exceeds BM25_MAX_CHUNKS ({self.config.BM25_MAX_CHUNKS})"
            )
            logger.warning("BM25 disabled: %s", self._disabled_reason)
            return
        if count:
            # try the persisted index first; fall back to a cold rebuild
            loaded = await self.load_from_disk()
            if not loaded:
                asyncio.create_task(self.rebuild())

    def _bm25(self) -> Any:
        return bm25s.BM25()

    # ------------------------------------------------------------- ingestion
    def mark_chunks_added(self, count: int) -> None:
        """Record that `count` chunks are not yet in the sparse index."""
        if not self.enabled or self._rebuilding:
            return
        self._unsynced += max(0, count)

    def needs_rebuild(self) -> bool:
        return self.enabled and self._unsynced >= self.config.BM25_REBUILD_THRESHOLD

    async def maybe_rebuild(self) -> Optional[Dict[str, Any]]:
        if self.needs_rebuild():
            return await self.rebuild()
        return None

    # ------------------------------------------------------------------ query
    def exists_index(self) -> bool:
        return self.enabled and self._index is not None and len(self._doc_ids) > 0

    async def query(
        self, text: str, top_k: int,
        excluded: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """Return [(vector_id, bm25_score)] ranked by relevance."""
        if not self.exists_index():
            return []
        loop = asyncio.get_running_loop()
        try:
            tokenized = await loop.run_in_executor(
                None,
                lambda: bm25s.tokenize(
                    [text], stopwords="en", stemmer=_get_stemmer(), show_progress=False
                ),
            )
        except Exception as exc:
            logger.error("BM25 tokenization failed: %s", exc)
            return []
        try:
            k = max(1, min(top_k, len(self._doc_ids)))
            results, scores = await loop.run_in_executor(
                None,
                lambda: self._index.retrieve(
                    tokenized, k=k, return_as="tuple", show_progress=False
                ),
            )
        except Exception as exc:
            logger.error("BM25 retrieve failed: %s", exc)
            return []

        out: List[Tuple[int, float]] = []
        for doc_index, score in zip(results[0], scores[0]):
            idx = int(doc_index)
            if idx < 0 or idx >= len(self._doc_ids):
                continue
            vid = self._doc_ids[idx]
            if excluded and vid in excluded:
                continue
            out.append((vid, float(score)))
        return out

    # ---------------------------------------------------------------- rebuild
    async def rebuild(self) -> Dict[str, Any]:
        """Full rebuild from SQLite chunks (streamed in pages, bounded RAM)."""
        if not self.enabled:
            return {"skipped": True, "reason": "disabled"}
        if self._disabled_reason:
            return {"skipped": True, "reason": self._disabled_reason}
        if self._rebuilding:
            return {"skipped": True, "reason": "already running"}
        self._rebuilding = True
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            doc_ids: List[int] = []
            texts: List[str] = []

            def _fetch(last: int, size: int):
                return self.store._connect().execute(
                    "SELECT vector_id, content FROM chunks"
                    " WHERE deleted=0 AND vector_id > ?"
                    " ORDER BY vector_id LIMIT ?",
                    (last, size),
                ).fetchall()

            last = 0
            page = 20000
            while True:
                rows = await loop.run_in_executor(None, _fetch, last, page)
                if not rows:
                    break
                doc_ids.extend(int(r["vector_id"]) for r in rows)
                texts.extend((r["content"] or "") for r in rows)
                last = int(rows[-1]["vector_id"])

            if not texts:
                logger.info("BM25 rebuild skipped: empty corpus")
                return {"total_chunks": 0}

            def _build():
                tokenized = bm25s.tokenize(
                    texts, stopwords="en", stemmer=_get_stemmer(), show_progress=False
                )
                index = self._bm25()
                index.index(tokenized, show_progress=False)
                return index

            index = await loop.run_in_executor(None, _build)
            self._index = index
            self._doc_ids = doc_ids
            self._unsynced = 0
            await loop.run_in_executor(None, self._persist_index)
            elapsed = loop.time() - started
            logger.info("BM25 index rebuilt over %s chunks in %.1fs", len(doc_ids), elapsed)
            return {"total_chunks": len(doc_ids), "duration_sec": round(elapsed, 2)}
        except Exception as exc:
            logger.exception("BM25 rebuild failed: %s", exc)
            return {"error": str(exc)}
        finally:
            self._rebuilding = False

    def _persist_index(self) -> None:
        try:
            os.makedirs(self._index_path, exist_ok=True)
            self._index.save(self._index_path)
            # bm25s persists token statistics only; the stable vector_id
            # mapping lives next to it so loads line up 1:1 with index rows.
            with open(self._index_path / "doc_ids.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(str(doc_id) for doc_id in self._doc_ids))
        except Exception as exc:
            logger.warning("BM25 persist failed (will rebuild on next start): %s", exc)

    async def load_from_disk(self) -> bool:
        if not self.enabled:
            return False
        loop = asyncio.get_running_loop()

        def _load():
            doc_ids_path = self._index_path / "doc_ids.txt"
            if not doc_ids_path.exists():
                return None, []
            if not any(self._index_path.glob("index*")):
                return None, []
            index = self._bm25().load(self._index_path)
            with open(doc_ids_path, "r", encoding="utf-8") as fh:
                ids = [int(line.strip()) for line in fh if line.strip().isdigit()]
            return index, ids

        try:
            index, doc_ids = await loop.run_in_executor(None, _load)
            if index is None or not doc_ids:
                return False
            self._index = index
            self._doc_ids = doc_ids
            logger.info("Loaded BM25 index covering %s chunks", len(doc_ids))
            return True
        except Exception as exc:
            logger.warning("BM25 disk load failed: %s", exc)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "disabled_reason": self._disabled_reason,
            "doc_count": len(self._doc_ids) if self._index else 0,
            "unsynced_chunks": self._unsynced,
            "rebuilding": self._rebuilding,
        }
