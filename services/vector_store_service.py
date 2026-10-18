"""FAISS-backed vector store rewritten for 1M+ documents.

Scaling profile
---------------
* <= IVF_SWITCH_THRESHOLD (default 250k) vectors -> exact ``IndexFlatIP``
  wrapped in ``IndexIDMap2`` (stable external ids == SQLite vector_ids).
* Beyond the threshold -> compacts into ``IndexIVFScalarQuantizer`` (SQ8,
  METRIC_INNER_PRODUCT) trained on a sample of the corpus.  SQ8 reduces RAM
  ~4x vs. float32 and IVF keeps search time near-constant with nprobe tuning
  (FAISS wiki guidance for 1M-10M vectors).
* Deletions are tombstones in SQLite; the index compacts (rebuilds) in the
  background once enough tombstones accumulate instead of freezing search.
* Embeddings live in SQLAlchemy-style BLOBs in SQLite -- the FAISS index can
  always be re-derived from them, so crash recovery is trivial.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from services.sqlite_store import SQLiteStore, dumps_json, loads_json
from core.models import SearchResult
import config

logger = logging.getLogger(__name__)

try:  # optional dependency for tests / tiny installs
    import faiss
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None
    FAISS_AVAILABLE = False


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _iter_in_batches(seq: Sequence[Any], batch: int):
    for i in range(0, len(seq), batch):
        yield seq[i:i + batch]


class VectorStoreService:
    """Auto-tiering FAISS vector store backed by SQLite."""

    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        index_name: Optional[str] = None,
    ):
        self.config = config.config
        self.store = store or SQLiteStore(self.config.DB_PATH)
        self.index_name = index_name or self.config.INDEX_NAME

        self.index: Any = None
        self.mode: str = self.config.VECTOR_INDEX_MODE  # auto|flat|ivf|hnsw
        self._tier: Optional[Any] = None
        self._persist_lock = asyncio.Lock()
        import itertools as _it
        self._persist_counter = _it.count()
        self._dimension: Optional[int] = None
        self._dirty_vectors = 0
        self._compacting = False
        self._last_persist = 0.0
        self._initialised = False

        self.index_path = Path(self.config.DATA_DIR) / "faiss" / f"{self.index_name}.faiss"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ boot
    async def initialize(self):
        """Load or rebuild the FAISS index. Idempotent."""
        self._initialised = True
        if not FAISS_AVAILABLE:
            logger.warning("faiss not installed - vector search disabled")
            return
        await self.store.initialize()

        active = await self._count_active()
        expected_dim = self._resolve_dimension()
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                self._dimension = int(index.d)
                index_class = index.__class__.__name__
                mode = self.mode
                if mode == "auto":
                    mode = "ivf" if index_class != "IndexIDMap2" else "flat"
                if mode == "ivf" and index_class == "IndexIDMap2":
                    inner = index.index
                    if inner.__class__.__name__ != "IndexFlatIP":
                        mode = "hnsw"
                self._tier = self._tier_from_index(index, mode, self._dimension, active)
                logger.info("Loaded FAISS index with %s vectors", index.ntotal)
            except Exception as exc:
                logger.warning("Failed to load index from disk (%s); rebuilding.", exc)
                self._tier = None
        if self._tier is None and active:
            # double-check dimension from persisted meta before a cold rebuild
            stored_dim = await self._stored_dimension()
            self._dimension = stored_dim or self._resolve_dimension()
            logger.info("Rebuilding index for %s existing vectors...", active)
            rebuild_kind = self.mode
            if self.mode == "auto":
                # tiny libraries must stay flat: IVF clustering cannot train
                # on fewer vectors than clusters (faiss 'nx >= k').
                rebuild_kind = "ivf" if active >= self.config.IVF_SWITCH_THRESHOLD else "flat"
            self._tier = self._make_tier(rebuild_kind, self._dimension, active)
            await self._build_from_sqlite()
            await self.persist(force=True)
        # auto-tier: if the persisted tier is flat but the corpus has outgrown
        # the threshold, upgrade in the background at startup
        if (
            self._tier is not None
            and self._tier.kind == "flat"
            and active >= self.config.IVF_SWITCH_THRESHOLD
            and self.mode == "auto"
        ):
            asyncio.create_task(self.compact(force=True))

    def _resolve_dimension(self) -> int:
        # Dimension drift is dangerous; reconcile with the loading index /
        # first embedding batch at runtime. Config acts as a hint only.
        return int(getattr(self.config, "EMBEDDING_DIMENSIONS", 0) or 0) or 384

    async def _stored_dimension(self) -> Optional[int]:
        """Read embedding dimension from SQLite, or None when unknown."""
        try:
            row = await self.store.fetch_one(
                "SELECT length(embedding) AS b FROM chunk_vectors LIMIT 1"
            )
            if row and row["b"]:
                return int(row["b"]) // 4  # float32
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ tiers
    class _Tier:
        kind = "flat"

        def add(self, ids: Sequence[int], vectors: np.ndarray) -> None: ...
        def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]: ...
        @property
        def ntotal(self) -> int: ...

    class _FlatTier(_Tier):
        kind = "flat"

        def __init__(self, dim: int):
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

        @property
        def ntotal(self) -> int:
            return int(self.index.ntotal)

        def add(self, ids, vectors):
            ids_arr = np.asarray(ids, dtype="int64")
            self.index.add_with_ids(vectors.astype("float32", copy=False), ids_arr)

        def search(self, query, top_k):
            return self.index.search(query, top_k)

        def to_disk(self, path: Path) -> None:
            faiss.write_index(self.index, str(path))

    class _IVFTier(_Tier):
        kind = "ivf"

        def __init__(self, dim: int, nlist: int, nprobe: int):
            quantizer = faiss.IndexFlatIP(dim)
            # 8-bit scalar quantizer: 4x memory saving, OK recall for RAG
            self.index = faiss.IndexIVFScalarQuantizer(
                quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.nprobe = nprobe
            self.trained = False

        @property
        def ntotal(self) -> int:
            return int(self.index.ntotal)

        def add(self, ids, vectors):
            if not self.trained:
                raise RuntimeError("IVF tier used before training")
            ids_arr = np.asarray(ids, dtype="int64")
            self.index.add_with_ids(vectors.astype("float32", copy=False), ids_arr)

        def search(self, query, top_k):
            return self.index.search(query, top_k)

        def to_disk(self, path: Path) -> None:
            faiss.write_index(self.index, str(path))

    class _HNswTier(_Tier):
        kind = "hnsw"

        def __init__(self, dim: int, m: int = 32):
            self.index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(dim, m))

        @property
        def ntotal(self) -> int:
            return int(self.index.ntotal)

        def add(self, ids, vectors):
            ids_arr = np.asarray(ids, dtype="int64")
            self.index.add_with_ids(vectors.astype("float32", copy=False), ids_arr)

        def search(self, query, top_k):
            return self.index.search(query, top_k)

        def to_disk(self, path: Path) -> None:
            faiss.write_index(self.index, str(path))

    def _make_tier(self, mode: str, dim: int, active: int):
        input_mode = mode
        if mode == "auto":
            mode = "flat" if active < self.config.IVF_SWITCH_THRESHOLD else "ivf"
        if mode == "ivf":
            nlist = self.config.IVF_NLIST or int(min(16384, max(1024, 4 * max(active, 1) ** 0.5)))
            tier = self._IVFTier(dim, nlist, self.config.IVF_NPROBE)
            logger.info("IVF tier: nlist=%s nprobe=%s", nlist, self.config.IVF_NPROBE)
            return tier
        if mode == "hnsw":
            return self._HNswTier(dim)
        if mode not in {"auto", "flat", "ivf", "hnsw"}:
            logger.warning("Unknown VECTOR_INDEX_MODE '%s' - falling back to flat", input_mode)
        return self._FlatTier(dim)

    def _tier_from_index(self, index, mode: str, dim: int, active: int):
        """Wrap an already-loaded FAISS index in the right tier (no data loss)."""
        name = index.__class__.__name__
        if name == "IndexIDMap2":
            inner = index.index.__class__.__name__
            if inner == "IndexHNSWFlat":
                tier = self._HNswTier(dim)
                tier.index = index
                return tier
            tier = self._FlatTier(dim)
            tier.index = index
            return tier
        if name == "IndexIVFScalarQuantizer":
            tier = self._IVFTier(dim, int(index.nlist), self.config.IVF_NPROBE)
            tier.index = index
            tier.trained = True
            return tier
        logger.warning("Unknown faiss index type on disk: %s - treating as flat", name)
        return self._FlatTier(dim)

    # ------------------------------------------------------------- ingestion
    async def add_chunks(self, chunks: List[Any]) -> bool:
        """Ingest a batch of chunk records (dicts or pydantic Chunk models)."""
        if not chunks or not FAISS_AVAILABLE:
            return not chunks
        if not self._initialised:
            await self.initialize()
        if self._tier is None and self._dimension is not None:
            # dimension known (e.g. loaded registry) but tier missing
            self._tier = self._make_tier(self.mode, self._dimension, 0)

        rows: List[Tuple[Any, ...]] = []
        vecs: List[np.ndarray] = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_record = chunk
            else:
                chunk_record = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding,
                }
            embedding = chunk_record.get("embedding")
            if embedding is None:
                continue
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            if not len(embedding):
                continue
            meta = chunk_record.get("metadata")
            rows.append((
                chunk_record["chunk_id"],
                chunk_record["document_id"],
                chunk_record["content"],
                int(chunk_record.get("chunk_index", 0)),
                int(chunk_record.get("start_pos", 0)),
                int(chunk_record.get("end_pos", 0)),
                dumps_json(meta or {}),
                chunk_record.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ))
            vecs.append([float(x) for x in embedding])

        if not rows:
            logger.warning("No valid embeddings in batch")
            return False

        # allocate stable vector ids first so faiss and sqlite agree
        def _insert():
            conn = self.store._connect()
            ids = []
            with conn:
                for row in rows:
                    cur = conn.execute(
                        "INSERT INTO chunks(chunk_id, document_id, content, chunk_index,"
                        " start_pos, end_pos, metadata, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?)", row,
                    )
                    ids.append(cur.lastrowid)
            embed_rows = [(vid, np.asarray(vecs[i], dtype="float32").tobytes()) for i, vid in enumerate(ids)]
            conn.executemany("INSERT OR REPLACE INTO chunk_vectors(vector_id, embedding) VALUES (?,?)", embed_rows)
            conn.commit()
            return ids
        vector_ids = await asyncio.get_running_loop().run_in_executor(None, _insert)

        vectors = _normalize(np.asarray(vecs, dtype="float32"))
        incoming_dim = int(vectors.shape[1])
        if self._tier is None:
            # First batch defines the dimension; build the tier now.
            self._dimension = incoming_dim
            self._tier = self._make_tier(self.mode, incoming_dim, await self._count_active())
        elif self._dimension is None:
            self._dimension = incoming_dim
        elif incoming_dim != self._dimension:
            logger.error(
                "Embedding dimension mismatch: index expects %s dims, batch has %s - "
                "batch rejected (clear_all + re-ingest required)",
                self._dimension, incoming_dim,
            )
            ids_tup = [(int(v),) for v in vector_ids]
            await self.store.execute_many(
                "DELETE FROM chunk_vectors WHERE vector_id = ?", ids_tup
            )
            await self.store.execute_many("DELETE FROM chunks WHERE vector_id = ?", ids_tup)
            return False


        # IVF needs training; until the corpus is large enough the tier stays
        # untrained and we simply return (rows already in SQLite, served once
        # training streams them in).
        tier = self._tier
        if tier.kind == "ivf" and not tier.trained:
            active = await self._count_active()
            if active >= max(self.config.IVF_NLIST or 1024, 1500) * 30:
                await self._train_ivf(tier)
                await self._maybe_persist()
                return True
            await self._maybe_persist()
            return True

        if tier.kind == "flat" and tier.ntotal >= self.config.IVF_SWITCH_THRESHOLD:
            logger.info("Threshold hit (%s vectors) - compacting to IVF/SQ8", self.config.IVF_SWITCH_THRESHOLD)
            await self.compact()
            # the trigger batch is already inside via the SQLite stream
            await self._maybe_persist()
            return True

        batch_ids: List[List[int]] = list(_iter_in_batches(vector_ids, max(1, self.config.FAISS_PERSIST_BATCH)))
        for bi, bv in zip(batch_ids, _iter_in_batches(list(vectors), max(1, self.config.FAISS_PERSIST_BATCH))):
            tier.add(bi, np.asarray(bv, dtype="float32"))

        self._dirty_vectors += len(vector_ids)
        await self._maybe_persist()
        logger.info("Added %s vectors (total=%s, mode=%s)", len(vector_ids), self._tier.ntotal, self._tier.kind)
        return True

    async def _train_ivf(self, tier: "_IVFTier") -> None:
        """Train + populate an IVF tier from SQLite in bounded RAM."""
        active = await self._count_active()
        if active < 64:
            # too few vectors to cluster (faiss 'nx >= k'): serve flat instead
            tier.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
            tier.trained = True
            tier.kind = "flat"
            async for row in self._stream_vectors():
                vec = np.frombuffer(row[1], dtype="float32").reshape(1, -1)
                vec = _normalize(vec)
                tier.add([row[0]], vec)
            logger.info("Only %s active vectors - IVF skipped, staying flat", active)
            return
        auto_nlist = self.config.IVF_NLIST or int(min(16384, max(1024, 4 * max(active, 1) ** 0.5)))
        # FAISS wants >= 30-39x nlist training points; shrink nlist to match
        # the sample we can actually provide (benign warning otherwise).
        nlist = min(auto_nlist, max(64, active // 39))
        tier.index = faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatIP(self._dimension), self._dimension, nlist,
            faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT,
        )
        tier.index.nprobe = min(self.config.IVF_NPROBE, nlist)

        sample_target = min(max(active, 1), nlist * 200, 200_000)
        sample_rows = await self._sample_vectors(sample_target)
        if not sample_rows:
            # keep a flat tier instead - IVF has nothing to train on
            tier.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
            tier.trained = True
            tier.kind = "flat"
            return

        train_matrix = _normalize(np.asarray(
            [np.frombuffer(r[1], dtype="float32") for r in sample_rows],
            dtype="float32",
        ))
        logger.info("Training IVF/SQ8 on %s sample vectors (nlist=%s)", len(sample_rows), nlist)
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: tier.index.train(train_matrix)
        )
        tier.trained = True

        inserted = 0
        async for row in self._stream_vectors():
            vec = np.frombuffer(row[1], dtype="float32").reshape(1, -1)
            vec = _normalize(vec)
            tier.add([row[0]], vec)
            inserted += 1
            if inserted % self.config.FAISS_PERSIST_BATCH == 0:
                logger.info("Compaction progress: %s/%s vectors added", inserted, active)

        await self.persist(force=True)
        logger.info("IVF compaction complete: %s vectors", tier.ntotal)

    async def _stream_vectors(self):
        """Async generator through (vector_id, embedding) pairs of active rows."""
        batch = 5000
        last_id = 0
        sql_active = (
            "SELECT c.vector_id AS vector_id, v.embedding AS embedding FROM chunks c"
            " JOIN chunk_vectors v ON v.vector_id = c.vector_id"
            " WHERE c.vector_id > ? AND c.deleted = 0 ORDER BY c.vector_id LIMIT ?"
        )
        while True:
            rows = await self.store.fetch_all(sql_active, (last_id, batch))
            if not rows:
                return
            for row in rows:
                yield (row["vector_id"], row["embedding"])
                last_id = row["vector_id"]
            if len(rows) < batch:
                return

    async def _sample_vectors(self, size: int) -> List[Tuple[int, bytes]]:
        sql = (
            "SELECT c.vector_id AS vector_id, v.embedding AS embedding FROM chunks c"
            " JOIN chunk_vectors v ON v.vector_id = c.vector_id"
            " WHERE c.deleted = 0 ORDER BY RANDOM() LIMIT ?"
        )
        rows = await self.store.fetch_all(sql, (size,))
        return [(r["vector_id"], r["embedding"]) for r in rows]

    # ---------------------------------------------------------------- search
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not FAISS_AVAILABLE or self._tier is None or self._tier.ntotal == 0:
            return []
        try:
            query_vec = np.asarray([query_embedding], dtype="float32")
            query_vec = _normalize(query_vec)

            overfetch = max(top_k * self.config.SEARCH_OVERFETCH, 64)
            cap = self.config.SEARCH_MAX_OVERFETCH
            escalation = 0
            while overfetch <= cap:
                scores, ids = await asyncio.get_running_loop().run_in_executor(
                    None, lambda k=min(overfetch, self._tier.ntotal): self._tier.search(query_vec, k)
                )
                candidates = [
                    (int(i), float(s))
                    for i, s in zip(ids[0].tolist(), scores[0].tolist())
                    if i != -1
                ]
                filtered = await self._filter_candidates(candidates, filters)
                if len(filtered) >= top_k or overfetch >= cap:
                    break
                overfetch = min(overfetch * 3, cap)
                escalation += 1
                if escalation > 2:
                    break

            results = [
                SearchResult(
                    chunk_id=row["chunk_id"] or str(row["vector_id"]),
                    document_id=row["document_id"],
                    content=row["content"],
                    score=score,
                    metadata={
                        **(loads_json(row["metadata"]) or {}),
                        # stable SQLite vector_id, used by the hybrid fusion
                        # layer without an extra round trip
                        "_vector_id": int(row["vector_id"]),
                    },
                )
                for row, score in filtered[:top_k]
            ]
            return results
        except Exception as exc:
            logger.exception("Vector search failed: %s", exc)
            return []

    async def _filter_candidates(
        self,
        candidates: Sequence[Tuple[int, float]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[Any, float]]:
        """Fetch metadata for candidate ids from SQLite and apply filters."""
        if not candidates:
            return []
        ids = [c[0] for c in candidates]
        id_slice = _iter_in_batches(ids, 900)
        rows_by_id: Dict[int, sqlite3.Row] = {}
        for part in id_slice:
            placeholders = ",".join(["?"] * len(part))
            rows = await self.store.fetch_all(
                f"SELECT vector_id, chunk_id, document_id, content, metadata, deleted"
                f" FROM chunks WHERE vector_id IN ({placeholders})",
                part,
            )
            for row in rows:
                rows_by_id[row["vector_id"]] = row

        result: List[Tuple[Any, float]] = []
        for vid, score in candidates:
            row = rows_by_id.get(vid)
            if row is None or row["deleted"]:
                continue
            if filters and not self._apply_filters(row, loads_json(row["metadata"]) or {}, filters):
                continue
            result.append((row, score))
        # ANN answers are unordered across tiers callers, so re-sort
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    @staticmethod
    def _apply_filters(row: sqlite3.Row, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "document_id":
                if row["document_id"] != value:
                    return False
            elif key == "document_ids":
                if row["document_id"] not in set(value):
                    return False
            elif key in ("content_length_min", "content_length_max"):
                length = len(row["content"] or "")
                if key == "content_length_min" and length < value:
                    return False
                if key == "content_length_max" and length > value:
                    return False
            elif key in ("category", "doc_type"):
                if meta.get(key) != value:
                    return False
            elif key == "tags":
                wanted = set(value if isinstance(value, list) else {value})
                if not wanted.intersection(set(meta.get("tags", []))):
                    return False
        return True

    # ------------------------------------------------------------- lifecycle
    async def _maybe_persist(self) -> None:
        if self._dirty_vectors >= self.config.FAISS_PERSIST_BATCH:
            await self.persist()

    async def persist(self, force: bool = False) -> None:
        if not FAISS_AVAILABLE or self._tier is None:
            return
        if not force and self._dirty_vectors < self.config.FAISS_PERSIST_BATCH:
            return
        if self._tier.kind == "ivf" and not self._tier.trained:
            return
        # Serialize: concurrent ingest tasks can both hit the checkpoint
        # threshold, and two writers on the same tmp file deadlock with a
        # Windows sharing violation (Permission denied).
        async with self._persist_lock:
            if not force and self._dirty_vectors < self.config.FAISS_PERSIST_BATCH:
                return
            if self._tier.kind == "ivf" and not self._tier.trained:
                return
            tmp_path = self.index_path.with_suffix(f".tmp{os.getpid()}.{next(self._persist_counter)}")

            def _write():
                self._tier.to_disk(tmp_path)
            await asyncio.get_running_loop().run_in_executor(None, _write)
            tmp_path.replace(self.index_path)
            self._dirty_vectors = 0
            self._last_persist = time.time()
            logger.info("Persisted FAISS index (%s vectors) -> %s", self._tier.ntotal, self.index_path)

    async def _count_active(self) -> int:
        try:
            row = await self.store.fetch_one("SELECT COUNT(*) AS c FROM chunks WHERE deleted=0")
            return int(row["c"]) if row else 0
        except Exception:
            return 0

    async def _build_from_sqlite(self) -> None:
        """Full rebuild from SQLite (used at boot for crash recovery)."""
        active = await self._count_active()
        if self._tier.kind == "ivf":
            await self._train_ivf(self._tier)
            return
        # flat tier
        total_added = 0
        async for chunk in self._stream_vectors():
            vec = np.frombuffer(chunk[1], dtype="float32").reshape(1, -1)
            vec = _normalize(vec)
            self._tier.add([chunk[0]], vec)
            total_added += 1
            if total_added % 100_000 == 0:
                logger.info("Rebuild progress: %s/%s", total_added, active)

    async def compact(self, force: bool = False) -> bool:
        """Rebuild the index, dropping tombstones, and re-tier if needed."""
        if self._compacting:
            return False
        self._compacting = True
        try:
            active = await self._count_active()
            should_tier = (
                self.mode == "auto" and active >= self.config.IVF_SWITCH_THRESHOLD
            )
            new_mode = "ivf" if should_tier else self._tier.kind if self._tier else "flat"
            new_mode = new_mode if new_mode != "auto" else self.mode
            if self._tier is None:
                return False
            if self._tier.kind == new_mode and not force:
                return False

            new_tier = self._make_tier(new_mode, self._dimension, active)
            if new_tier.kind == "ivf":
                await self._train_ivf(new_tier)
            else:
                added = 0
                async for row in self._stream_vectors():
                    vec = np.frombuffer(row[1], dtype="float32").reshape(1, -1)
                    vec = _normalize(vec)
                    new_tier.add([row[0]], vec)
                    added += 1
                logger.info("Flat compaction added %s active vectors", added)

            await self.persist(force=True)
            self._tier = new_tier
            self.mode = new_tier.kind
            self._dirty_vectors = 0
            return True
        finally:
            self._compacting = False

    async def delete_document(self, document_id: str) -> bool:
        if not self._initialised:
            await self.initialize()
        row = await self.store.fetch_one(
            "SELECT COUNT(*) AS c FROM chunks WHERE document_id=? AND deleted=0", (document_id,)
        )
        count = int(row["c"]) if row else 0
        if not count:
            return False
        await self.store.execute(
            "UPDATE chunks SET deleted=1 WHERE document_id=?", (document_id,)
        )
        tombstones = await self._count_tombstones()
        if tombstones >= self.config.COMPACTION_TOMBSTONE_THRESHOLD:
            asyncio.create_task(self.compact())
        logger.info("Tombstoned %s chunks for document %s", count, document_id)
        return True

    async def _count_tombstones(self) -> int:
        row = await self.store.fetch_one("SELECT COUNT(*) AS c FROM chunks WHERE deleted=1")
        return int(row["c"]) if row else 0

    async def clear_all(self) -> bool:
        await self.store.script(["DELETE FROM chunks", "DELETE FROM chunk_vectors"])
        if self.index_path.exists():
            try:
                self.index_path.unlink()
            except Exception:
                pass
        self._tier = None
        self._dimension = None
        return True

    async def get_stats(self) -> Dict[str, Any]:
        active = await self._count_active()
        tombstones = await self._count_tombstones()
        return {
            "total_vectors": self._tier.ntotal if self._tier else 0,
            "active_vectors": active,
            "tombstones": tombstones,
            "dimension": self._dimension,
            "index_mode": self._tier.kind if self._tier else self.mode,
            "faiss_available": FAISS_AVAILABLE,
            "index_file_exists": self.index_path.exists(),
            "dirty_vectors": self._dirty_vectors,
        }
