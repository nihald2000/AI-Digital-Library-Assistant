"""Hybrid retrieval fusion: dense + sparse -> Reciprocal Rank Fusion -> optional rerank.

Research basis (see SCALING.md for citations):
* Hybrid fusion consistently outperforms either single leg (+7% NDCG on
  e-commerce benchmarks, +8pp Recall@5 on text+table finance docs).
* A cross-encoder rerank stage on the fused candidate pool adds another
  +12-17pp MRR/Recall on production RAG benchmarks.

The service takes per-leg results as (id, score) pairs and returns an ordered
fused list. RRF is scale-free (uses only ranks) which is why we prefer it
over score-based weighted fusion: FAISS cosine similarities and BM25 scores
live on incomparable scales.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    vector_id: int
    fused_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


def reciprocal_rank_fusion(
    dense: Sequence[Tuple[int, float]],
    sparse: Sequence[Tuple[int, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[FusionResult]:
    """Fuse the two ranked lists. Higher fused score = more relevant."""
    scores: Dict[int, FusionResult] = {}

    for rank, (vid, raw) in enumerate(dense):
        entry = scores.setdefault(vid, FusionResult(vector_id=vid, fused_score=0.0))
        entry.dense_rank = rank
        entry.dense_score = raw
        entry.fused_score += dense_weight / (k + rank + 1)

    for rank, (vid, raw) in enumerate(sparse):
        entry = scores.setdefault(vid, FusionResult(vector_id=vid, fused_score=0.0))
        entry.sparse_rank = rank
        entry.sparse_score = raw
        entry.fused_score += sparse_weight / (k + rank + 1)

    return sorted(scores.values(), key=lambda e: e.fused_score, reverse=True)


class Reranker:
    """Optional cross-encoder reranking of the fused candidate pool."""

    def __init__(self, model_name: str, candidate_pool: int):
        self.model_name = model_name
        self.pool = max(10, candidate_pool)
        self._model: Any = None
        self._loading = False
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed:
            return
        self._loading = True
        try:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("Reranker '%s' unavailable (%s) - skipping rerank", self.model_name, exc)
            self._failed = True
        finally:
            self._loading = False

    @property
    def ready(self) -> bool:
        return not self._failed and config.config.RERANK_ENABLED

    async def rerank(
        self, query: str, candidates: List[FusionResult],
        content_lookup: Dict[int, str],
    ) -> List[FusionResult]:
        """Return candidates re-scored by the cross-encoder."""
        if not self.ready or not candidates:
            return candidates
        pool = candidates[: self.pool]
        if self._model is None:
            await asyncio.get_running_loop().run_in_executor(None, self._load)
            if self._model is None:
                return candidates
        pairs = [
            (query, content_lookup.get(c.vector_id, "")) for c in pool
        ]
        try:
            scores = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._model.predict(pairs, show_progress_bar=False)
            )
        except Exception as exc:
            logger.warning("Rerank prediction failed: %s", exc)
            return candidates
        reranked = sorted(
            zip(pool, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        rest = candidates[self.pool:]
        return [c for c, _ in reranked] + rest


class HybridSearchService:
    """Orchestrates dense + sparse fans and the fusion/rerank stages."""

    def __init__(
        self,
        vector_search: Any,
        bm25_service: Any,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_search = vector_search
        self.bm25 = bm25_service
        self.config = config.config
        self.reranker = reranker or Reranker(
            self.config.RERANK_MODEL, self.config.RERANK_CANDIDATE_POOL
        )

    async def hydrate_candidates(
        self, fused: List[FusionResult]
    ) -> List[Tuple[FusionResult, Dict[str, Any]]]:
        """Pull chunk rows for the fused ids (SQLite batched read)."""
        if not fused:
            return []
        lookup: Dict[int, Dict[str, Any]] = {}
        ids = [f.vector_id for f in fused]
        batch = 900
        for start in range(0, len(ids), batch):
            part = ids[start:start + batch]
            placeholders = ",".join("?" for _ in part)
            rows = await self.vector_search.store.fetch_all(
                f"SELECT vector_id, chunk_id, document_id, content, metadata"
                f" FROM chunks WHERE vector_id IN ({placeholders}) AND deleted=0",
                part,
            )
            for row in rows:
                try:
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                except Exception:
                    meta = {}
                lookup[row["vector_id"]] = {
                    "chunk_id": row["chunk_id"] or str(row["vector_id"]),
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "metadata": meta,
                }
        return [
            (entry, lookup[entry.vector_id]) for entry in fused
            if entry.vector_id in lookup
        ]

    async def fuse(
        self,
        dense: Sequence[Tuple[int, float]],
        sparse: Sequence[Tuple[int, float]],
    ) -> List[FusionResult]:
        fusion = reciprocal_rank_fusion(
            dense,
            sparse,
            k=self.config.RRF_K,
        )
        return fusion

    def stats(self) -> Dict[str, Any]:
        return {
            "rrf_k": self.config.RRF_K,
            "rerank_enabled": self.config.RERANK_ENABLED,
            "reranker_ready": self.reranker.ready,
            "rerank_model": self.reranker.model_name,
            "bm25_enabled": getattr(self.bm25, "enabled", False),
        }
