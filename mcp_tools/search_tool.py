"""Search tool upgraded for the 1M+ scale retrieval path.

Pipeline per query:
  1. embed the query (LRU-cached on the embedding service)
  2. dense leg: FAISS search with overfetch escalation
  3. sparse leg: BM25 over SQLite chunks (bm25s), skipped when unavailable
  4. Reciprocal Rank Fusion of the two ranked lists
  5. hydrate fused candidates in one batched SQL round trip
  6. optional cross-encoder rerank of the top N candidates
  7. document-level metadata enrichment (batched, not the legacy N+1)

The public signatures are unchanged:
  search(query, top_k, filters, similarity_threshold)
  agentic_search, multi_query_search, search_by_document, search_by_category,
  search_with_date_range, get_search_suggestions, explain_search
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from core.models import SearchResult, Document
from services.embedding_service import EmbeddingService
from services.document_store_service import DocumentStoreService
import config

logger = logging.getLogger(__name__)


class SearchTool:
    def __init__(
        self,
        vector_store: Any,
        embedding_service: EmbeddingService,
        document_store: Optional[DocumentStoreService] = None,
        llamaindex_service: Any = None,
        hybrid_service: Any = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_store = document_store
        self.llamaindex_service = llamaindex_service
        self.hybrid_service = hybrid_service
        self.config = config.config

    # ------------------------------------------------------------------ query
    async def search(
        self, query: str, top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Hybrid (dense + sparse) search with RRF fusion."""
        started = time.perf_counter()
        try:
            if not query or not query.strip():
                logger.warning("Empty search query provided")
                return []
            top_k = max(1, int(top_k))
            threshold = (
                similarity_threshold
                if similarity_threshold is not None
                else self.config.SIMILARITY_THRESHOLD
            )
            logger.info("Hybrid search: '%s' (top_k=%s)", query, top_k)

            fused = await self._hybrid_fuse(query, top_k, filters)
            if not fused:
                logger.info("No candidates from either leg")
                return []

            hydrated = await self.hybrid_service.hydrate_candidates(fused)

            # optional cross-encoder rerank
            if self.hybrid_service and len(hydrated) > 1:
                try:
                    content_lookup = {entry.vector_id: row["content"] for entry, row in hydrated}
                    row_by_id = {entry.vector_id: row for entry, row in hydrated}
                    reranked = await self.hybrid_service.reranker.rerank(
                        query, [entry for entry, _ in hydrated], content_lookup
                    )
                    hydrated = [
                        (entry, row_by_id[entry.vector_id])
                        for entry in reranked if entry.vector_id in row_by_id
                    ]
                except Exception as exc:
                    logger.warning("Rerank skipped: %s", exc)

            results: List[SearchResult] = []
            for entry, chunk in hydrated:
                # fused scores are rank-based, so the legacy similarity
                # threshold applies to the dense score when that leg fired.
                dense_floor = self.config.SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
                dense_floor = min(dense_floor, 0.95) if dense_floor > 0 else dense_floor
                if entry.dense_score is not None and entry.dense_score < max(0.05, dense_floor):
                    continue
                results.append(SearchResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    score=entry.fused_score,
                    metadata={
                        **(chunk.get("metadata") or {}),
                        "_dense_score": entry.dense_score,
                        "_sparse_score": entry.sparse_score,
                        "_dense_rank": entry.dense_rank,
                        "_sparse_rank": entry.sparse_rank,
                    },
                ))
            results = results[:top_k]

            # document metadata enrichment (one batched call, not N+1)
            if self.document_store and results:
                doc_ids = [r.document_id for r in results]
                metas = await self.document_store.get_documents_metadata_batch(doc_ids)
                for result in results:
                    meta = metas.get(result.document_id)
                    if meta:
                        result.metadata.update({
                            "document_filename": meta.get("filename"),
                            "document_type": meta.get("doc_type"),
                            "document_tags": meta.get("tags"),
                            "document_category": meta.get("category"),
                            "document_created_at": meta.get("created_at"),
                            "document_summary": meta.get("summary"),
                        })

            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Hybrid search returned %s results in %.0fms",
                len(results), elapsed_ms,
            )
            return results
        except Exception as exc:
            logger.exception("Hybrid search failed: %s", exc)
            return []

    async def _hybrid_fuse(
        self, query: str, top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Returns a list of FusionResult entries (dense-only or RRF-fused)."""
        from services.hybrid_search_service import FusionResult

        # Dense leg always runs. is_query=True applies the model's query-side
        # instruction (e.g. harrier's "web_search_query") when available.
        query_embedding = await self.embedding_service.generate_single_embedding(query, is_query=True)
        if query_embedding is None:
            return []
        dense_rows = await self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k,
            filters=filters,
        )
        dense: List[Tuple[int, float]] = []
        for row in dense_rows:
            vid = (row.metadata or {}).get("_vector_id")
            if vid is not None:
                dense.append((int(vid), row.score))

        # Sparse leg when available
        sparse: List[Tuple[int, float]] = []
        if (
            self.hybrid_service
            and self.hybrid_service.bm25.exists_index()
        ):
            try:
                sparse = await self.hybrid_service.bm25.query(query, max(top_k * 3, 15))
            except Exception as exc:
                logger.warning("Sparse leg failed (%s) - dense only", exc)

        if not dense and not sparse:
            return []
        if not sparse:
            # Dense-only path routes through the same FusionResult structure
            # so hydrate/rerank/threshold logic stays uniform.
            return [
                FusionResult(vector_id=vid, fused_score=score, dense_score=score)
                for vid, score in dense
            ]

        return await self.hybrid_service.fuse(dense, sparse)

    # ------------------------------------------------------------- utilities
    async def agentic_search(self, query: str) -> str:
        """Agentic search using the LlamaIndex ReAct agent."""
        if not self.llamaindex_service:
            logger.warning("LlamaIndex service not available for agentic search")
            return "Agentic search not available."
        try:
            logger.info("Agentic search: '%s'", query)
            return await self.llamaindex_service.query(query)
        except Exception as exc:
            logger.error("Error performing agentic search: %s", exc)
            return f"Error performing agentic search: {str(exc)}"

    async def multi_query_search(
        self, queries: List[str], top_k: int = 5,
        aggregate_method: str = "rrf",
    ) -> List[SearchResult]:
        """Run the fused query set in parallel and merge by score."""
        try:
            valid = [q for q in queries if q and q.strip()]
            if not valid:
                return []
            # each individual call already runs the hybrid pipeline; running
            # them concurrently keeps wall time close to a single search
            nested = await asyncio.gather(*(self.search(q, top_k) for q in valid))
            flat_results: List[SearchResult] = [r for batch in nested for r in batch]
            if not flat_results:
                return []
            if aggregate_method == "rrf":
                return self._dedup_keep_max(flat_results, top_k)
            if aggregate_method == "intersect":
                counts: Dict[str, int] = {}
                for r in flat_results:
                    counts[r.chunk_id] = counts.get(r.chunk_id, 0) + 1
                keep = {cid for cid, c in counts.items() if c > 1}
                return self._dedup_keep_max(
                    [r for r in flat_results if r.chunk_id in keep], top_k
                )
            if aggregate_method == "average":
                grouped: Dict[str, List[SearchResult]] = {}
                for r in flat_results:
                    grouped.setdefault(r.chunk_id, []).append(r)
                averages = [
                    SearchResult(
                        chunk_id=cid,
                        document_id=group[0].document_id,
                        content=group[0].content,
                        score=sum(g.score for g in group) / len(group),
                        metadata=group[0].metadata,
                    )
                    for cid, group in grouped.items()
                ]
                return sorted(averages, key=lambda r: r.score, reverse=True)[:top_k]
            return self._dedup_keep_max(flat_results, top_k)
        except Exception as exc:
            logger.error("Multi-query search failed: %s", exc)
            return []

    @staticmethod
    def _dedup_keep_max(results: List[SearchResult], top_k: int) -> List[SearchResult]:
        best: Dict[str, SearchResult] = {}
        for result in results:
            current = best.get(result.chunk_id)
            if current is None or result.score > current.score:
                best[result.chunk_id] = result
        out = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return out[:top_k]

    async def search_by_document(self, document_id: str, query: str, top_k: int = 5) -> List[SearchResult]:
        return await self.search(query, top_k, filters={"document_id": document_id})

    async def search_by_category(self, category: str, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.document_store:
            return await self.search(query, top_k)
        docs = await self.document_store.search_documents(
            category, fields=["category"], limit=self.config.SEARCH_MAX_OVERFETCH
        )
        if not docs:
            logger.info("No documents found in category '%s'", category)
            return []
        doc_ids = [doc.id for doc in docs]
        return await self.search(query, top_k, filters={"document_ids": doc_ids})

    async def search_with_date_range(
        self, query: str, start_date, end_date, top_k: int = 5
    ) -> List[SearchResult]:
        if not self.document_store:
            return await self.search(query, top_k)
        # pushed down to the SQLite index on created_at - no full scans
        all_matched: List[Document] = []
        page, offset = 1000, 0
        filters = {"created_after": start_date, "created_before": end_date}
        while True:
            page_docs = await self.document_store.list_documents(
                limit=page, offset=offset, filters=filters
            )
            if not page_docs:
                break
            all_matched.extend(page_docs)
            offset += page
            if len(all_matched) >= self.config.SEARCH_MAX_OVERFETCH:
                all_matched = all_matched[: self.config.SEARCH_MAX_OVERFETCH]
                break
        if not all_matched:
            logger.info("No documents found in date range")
            return []
        return await self.search(query, top_k, filters={"document_ids": [d.id for d in all_matched]})

    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        try:
            if len(partial_query) < 2:
                return []
            sparse_leg = (
                getattr(self.hybrid_service.bm25, "enabled", False)
                if self.hybrid_service else False
            )
            if self.hybrid_service and sparse_leg and self.hybrid_service.bm25.exists_index():
                # Lexical suggestions scale better than a brute-force content scan
                rows = await self.hybrid_service.bm25.query(partial_query, top_k=20, excluded=None)
                if not rows:
                    return []
                vid_list = [vid for vid, _ in rows]
                chunks_raw = await self.vector_store.store.fetch_all(
                    "SELECT content FROM chunks WHERE vector_id IN (%s)"
                    % ",".join("?" for _ in vid_list),
                    vid_list,
                )
                contents = [r["content"] for r in chunks_raw]
            else:
                results = await self.search(partial_query, top_k=20)
                contents = [r.content for r in results]
            suggestions = set()
            for content in contents:
                words = content.lower().split()
                for i, word in enumerate(words):
                    if partial_query.lower() in word:
                        suggestions.add(word.strip('.,!?;:'))
                        if i > 0:
                            suggestions.add(f"{words[i-1]} {word}".strip('.,!?;:'))
                        if i < len(words) - 1:
                            suggestions.add(f"{word} {words[i+1]}".strip('.,!?;:'))
            filtered = [
                s for s in suggestions
                if len(s) > len(partial_query) and s.startswith(partial_query.lower())
            ]
            return sorted(filtered)[:limit]
        except Exception as exc:
            logger.error("Error getting search suggestions: %s", exc)
            return []

    async def explain_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            explanation: Dict[str, Any] = {
                "query": query,
                "steps": [],
                "results_analysis": {},
                "performance_metrics": {},
            }
            started = time.perf_counter()

            embed_start = time.perf_counter()
            query_embedding = await self.embedding_service.generate_single_embedding(query, is_query=True)
            explain_step = {
                "step": "embedding_generation",
                "description": "Converting the query into a dense vector",
                "details": {
                    "embedding_dimension": len(query_embedding) if query_embedding else 0,
                    "generation_time_ms": round((time.perf_counter() - embed_start) * 1000, 2),
                },
            }
            explanation["steps"].append(explain_step)

            search_start = time.perf_counter()
            results = await self.search(query, top_k=top_k)
            search_ms = round((time.perf_counter() - search_start) * 1000, 2)

            explanation["steps"].append({
                "step": "hybrid_retrieval",
                "description": "Dense (FAISS) + sparse (BM25) fusion with RRF",
                "details": {
                    "elapsed_ms": search_ms,
                    "results_found": len(results),
                    "hybrid_stats": (
                        self.hybrid_service.stats() if self.hybrid_service else {}
                    ),
                },
            })

            if results:
                explanation["results_analysis"] = {
                    "total_results": len(results),
                    "average_score": sum(r.score for r in results) / len(results),
                    "unique_documents": len({r.document_id for r in results}),
                    "content_lengths": [len(r.content) for r in results],
                }

            explanation["performance_metrics"] = {
                "total_time_ms": round((time.perf_counter() - started) * 1000, 2),
                "embedding_time_ms": explain_step["details"]["generation_time_ms"],
                "retrieval_time_ms": search_ms,
            }
            return explanation
        except Exception as exc:
            logger.error("Error explaining search: %s", exc)
            return {"error": str(exc)}
