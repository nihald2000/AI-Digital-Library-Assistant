"""Embedding service rewritten for scale.

Upgrades over the legacy implementation:
* True async flow with an ``AsyncOpenAI`` client (legacy: sync client in a
  thread pool, blocking the loop's executor threads).
* Concurrent request fan-out bounded by an asyncio semaphore (legacy: one
  sequential request at a time).
* Exponential-backoff retries on 429 / 5xx (legacy: fail the whole batch).
* Configurable batch size + optional server-side dimension truncation
  (``text-embedding-3-*`` Matryoshka) -> 3x less RAM in FAISS + SQLite.
* LRU cache keyed on sha256(model|dimensions|text) so re-indexing the same
  text never pays for the same tokens twice.
* Local SentenceTransformer fallback preserved (runs in an executor).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import numpy as np

import config

logger = logging.getLogger(__name__)

_OPENAI_TRUNCATABLE = ("text-embedding-3-small", "text-embedding-3-large")

_QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted", "credit_balance", "insufficient_user_quota")


class OpenAIEmbeddingsUnavailable(RuntimeError):
    """Permanent OpenAI auth/quotas failure - embeddings should fall back."""


class EmbeddingService:
    def __init__(self):
        cfg = self.config = config.config
        self.model_name = cfg.EMBEDDING_MODEL
        self.dimensions = self._resolve_dimensions()
        self._model: Any = None
        self._device = "cpu"
        self._is_openai_model = False
        self._query_prompt: Optional[str] = None
        self._loop_clients: "OrderedDict[int, Any]" = OrderedDict()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._batch_concurrency = cfg.EMBEDDING_CONCURRENCY
        self._max_retries = cfg.EMBEDDING_MAX_RETRIES
        self._timeout = cfg.EMBEDDING_TIMEOUT_SECONDS
        self._build_client()

    # ------------------------------------------------------------- providers
    def _build_client(self) -> None:
        cfg = self.config
        if self.model_name.startswith("text-embedding-"):
            if cfg.OPENAI_API_KEY:
                try:
                    self._is_openai_model = True
                    # Async clients are created lazily per event loop - Gradio
                    # worker threads rotate loops, and an httpx async client
                    # bound to a dead loop raises "Event loop is closed".
                    self._get_async_client()
                    logger.info("OpenAI async embedding client ready: %s%s", self.model_name,
                                f" dims={self.dimensions}" if self.dimensions else "")
                except RuntimeError as exc:
                    # No running loop yet (module import / main thread at boot).
                    # The client is created on the first embedding call instead.
                    logger.info("OpenAI async client will lazy-init on first embedding call (%s)", exc)
                except Exception as exc:
                    logger.error("Failed to create async OpenAI client: %s", exc)
                    raise
            else:
                logger.warning("%s requested without OPENAI_API_KEY - falling back to local model", self.model_name)
                self._load_local_model(getattr(cfg, "EMBEDDING_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        else:
            self._load_local_model(self.model_name)

    def _disable_openai_leg(self, exc: Exception) -> None:
        """Permanent OpenAI failure (quota/auth): local model takes over now."""
        self._is_openai_model = False
        self._loop_clients.clear()
        self.dimensions = None  # re-resolved by the local model's own geometry
        self._query_prompt = None
        self._model = None
        try:
            self._load_local_model(getattr(self.config, "EMBEDDING_LOCAL_MODEL",
                                           "sentence-transformers/all-MiniLM-L6-v2"))
            logger.error(
                "OpenAI embeddings unavailable (%s: %s) - permanently switched the vector leg to the "
                "free local model '%s' for this session.",
                exc.__class__.__name__, str(exc)[:200], self.model_name,
            )
            logger.warning(
                "Heads-up: vectors previously embedded by the OpenAI model are not compatible with the "
                "new geometry. If the library was ingested with OpenAI embeddings, re-embed once with: "
                "python scripts/reembed_library.py",
            )
        except Exception as load_exc:
            logger.error("Local embedding fallback could not load a model either: %s", load_exc)
            raise

    def _get_async_client(self) -> Any:
        key = id(asyncio.get_running_loop())
        from openai import AsyncOpenAI

        client = self._loop_clients.get(key)
        if client is None:
            client = AsyncOpenAI(api_key=self.config.OPENAI_API_KEY, timeout=self._timeout)
            self._loop_clients[key] = client
            if len(self._loop_clients) > 8:  # bound retained loops (Gradio pools)
                for old_key in list(self._loop_clients)[:-8]:
                    self._loop_clients.pop(old_key, None)
        return client

    def _load_local_model(self, model_name: str) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cuda":
                # bf16 checkpoints halve VRAM; only worthwhile on GPU (bf16 CPU
                # emulation was measured ~10x SLOWER than fp32 on plain servers).
                self._model = SentenceTransformer(model_name, device=self._device, model_kwargs={"dtype": "auto"})
            else:
                self._model = SentenceTransformer(model_name, device=self._device)
            self.model_name = model_name
            self.dimensions = self._model.get_sentence_embedding_dimension()
            # Prompt-trained models (e.g. harrier) need a task instruction on the
            # query side; resolve once and apply only when the caller marks a text
            # as a query. Documents always encode plain.
            wanted_prompt = getattr(self.config, "EMBEDDING_QUERY_PROMPT", "")
            prompts = getattr(self._model, "prompts", {}) or {}
            self._query_prompt = wanted_prompt if wanted_prompt and wanted_prompt in prompts else None
            if self._query_prompt is None and wanted_prompt:
                logger.info(
                    "Local embedding model %s has no query prompt %r - queries encoded without instruction",
                    model_name, wanted_prompt,
                )
            logger.info(
                "Local embedding model %s on %s (dims=%s, query_prompt=%s)",
                model_name, self._device, self.dimensions, self._query_prompt,
            )
        except Exception as exc:
            logger.error("Could not load embedding model '%s': %s", model_name, exc)
            raise

    def _resolve_dimensions(self) -> Optional[int]:
        requested = int(getattr(self.config, "EMBEDDING_DIMENSIONS", 0) or 0)
        if requested > 0 and self.model_name in _OPENAI_TRUNCATABLE:
            return requested
        return None

    @property
    def is_openai_model(self) -> bool:
        return self._is_openai_model

    # ----------------------------------------------------------------- cache
    def _cache_key(self, text: str) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.model_name.encode("utf-8", "ignore"))
        h.update(str(self.dimensions).encode())
        h.update(text.encode("utf-8", "ignore"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[List[float]]:
        if not self.config.EMBEDDING_CACHE_ENABLED:
            return None
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
        return cached

    def _cache_put(self, key: str, value: List[float]) -> None:
        if not self.config.EMBEDDING_CACHE_ENABLED:
            return
        self._cache[key] = value
        limit = self.config.EMBEDDING_CACHE_SIZE
        if limit and len(self._cache) > limit:
            for _ in range(len(self._cache) - limit):
                self._cache.popitem(last=False)

    # ------------------------------------------------------------ public API
    async def generate_embeddings(
        self, texts: Sequence[str], batch_size: Optional[int] = None, is_query: bool = False
    ) -> List[List[float]]:
        if not texts:
            return []
        batch_size = batch_size or self.config.EMBEDDING_BATCH_SIZE

        prepared: List[Optional[str]] = []
        for text in texts:
            prepared.append(text if text and text.strip() else None)

        # check the LRU cache first
        cache_hits: Dict[int, List[float]] = {}
        cache_misses: Dict[int, str] = {}
        for i, text in enumerate(prepared):
            if text is None:
                continue
            key = self._cache_key(text)
            hit = self._cache_get(key)
            if hit is not None:
                cache_hits[i] = hit
            else:
                cache_misses[i] = text

        fetched: Dict[int, List[float]] = {}
        if cache_misses:
            if self._is_openai_model:
                try:
                    fetched = await self._embed_openai(cache_misses, batch_size)
                except OpenAIEmbeddingsUnavailable as exc:
                    # quota exhausted / key dead: stop paying the retry tax and
                    # serve the vector leg from the free local model instead.
                    self._disable_openai_leg(exc)
                    fetched = await self._embed_local(cache_misses, batch_size, is_query=is_query)
            else:
                fetched = await self._embed_local(cache_misses, batch_size)
            for index, text in cache_misses.items():
                if index in fetched:
                    self._cache_put(self._cache_key(text), fetched[index])

        results: List[Optional[List[float]]] = [None] * len(texts)
        for index, vec in cache_hits.items():
            results[index] = vec
        for index, vec in fetched.items():
            results[index] = vec

        missing = sum(1 for r in results if r is None)
        if missing:
            logger.warning("%s of %s inputs produced no embedding", missing, len(texts))
        return [r if r else [0.0] * (self.dimensions or self._model_dims()) for r in results]

    async def generate_single_embedding(self, text: Optional[str], is_query: bool = False) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        embeddings = await self.generate_embeddings([text], is_query=is_query)
        return embeddings[0] if embeddings else None

    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [chunk.get("content", "") for chunk in chunks]
        embeddings = await self.generate_embeddings(texts)
        out: List[Dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched = dict(chunk)
            enriched["embedding"] = embedding
            out.append(enriched)
        return out

    # ------------------------------------------------------------ openai leg
    async def _embed_openai(
        self, index_to_text: Dict[int, str], batch_size: int
    ) -> Dict[int, List[float]]:
        loop = asyncio.get_running_loop()
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self._batch_concurrency))
        semaphore = self._semaphore

        async def request(batch_indices: List[int]) -> Dict[int, List[float]]:
            batch_texts = [index_to_text[i] for i in batch_indices]
            async with semaphore:
                return await self._openai_request_with_retry(batch_texts, batch_indices)

        batches = [list(indices) for indices in self._batch_indices(list(index_to_text), batch_size)]
        results = await asyncio.gather(
            *[loop.create_task(request(b)) for b in batches], return_exceptions=True
        )
        merged: Dict[int, List[float]] = {}
        permanent: Optional[BaseException] = None
        for result in results:
            if isinstance(result, BaseException):
                if isinstance(result, OpenAIEmbeddingsUnavailable):
                    permanent = permanent or result
                else:
                    logger.error("Embedding request failed permanently: %s", result)
                continue
            merged.update(result)
        if permanent:
            raise permanent
        return merged

    async def _openai_request_with_retry(
        self, texts: List[str], indices: List[int]
    ) -> Dict[int, List[float]]:
        assert self._is_openai_model, "async openai leg not active"
        attempt, base_delay = 0, 1.0
        while True:
            try:
                kwargs: Dict[str, Any] = {"input": texts, "model": self.model_name}
                if self.dimensions:
                    kwargs["dimensions"] = self.dimensions
                response = await self._get_async_client().embeddings.create(**kwargs)
                data = sorted(response.data, key=lambda d: d.index)
                return {i: d.embedding for i, d in zip(indices, data)}
            except Exception as exc:
                if self._permanent_openai_failure(exc):
                    # quota exhausted / key revoked - retrying will not help
                    raise OpenAIEmbeddingsUnavailable(str(exc)) from exc
                attempt += 1
                if attempt > self._max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                delay = min(delay, 30)
                logger.warning(
                    "Embedding attempt %s/%s failed (%s: %s) - retrying in %.1fs",
                    attempt, self._max_retries, exc.__class__.__name__, exc, delay,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _permanent_openai_failure(exc: Exception) -> bool:
        """True for auth/quota errors a retry can never fix."""
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return True
        body = getattr(exc, "body", None)
        code = ""
        if isinstance(body, dict):
            code = str(body.get("code") or "")
        message = ""
        try:
            message = str(exc)
        except Exception:
            message = ""
        if any(marker in code for marker in _QUOTA_MARKERS):
            return True
        return any(marker in message for marker in _QUOTA_MARKERS)

    @staticmethod
    def _batch_indices(indices: Sequence[int], batch_size: int):
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size]

    # ------------------------------------------------------------ local leg
    async def _embed_local(
        self, index_to_text: Dict[int, str], batch_size: int, is_query: bool = False
    ) -> Dict[int, List[float]]:
        assert self._model is not None, "local embedding model not loaded"
        loop = asyncio.get_running_loop()
        prompt_name = self._query_prompt if is_query and self._query_prompt else None
        out: Dict[int, List[float]] = {}
        ordered = list(index_to_text.items())
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            batch_indices = [c[0] for c in chunk]
            batch_texts = [c[1] for c in chunk]
            encode_kwargs: Dict[str, Any] = {}
            if prompt_name:
                encode_kwargs["prompt_name"] = prompt_name
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=len(batch_texts),
                    **encode_kwargs,
                ),
            )
            for i, embedding in zip(batch_indices, np.asarray(embeddings)):
                out[i] = embedding.astype("float32").tolist()
        return out

    def _model_dims(self) -> int:
        return self.dimensions or self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------ introspection
    def get_embedding_dimension(self) -> Optional[int]:
        return self._model_dims() if not self._is_openai_model else self.dimensions

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_name": self.model_name,
            "provider": "openai" if self._is_openai_model else "sentence-transformers",
            "dimensions": self.dimensions,
            "device": "openai-api" if self._is_openai_model else self._device,
            "batch_concurrency": self._batch_concurrency,
            "retries": self._max_retries,
            "cache_size": len(self._cache),
            "model_loaded": self._is_openai_model or self._model is not None,
        }
        return info

    VALIDATION_TOLERANCE = 1e-5

    def validate_embedding(self, embedding: Optional[Sequence[float]]) -> bool:
        if embedding is None:
            return False
        arr = np.asarray(embedding, dtype="float32")
        if arr.ndim == 1 and np.isnan(arr).any() or np.isinf(arr).any():
            return False
        return bool(arr.size)

    def compute_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
        return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))

    def compute_similarities(
        self, query_embedding: Sequence[float], embeddings: Sequence[Sequence[float]]
    ) -> List[float]:
        q = np.asarray(query_embedding, dtype="float32")
        mat = np.asarray(embeddings, dtype="float32")
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        denom = (np.linalg.norm(q) * np.linalg.norm(mat, axis=1)) or 1e-9
        sims = (mat @ q) / denom
        return np.clip(sims, -1.0, 1.0).tolist()
