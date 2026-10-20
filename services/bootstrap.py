"""Service bootstrap: one shared wiring for app.py (Gradio) and mcp_server.py.

Creates the storage core and all downstream services once per process, boots
the background maintenance task (FAISS checkpointing, sparse index rebuild,
vector index compaction) and exposes a clean shutdown handle.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from services.sqlite_store import SQLiteStore
from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.ocr_service import OCRService
from services.bm25_service import BM25Service
from services.hybrid_search_service import HybridSearchService
from services.pii_service import PIIService
from services.ingestion_queue_service import IngestionQueueService
from services.llamaindex_service import LlamaIndexService
from services.elevenlabs_service import ElevenLabsService
from services.podcast_generator_service import PodcastGeneratorService
from mcp_tools.ingestion_tool import IngestionTool
from mcp_tools.search_tool import SearchTool
from mcp_tools.generative_tool import GenerativeTool
from mcp_tools.voice_tool import VoiceTool
from mcp_tools.podcast_tool import PodcastTool

import config

logger = logging.getLogger(__name__)


class AppServices:
    """Initialized once; both entry points share this instance."""

    def __init__(self):
        cfg = config.config
        self.store: SQLiteStore = SQLiteStore(cfg.DB_PATH)
        self.vector_store = VectorStoreService(store=self.store)
        self.document_store = DocumentStoreService(store=self.store)
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.ocr_service = OCRService()
        self.bm25_service = BM25Service(store=self.store)
        # vector store talks back on the sparse leg for hybrid fusions
        self.vector_store.bm25_service = self.bm25_service

        self.hybrid_search = HybridSearchService(
            vector_search=self.vector_store,
            bm25_service=self.bm25_service,
        )
        self.pii_service = PIIService(store=self.store)

        self.ingestion_tool = IngestionTool(
            vector_store=self.vector_store,
            document_store=self.document_store,
            embedding_service=self.embedding_service,
            ocr_service=self.ocr_service,
        )
        self.ingest_queue = IngestionQueueService(
            ingestion_tool=self.ingestion_tool, store=self.store
        )
        self.llamaindex_service = LlamaIndexService(self.document_store)

        self.elevenlabs_service = ElevenLabsService(self.llamaindex_service)
        self.podcast_generator = PodcastGeneratorService(
            llamaindex_service=self.llamaindex_service,
            llm_service=self.llm_service,
        )

        self.search_tool = SearchTool(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            document_store=self.document_store,
            llamaindex_service=self.llamaindex_service,
            hybrid_service=self.hybrid_search,
        )
        self.generative_tool = GenerativeTool(
            llm_service=self.llm_service,
            search_tool=self.search_tool,
            pii_service=self.pii_service,
        )
        self.voice_tool = VoiceTool(self.elevenlabs_service)
        self.podcast_tool = PodcastTool(self.podcast_generator)

        self._maintenance_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ boot
    async def start(self) -> None:
        await self.store.initialize()
        await self.vector_store.initialize()
        await self.bm25_service.initialize()
        await self.pii_service.initialize()
        if config.config.INGEST_QUEUE_ENABLED:
            await self.ingest_queue.initialize()
        logger.info("AppServices ready")

    async def start_maintenance(self) -> None:
        """Background loops: sparse index rebuild + queued ingestion drain.

        FAISS persistence intentionally runs from the main event loop
        (inside add/delete flows) - FAISS is not thread-safe for concurrent
        read + write against the same index instance.
        """
        if self._maintenance_task and not self._maintenance_task.done():
            return

        async def _loop():
            interval = 15
            if config.config.INGEST_QUEUE_ENABLED:
                await self.ingest_queue.initialize()
                await self.ingest_queue.recover_stale_running()
                # register on the service so get_stats().worker_alive is truthful
                self.ingest_queue._workers = [
                    asyncio.create_task(
                        self.ingest_queue.run_worker_loop(), name="ingest-queue-worker"
                    )
                ]
                logger.info("Ingestion queue worker enabled")
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        rebuild_info = await self.bm25_service.maybe_rebuild()
                        if rebuild_info:
                            logger.info("BM25 maintenance: %s", rebuild_info)
                    except Exception as exc:
                        logger.exception("Maintenance loop error: %s", exc)
            except asyncio.CancelledError:
                for task in worker_tasks:
                    task.cancel()
                return

        self._maintenance_task = asyncio.create_task(_loop(), name="rag-maintenance")

    def start_maintenance_in_thread(self) -> None:
        """Launch the maintenance loop on a dedicated thread (its own loop)."""
        import threading

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.start_maintenance())
                loop.run_forever()
            except Exception as exc:
                logger.exception("Maintenance thread crashed: %s", exc)
            finally:
                loop.close()

        threading.Thread(target=_runner, name="rag-maintenance", daemon=True).start()

    async def shutdown(self) -> None:
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except (asyncio.CancelledError, Exception):
                pass
            self._maintenance_task = None
        try:
            await self.vector_store.persist(force=True)
        except Exception as exc:
            logger.warning("Final vector persist failed: %s", exc)
        await self.store.close()
        logger.info("AppServices shut down cleanly")


_services: Optional[AppServices] = None


def get_services() -> AppServices:
    """Lazy singleton shared between processes in a single event loop."""
    global _services
    if _services is None:
        _services = AppServices()
    return _services
