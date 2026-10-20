import logging
import os
from typing import List, Optional, Any
from pathlib import Path
import shutil
import asyncio

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
    SummaryIndex
)
import tiktoken
from llama_index.core.llms import LLMMetadata
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

import config
from services.document_store_service import DocumentStoreService
from services.mcp_pii import AGENTIC_PII_NAMESPACE
from services import llm_providers
from services.llm_providers import permanent_llm_error as _permanent_openai_error

logger = logging.getLogger(__name__)


_AGENTIC_CONTEXT_WINDOW = 128_000


class _OpenAICompat(OpenAI):
    """OpenAI/Nebius-compatible LLM with the llama_index registry bypassed.

    llama_index validates model names against its baked-in OpenAI list in the
    `metadata` property. Two perfectly valid models are unknown there -
    `gpt-5.1-chat-latest` and every Nebius model (e.g.
    `meta-llama/Llama-3.3-70B-Instruct`) - so ReActAgent init died with
    "Unknown model". Same for tiktoken, which cannot encode foreign names.
    NOTE: pydantic model base -> class constants must be annotated or module-level.
    """

    @property
    def metadata(self) -> "LLMMetadata":
        return LLMMetadata(
            context_window=_AGENTIC_CONTEXT_WINDOW,
            num_output=self.max_tokens or 4096,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )

    @property
    def _tokenizer(self):
        try:
            return tiktoken.encoding_for_model(self._get_model_name())
        except Exception:
            return tiktoken.get_encoding("o200k_base")


# _permanent_openai_error now lives in services/llm_providers (shared with
# the REST chain) and is imported as permanent_llm_error above.


class AgenticQueryError(RuntimeError):
    """Raised by query_strict when the agentic pipeline cannot answer."""


class LlamaIndexService:
    def __init__(self, document_store: DocumentStoreService, pii_service: Any = None,
                 embedding_service: Any = None):
        self.document_store = document_store
        self.config = config.config
        self.pii_service = pii_service
        self.embedding_service = embedding_service
        self._local_embeddings_set = False
        self.storage_dir = Path(self.config.DATA_DIR) / "llamaindex_storage"
        self.index = None
        self.agent = None
        self.is_initialized = False
        self._last_sync = None  # (synced_doc_count, monotonic_time)
        self._openai_retry_done = False  # one local-fallback retry per process
        self._syncing = False

        self._initialize_settings()
        # We don't fully initialize index here because we need async access to doc store
        # But we try to load existing storage if available
        self._try_load_from_storage()

    def _initialize_settings(self):
        """Initialize LlamaIndex settings (LLM, Embeddings)."""
        try:
            # LLM Setup - one shared provider chain (free tiers first):
            # Gemini -> Groq -> OpenRouter -> Nebius -> OpenAI. All legs are
            # OpenAI-compatible, so the same _OpenAICompat wrapper (which
            # bypasses llama_index's stale model-name registry) covers all.
            leg = llm_providers.pick_llm_provider(self.config)
            if leg:
                kwargs = {"model": leg["model"], "api_key": leg["api_key"]}
                if leg["name"] != "openai":
                    kwargs["api_base"] = leg["base_url"]
                Settings.llm = _OpenAICompat(max_retries=1, **kwargs)
                logger.info("LlamaIndex using %s model: %s", leg["name"], leg["model"])
            else:
                logger.warning(
                    "No LLM API key found (GEMINI_API_KEY / GROQ_API_KEY / "
                    "OPENROUTER_API_KEY / NEBIUS_API_KEY / OPENAI_API_KEY). "
                    "Agentic features may fail."
                )

            # Embedding Setup
            if self.config.EMBEDDING_MODEL.startswith("text-embedding-"):
                if self.embedding_service and not self.embedding_service.openai_embeddings_active:
                    # The main embedding leg already learned this key is dead
                    # (quota exhausted / invalid) - do NOT point LlamaIndex at
                    # it; every agentic build would 429 forever.
                    logger.info("OpenAI embedding leg is inactive - LlamaIndex uses the local model directly")
                    self._ensure_local_embeddings()
                elif self.config.OPENAI_API_KEY:
                    Settings.embed_model = OpenAIEmbedding(
                        model=self.config.EMBEDDING_MODEL,
                        api_key=self.config.OPENAI_API_KEY
                    )
                    logger.info(f"LlamaIndex using OpenAI embeddings: {self.config.EMBEDDING_MODEL}")
                else:
                    logger.warning("OpenAI embedding model requested but no API key found. Falling back to HuggingFace.")
                    self._ensure_local_embeddings()
            else:
                self._ensure_local_embeddings()
                logger.info(f"LlamaIndex using HuggingFace embeddings: {self.config.EMBEDDING_MODEL}")

        except Exception as e:
            logger.error(f"Error initializing LlamaIndex settings: {str(e)}")

    def _ensure_local_embeddings(self):
        """Point Settings.embed_model at the configured local (HuggingFace) model."""
        if getattr(self, "_local_embeddings_set", False) and isinstance(
            getattr(self, "_settings_embed", None), HuggingFaceEmbedding
        ):
            return
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=getattr(self.config, "EMBEDDING_LOCAL_MODEL",
                               "sentence-transformers/all-MiniLM-L6-v2")
        )
        self._local_embeddings_set = True

    def _try_load_from_storage(self):
        """Try to load index from storage synchronously"""
        try:
            if self.storage_dir.exists():
                logger.info("Loading LlamaIndex from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
                self.index = load_index_from_storage(storage_context)
                self._initialize_agent()
                self.is_initialized = True
            else:
                logger.info("No existing LlamaIndex storage found. Waiting for async initialization.")
        except Exception as e:
            logger.error(f"Error loading LlamaIndex from storage: {str(e)}")

    async def initialize(self):
        """Async initialization to sync documents and build index"""
        try:
            logger.info("Starting LlamaIndex async initialization...")
            
            # If we already have an index, we might still want to sync if it's empty or stale
            # For now, if no index exists, we definitely need to build it
            if self.index is None:
                await self.sync_from_document_store()
            
            self.is_initialized = True
            logger.info("LlamaIndex async initialization complete.")
            
        except Exception as e:
            logger.error(f"Error during LlamaIndex async initialization: {str(e)}")

    async def sync_from_document_store(self):
        """Sync documents from DocumentStore to LlamaIndex (bounded).

        Agentic RAG builds a *second* full in-RAM index, so at 1M+ documents
        this must stay capped (LLAMAINDEX_MAX_DOCS) and stream rather than
        materialize the whole library.
        """
        try:
            if not self.config.LLAMAINDEX_ENABLED:
                logger.info("LlamaIndex disabled (LLAMAINDEX_ENABLED=false) - skipping sync")
                return
            if self._syncing:
                # boot init + maintenance + explicit UI calls can race; a
                # duplicate full rebuild is wasted CPU and can interleave
                # FAISS-external writes
                logger.info("LlamaIndex sync already in progress - skipping")
                return
            self._syncing = True
            try:
                await self._sync_body()
            finally:
                self._syncing = False
        except Exception as e:
            logger.error(f"Error syncing LlamaIndex: {str(e)}")
            if self.embedding_service and not self._openai_retry_done and _permanent_openai_error(e):
                # the dead OpenAI key crippled the agentic build too - switch
                # to the local model (single source of truth) and retry once
                self._openai_retry_done = True
                try:
                    self.embedding_service._disable_openai_leg(e)
                    self._ensure_local_embeddings()
                    await self.sync_from_document_store()
                except Exception as retry_exc:
                    logger.error("LlamaIndex local-embedding retry failed: %s", retry_exc)

    async def _sync_body(self) -> None:
        try:
            total_docs = await self.document_store.count_documents()
        except Exception:
            total_docs = -1
        if 0 <= total_docs > self.config.LLAMAINDEX_MAX_DOCS:
            logger.warning(
                "Agentic index cap hit: %s documents > LLAMAINDEX_MAX_DOCS (%s). "
                "Agentic features disabled for scale; use semantic_search instead.",
                total_docs, self.config.LLAMAINDEX_MAX_DOCS,
            )
            return

        logger.info("Syncing documents from DocumentStore to LlamaIndex (capped at %s)...",
                    self.config.LLAMAINDEX_MAX_DOCS)
        llama_docs = []
        count = 0
        async for doc in self.document_store.iter_documents(batch_size=500):
            count += 1
            if count > self.config.LLAMAINDEX_MAX_DOCS:
                break
            # PII boundary: the agentic pipeline (voice QA, podcast,
            # agentic search) never sees raw sensitive values. Content is
            # masked into the shared agentic namespace; user-facing
            # outputs (answers, transcripts) are unmasked at delivery.
            llama_docs.append(Document(
                text=self._mask_for_agentic(doc.content or ""),
                metadata={
                    "filename": doc.filename,
                    "document_id": doc.id,
                    **doc.metadata
                }
            ))
        if not llama_docs:
            # An empty library can't have an empty VectorStoreIndex -
            # seed a single placeholder so the agent still boots and can
            # answer "library is empty" instead of being dead forever.
            logger.warning("No documents found in DocumentStore to sync.")
            self.index = VectorStoreIndex.from_documents(
                [Document(text="The library is currently empty. No documents have been ingested yet.")]
            )
        else:
            logger.info("Building LlamaIndex with %s documents...", len(llama_docs))
            self.index = VectorStoreIndex.from_documents(llama_docs)

        # Persist storage
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))

        # Re-initialize agent with new index
        self._initialize_agent()
        self._last_sync = (count, __import__("time").monotonic())
        logger.info("LlamaIndex sync complete.")

    def _mask_for_agentic(self, text: str) -> str:
        """Mask PII before it enters the agentic index (best effort)."""
        if not text or not self.pii_service:
            return text
        try:
            masked = self.pii_service.mask_for_agent(
                text, namespace=AGENTIC_PII_NAMESPACE
            )
            return masked.masked_text
        except Exception as exc:
            logger.warning("Agentic PII masking failed, using raw text: %s", exc)
            return text

    def restore_for_user(self, text: str) -> str:
        """Swap agentic placeholders back to real values (memory-only, TTL)."""
        if not text or not self.pii_service:
            return text
        try:
            return self.pii_service.restore_for_user(text, AGENTIC_PII_NAMESPACE)
        except Exception as exc:
            logger.warning("Agentic PII restore failed: %s", exc)
            return text

    async def sync_if_stale(self) -> bool:
        """Resync when the library changed (debounced to one sync/minute)."""
        try:
            total = await self.document_store.count_documents()
        except Exception:
            return False
        last = self._last_sync
        if last and last[0] == total:
            return False
        import time as _t
        if last and (_t.monotonic() - last[1]) < 60.0:
            return False  # debounce
        await self.sync_from_document_store()
        return True

    async def sync_on_demand(self):
        """Manual trigger for syncing documents"""
        await self.sync_from_document_store()
        return True

    def _initialize_agent(self):
        """Initialize the ReAct agent with query engine tools.

        llama_index's workflow API replaced the legacy `from_tools`/`achat`
        surface: agents are plain constructors now and are driven with
        `await agent.run(input=...)`.
        """
        try:
            if not self.index:
                return
            if Settings.llm is None:
                logger.warning("No LLM configured for LlamaIndex agent - agentic features unavailable")
                return

            query_engine = self.index.as_query_engine()

            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description="Search and retrieve information from the document library. Use this for specific questions about content."
                )
            )

            # streaming=False: the compat LLM below is a legacy chat LLM whose
            # non-streaming path is the reliable one inside the workflow.
            self.agent = ReActAgent(
                name="library_agent",
                description="Answers questions about the user's document library.",
                tools=[query_engine_tool],
                llm=Settings.llm,
                streaming=False,
                timeout=120.0,
                verbose=False,
            )
            logger.info("LlamaIndex ReAct agent initialized")

        except Exception as e:
            logger.error(f"Error initializing LlamaIndex agent: {str(e)}")

    @staticmethod
    def _agent_output_text(result: Any) -> str:
        """Extract answer text from a workflow AgentOutput (or legacy response)."""
        response = getattr(result, "response", None)
        if response is None:
            response = getattr(result, "result", None)
        inner = getattr(response, "response", None)  # AgentChatResponse-like
        text = inner if isinstance(inner, str) else response
        return str(text if text is not None else result)

    async def _agent_run(self, query_text: str) -> str:
        result = await self.agent.run(input=query_text)
        return self._agent_output_text(result)

    async def query(self, query_text: str) -> str:
        """Process a query using the agent (never raises - returns error text)."""
        if not self.agent:
            if not self.is_initialized:
                return "Agent is initializing, please try again in a moment."
            return "Agent failed to initialize. Please check logs."

        try:
            response = await self._agent_run(query_text)
            return response
        except Exception as e:
            logger.error(f"Error querying LlamaIndex agent: {str(e)}")
            return f"Error processing query: {str(e)}"

    async def query_strict(self, query_text: str) -> str:
        """Like query() but RAISES on failure so callers never mistake an
        error string for a successful answer (voice QA / podcast bug)."""
        self.raise_if_broken()
        try:
            response = await self._agent_run(query_text)
        except AgenticQueryError:
            raise
        except Exception as e:
            logger.error(f"Error querying LlamaIndex agent: {str(e)}")
            raise AgenticQueryError(str(e)) from e
        return self.restore_for_user(response)

    def raise_if_broken(self) -> None:
        """Fail loudly (an AgenticQueryError) when the agent is unusable."""
        if not self.is_initialized:
            raise AgenticQueryError("Agentic index is still initializing - try again in a moment.")
        if not self.agent:
            raise AgenticQueryError(
                "Agentic index not available (LLM/embedding setup missing or "
                "LLAMAINDEX_MAX_DOCS cap exceeded). Use semantic_search instead."
            )
