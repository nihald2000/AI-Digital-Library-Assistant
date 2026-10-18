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
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

import config
from services.document_store_service import DocumentStoreService

logger = logging.getLogger(__name__)

class LlamaIndexService:
    def __init__(self, document_store: DocumentStoreService):
        self.document_store = document_store
        self.config = config.config
        self.storage_dir = Path(self.config.DATA_DIR) / "llamaindex_storage"
        self.index = None
        self.agent = None
        self.is_initialized = False
        
        self._initialize_settings()
        # We don't fully initialize index here because we need async access to doc store
        # But we try to load existing storage if available
        self._try_load_from_storage()

    def _initialize_settings(self):
        """Initialize LlamaIndex settings (LLM, Embeddings)"""
        try:
            # LLM Setup
            if self.config.OPENAI_API_KEY:
                # Use configured OpenAI model (gpt-5.1-chat-latest or similar)
                Settings.llm = OpenAI(model=self.config.OPENAI_MODEL, api_key=self.config.OPENAI_API_KEY)
                logger.info(f"LlamaIndex using OpenAI model: {self.config.OPENAI_MODEL}")
            elif self.config.NEBIUS_API_KEY:
                # Use Nebius as OpenAI-compatible provider
                Settings.llm = OpenAI(
                    model=self.config.NEBIUS_MODEL, 
                    api_key=self.config.NEBIUS_API_KEY, 
                    api_base=self.config.NEBIUS_BASE_URL
                )
                logger.info(f"LlamaIndex using Nebius model: {self.config.NEBIUS_MODEL}")
            else:
                logger.warning("No API key found for LlamaIndex LLM (OpenAI or Nebius). Agentic features may fail.")

            # Embedding Setup
            if self.config.EMBEDDING_MODEL.startswith("text-embedding-"):
                if self.config.OPENAI_API_KEY:
                    Settings.embed_model = OpenAIEmbedding(
                        model=self.config.EMBEDDING_MODEL,
                        api_key=self.config.OPENAI_API_KEY
                    )
                    logger.info(f"LlamaIndex using OpenAI embeddings: {self.config.EMBEDDING_MODEL}")
                else:
                    logger.warning("OpenAI embedding model requested but no API key found. Falling back to HuggingFace.")
                    Settings.embed_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
            else:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=self.config.EMBEDDING_MODEL
                )
                logger.info(f"LlamaIndex using HuggingFace embeddings: {self.config.EMBEDDING_MODEL}")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex settings: {str(e)}")

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
                llama_docs.append(Document(
                    text=doc.content,
                    metadata={
                        "filename": doc.filename,
                        "document_id": doc.id,
                        **doc.metadata
                    }
                ))
            if not llama_docs:
                logger.warning("No documents found in DocumentStore to sync.")
                self.index = VectorStoreIndex.from_documents([])
            else:
                logger.info("Building LlamaIndex with %s documents...", len(llama_docs))
                self.index = VectorStoreIndex.from_documents(llama_docs)

            # Persist storage
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))

            # Re-initialize agent with new index
            self._initialize_agent()
            logger.info("LlamaIndex sync complete.")
        except Exception as e:
            logger.error(f"Error syncing LlamaIndex: {str(e)}")

    async def sync_on_demand(self):
        """Manual trigger for syncing documents"""
        await self.sync_from_document_store()
        return True

    def _initialize_agent(self):
        """Initialize the ReAct agent with query engine tools"""
        try:
            if not self.index:
                return

            query_engine = self.index.as_query_engine()
            
            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description="Search and retrieve information from the document library. Use this for specific questions about content."
                )
            )
            
            self.agent = ReActAgent.from_tools(
                [query_engine_tool],
                llm=Settings.llm,
                verbose=True
            )
            logger.info("LlamaIndex ReAct agent initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex agent: {str(e)}")

    async def query(self, query_text: str) -> str:
        """Process a query using the agent"""
        if not self.agent:
            if not self.is_initialized:
                return "Agent is initializing, please try again in a moment."
            return "Agent failed to initialize. Please check logs."
            
        try:
            response = await self.agent.achat(query_text)
            return str(response)
        except Exception as e:
            logger.error(f"Error querying LlamaIndex agent: {str(e)}")
            return f"Error processing query: {str(e)}"
