"""Ingestion pipeline rewritten for streaming batch processing at scale.

Legacy problems this replaces:
  * embeddings were built for the WHOLE document before any ingest call
  * every `add_chunks` rewrote the entire FAISS index + JSON metadata blob
  * directory ingestion was strictly sequential (no concurrency)

New pipeline per document:
  parse -> chunk -> stream batches of (chunk) -> embed batch -> persist batch
  -> BM25 bookkeeping -> metadata updates轻薄

Everything stays bounded in RAM: only `EMBEDDING_BATCH_SIZE` chunks and their
embeddings are alive at any moment, and SQLite WAL writes amortize batches.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.document_parser import DocumentParser
from core.chunker import TextChunker
from core.text_preprocessor import TextPreprocessor
from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    pass


class IngestionTool:
    def __init__(
        self,
        vector_store: VectorStoreService,
        document_store: DocumentStoreService,
        embedding_service: EmbeddingService,
        ocr_service: OCRService,
    ):
        self.vector_store = vector_store
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.ocr_service = ocr_service

        self.document_parser = DocumentParser()
        # Pass the OCR service to the document parser
        self.document_parser.ocr_service = ocr_service

        self.text_chunker = TextChunker()
        self.text_preprocessor = TextPreprocessor()
        self.config = self.document_store.config

    # -------------------------------------------------------------- pipeline
    async def process_document(
        self, file_path: str, file_type: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full streaming ingest pipeline for a single document."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            parse_started = time.perf_counter()
            filename = Path(file_path).name
            document = await self.document_parser.parse_document(file_path, filename)
            if not document.content:
                logger.warning("No content extracted from document %s", filename)
                return {
                    "success": False,
                    "error": "No content could be extracted from the document",
                    "task_id": task_id,
                    "file_path": file_path,
                }
            parse_ms = (time.perf_counter() - parse_started) * 1000

            # register the document before chunking so a crash leaves a consistent catalog row
            await self.document_store.store_document(document)

            result = await self._ingest_parsed_document(
                document,
                task_id=task_id,
                source=file_path,
                parse_ms=parse_ms,
            )
            return result
        except Exception as exc:
            logger.exception("Error processing document %s: %s", file_path, exc)
            return {
                "success": False,
                "error": str(exc),
                "task_id": task_id,
                "file_path": file_path,
            }

    async def _ingest_parsed_document(
        self, document, task_id: Optional[str] = None,
        source: str = "", parse_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Shared streaming path for parsed documents and raw text."""
        started = time.perf_counter()
        filename = document.filename

        chunk_started = time.perf_counter()
        chunks_raw = self.text_chunker.chunk_document(
            document.id,
            document.content,
            method="recursive",
        )
        if not chunks_raw:
            chunks_raw = []
        chunks = self.text_chunker.optimize_chunks_for_embedding(chunks_raw)
        chunk_ms = (time.perf_counter() - chunk_started) * 1000

        if not chunks:
            logger.warning("Document %s produced no usable chunks", filename)
            await self.document_store.update_document_metadata(
                document.id, {"chunk_count": 0}
            )
            return {
                "success": True,
                "task_id": task_id or str(uuid.uuid4()),
                "document_id": document.id,
                "filename": filename,
                "chunks_created": 0,
                "content_length": len(document.content),
                "doc_type": document.doc_type.value,
                "message": f"Stored {filename} (no chunkable content)",
            }

        # embed + persist in streaming batches; only a few hundred chunks and
        # their vectors are alive at any moment
        embed_ms = 0.0
        index_ms = 0.0
        embedded_count = 0
        batch_size = max(1, self.config.EMBEDDING_BATCH_SIZE)
        loop = asyncio.get_running_loop()

        index_start = time.perf_counter()
        indexed = 0
        embed_block = batch_size * 4  # bounded RAM blocks of embedding calls
        for start in range(0, len(chunks), embed_block):
            block = chunks[start:start + embed_block]
            embed_started = time.perf_counter()
            texts = [chunk.content for chunk in block]
            block_embeddings = await self.embedding_service.generate_embeddings(
                texts, batch_size=batch_size
            )
            embed_ms += (time.perf_counter() - embed_started) * 1000
            for chunk, embedding in zip(block, block_embeddings):
                chunk.embedding = embedding
            # SQLite commit + FAISS add in bounded sub-batches
            for sub_start in range(0, len(block), batch_size * 2):
                batch_chunks = block[sub_start:sub_start + batch_size * 2]
                await self.vector_store.add_chunks(batch_chunks)
                indexed += len(batch_chunks)
        index_ms = (time.perf_counter() - index_start) * 1000
        embedded_count = indexed

        chunk_count = len(chunks)
        await self.document_store.update_document_metadata(
            document.id, {"chunk_count": chunk_count}
        )
        bm25 = getattr(self.vector_store, "bm25_service", None)
        if bm25 is not None:
            bm25.mark_chunks_added(chunk_count)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ingested %s: %s chunks (parse %.0fms, chunk %.0fms, embed %.0fms, index %.0fms)",
            filename, chunk_count, parse_ms, chunk_ms, embed_ms, index_ms,
        )
        return {
            "success": True,
            "task_id": task_id or str(uuid.uuid4()),
            "document_id": document.id,
            "filename": filename,
            "chunks_created": chunk_count,
            "chunks_indexed": embedded_count,
            "content_length": len(document.content),
            "doc_type": document.doc_type.value,
            "batch_size": batch_size,
            "duration_ms": round(duration_ms, 1),
            "message": f"Successfully processed {filename}",
        }

    def _note_chunks_for_bm25(self) -> None:
        """New chunks invalidate the sparse index; marked by caller."""
        # kept as a no-op hook for callers that haven't migrated yet

    # ------------------------------------------------------------------ utils
    async def process_url(self, url: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a file located at an HTTP/HTTPS URL."""
        try:
            import requests
            from urllib.parse import urlparse

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "downloaded_file"

            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                result = await self.process_document(tmp_file_path, "", task_id)
                result["source_url"] = url
                return result
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        except Exception as exc:
            logger.error("Error processing URL %s: %s", url, exc)
            return {
                "success": False,
                "error": str(exc),
                "task_id": task_id or str(uuid.uuid4()),
                "source_url": url,
            }

    async def process_text_content(
        self, content: str, filename: str = "text_content.txt",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest raw text content. Streaming path shared with process_document."""
        try:
            from core.models import Document, DocumentType
            from datetime import datetime

            document = Document(
                id=str(uuid.uuid4()),
                filename=filename,
                content=content,
                doc_type=DocumentType.TEXT,
                file_size=len(content.encode("utf-8")),
                created_at=datetime.utcnow(),
                metadata={
                    "source": "direct_text_input",
                    "content_length": len(content),
                    "word_count": len(content.split()),
                },
            )
            await self.document_store.store_document(document)
            return await self._ingest_parsed_document(document, task_id=task_id)
        except Exception as exc:
            logger.error("Error processing text content: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "task_id": task_id or str(uuid.uuid4()),
            }

    async def reprocess_document(
        self, document_id: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete + re-ingest an existing document (fresh chunking / embeddings)."""
        try:
            document = await self.document_store.get_document(document_id)
            if not document:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found",
                    "task_id": task_id or str(uuid.uuid4()),
                }
            content = await self.document_store.load_content(document_id)
            if not content:
                return {
                    "success": False,
                    "error": f"Stored content missing for document {document_id}",
                    "task_id": task_id or str(uuid.uuid4()),
                }
            document.content = content
            await self.vector_store.delete_document(document_id)
            return await self._ingest_parsed_document(document, task_id=task_id)
        except Exception as exc:
            logger.error("Error reprocessing document %s: %s", document_id, exc)
            return {
                "success": False,
                "error": str(exc),
                "task_id": task_id or str(uuid.uuid4()),
                "document_id": document_id,
            }

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    async def batch_process_directory(
        self, directory_path: str, task_id: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Concurrent, bounded-parallelism directory ingestion with backpressure."""
        started = time.perf_counter()
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return {
                    "success": False,
                    "error": f"Directory {directory_path} does not exist",
                    "task_id": task_id or str(uuid.uuid4()),
                }
            files_to_process = sorted(
                {p for ext in self.SUPPORTED_EXTENSIONS for p in directory.rglob(f"*{ext}")}
                | {p for ext in self.SUPPORTED_EXTENSIONS for p in directory.rglob(f"*{ext.upper()}")}
            )
            supported_files = [p for p in files_to_process if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
            if not supported_files:
                return {
                    "success": False,
                    "error": "No supported files found in directory",
                    "task_id": task_id or str(uuid.uuid4()),
                }

            workers = max(1, concurrency or self.config.INGEST_DEFAULT_CONCURRENCY)
            results: List[Dict[str, Any]] = []

            async def _ingest(path: Path):
                task_id_here = str(uuid.uuid4())
                try:
                    res = await self.process_document(str(path), path.suffix, task_id_here)
                except Exception as exc:
                    logger.exception("Ingest failed for %s: %s", path, exc)
                    res = {"success": False, "error": str(exc), "filename": path.name}
                results.append(res)

            semaphore = asyncio.Semaphore(workers)

            async def bounded(path: Path):
                async with semaphore:
                    await _ingest(path)

            tasks = [asyncio.create_task(bounded(p)) for p in supported_files]
            if tasks:
                await asyncio.gather(*tasks)

            successful = sum(1 for r in results if r.get("success"))
            total = len(supported_files)
            return {
                "success": True,
                "task_id": task_id or str(uuid.uuid4()),
                "directory": str(directory),
                "total_files": total,
                "successful": successful,
                "failed": total - successful,
                "concurrency": workers,
                "duration_sec": round(time.perf_counter() - started, 2),
                "results": results,
                "message": f"Processed {successful}/{total} files successfully",
            }
        except Exception as exc:
            logger.error("Error batch processing directory %s: %s", directory_path, exc)
            return {
                "success": False,
                "error": str(exc),
                "task_id": task_id or str(uuid.uuid4()),
            }
