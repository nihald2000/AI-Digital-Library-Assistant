"""Document store rewritten on SQLite for 1M+ documents.

Replaces one-JSON-file-per-document with:
  - `documents` table  -> indexed metadata (created_at / doc_type / category)
  - `document_content` -> full text as its own table so list operations
    never pull megabytes of bodies into RAM

The public API is unchanged:
  store_document, get_document, list_documents, update_document_metadata,
  delete_document, search_documents, get_stats

New at scale:
  * `get_documents_batch(ids)` - one round trip for N documents
  * `iter_documents()` - constant-RAM streaming across the whole library
  * pagination/filtering handled by SQLite indexes instead of full FS scans
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from core.models import Document, DocumentType
from services.sqlite_store import SQLiteStore, dumps_json, loads_json
import config

logger = logging.getLogger(__name__)

_DOCUMENT_COLUMNS = (
    "id, filename, doc_type, file_size, created_at, metadata, tags, summary,"
    " category, language, content_length, chunk_count"
)


class DocumentStoreService:
    def __init__(self, store: Optional[SQLiteStore] = None):
        self.config = config.config
        self.store = store or SQLiteStore(self.config.DB_PATH)
        # small LRU for hot single-document reads (UI / enrichment)
        self._cache: Dict[str, Document] = {}
        self._cache_size_limit = 256

    async def initialize(self) -> None:
        await self.store.initialize()

    # ------------------------------------------------------------ ingesting
    async def store_document(self, document: Document) -> bool:
        try:
            created_at = document.created_at
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            meta_row = (
                document.id,
                document.filename,
                document.doc_type.value if hasattr(document.doc_type, "value") else str(document.doc_type),
                document.file_size,
                created_at.isoformat(),
                dumps_json(document.metadata or {}),
                dumps_json(document.tags or []),
                document.summary,
                document.category,
                document.language,
                len(document.content or ""),
            )
            await self.store.execute_many(
                "INSERT OR REPLACE INTO documents"
                " (id, filename, doc_type, file_size, created_at, metadata, tags,"
                "  summary, category, language, content_length)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [meta_row],
            )
            await self.store.execute_many(
                "INSERT OR REPLACE INTO document_content(id, content) VALUES (?,?)",
                [(document.id, document.content or "")],
            )
            self._add_to_cache(document.id, document)
            return True
        except Exception as exc:
            logger.error("Error storing document %s: %s", document.id, exc)
            return False

    async def get_document(self, document_id: str) -> Optional[Document]:
        if document_id in self._cache:
            return self._cache[document_id]
        batch = await self.get_documents_batch([document_id], include_content=True)
        return batch.get(document_id)

    async def get_documents_metadata_batch(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """One SQL round trip for N documents (no content bodies)."""
        if not doc_ids:
            return {}
        unique_ids = list(dict.fromkeys(doc_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        rows = await self.store.fetch_all(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id IN ({placeholders})",
            unique_ids,
        )
        return {row["id"]: self._row_to_metadata(row) for row in rows}

    async def get_documents_batch(
        self, doc_ids: List[str], include_content: bool = False
    ) -> Dict[str, Document]:
        metas = await self.get_documents_metadata_batch(doc_ids)
        if not include_content:
            return {
                doc_id: self._metadata_to_document(meta)
                for doc_id, meta in metas.items()
            }
        chunks: List[str] = []
        per_batch = 900
        contents: Dict[str, str] = {}
        id_list = [
            doc_id for doc_id in metas
        ]
        for start in range(0, len(id_list), per_batch):
            part = id_list[start:start + per_batch]
            placeholders = ",".join("?" for _ in part)
            rows = await self.store.fetch_all(
                f"SELECT id, content FROM document_content WHERE id IN ({placeholders})",
                part,
            )
            for row in rows:
                contents[row["id"]] = row["content"]

        out: Dict[str, Document] = {}
        for doc_id, meta in metas.items():
            doc = self._metadata_to_document(meta)
            doc.content = contents.get(doc_id, "")
            out[doc_id] = doc
            self._add_to_cache(doc_id, doc)
        return out

    async def list_documents(
        self, limit: int = 50, offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Indexed pagination - no filesystem scanning at 1M docs."""
        try:
            sql = f"SELECT {_DOCUMENT_COLUMNS} FROM documents"
            where, params = self._build_where(filters)
            if where:
                sql += f" WHERE {where}"
            sql += " ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
            params = params + [int(limit), int(offset)]
            rows = await self.store.fetch_all(sql, params)
            metas = {row["id"]: self._row_to_metadata(row) for row in rows}
            # preview-only content loaded in shared accessor
            return [self._metadata_to_document(meta) for meta in metas.values()]
        except Exception as exc:
            logger.error("Error listing documents: %s", exc)
            return []

    async def iter_documents(
        self, batch_size: int = 1000, with_content: bool = True,
    ) -> AsyncIterator[Document]:
        """Constant-RAM streaming across the whole library."""
        offset = 0
        while True:
            sql = f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
            rows = await self.store.fetch_all(sql, (batch_size, offset))
            if not rows:
                return
            if with_content:
                ids = [row["id"] for row in rows]
                docs = await self.get_documents_batch(ids, include_content=True)
                for row in rows:
                    doc = docs.get(row["id"])
                    if doc:
                        yield doc
            else:
                for row in rows:
                    yield self._metadata_to_document(self._row_to_metadata(row))
            offset += batch_size

    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        where, params = self._build_where(filters)
        sql = "SELECT COUNT(1) AS c FROM documents"
        if where:
            sql += f" WHERE {where}"
        row = await self.store.fetch_one(sql, params)
        return int(row["c"]) if row else 0

    # --------------------------------------------------------------- helpers
    def _row_to_metadata(self, row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "doc_type": row["doc_type"],
            "file_size": row["file_size"],
            "created_at": row["created_at"],
            "metadata": loads_json(row["metadata"]) or {},
            "tags": loads_json(row["tags"]) or [],
            "summary": row["summary"],
            "category": row["category"],
            "language": row["language"],
            "content_length": row["content_length"],
            "chunk_count": row["chunk_count"],
        }

    def _metadata_to_document(self, meta: Optional[Dict[str, Any]]) -> Document:
        return Document(
            id=meta["id"],
            filename=meta["filename"],
            content="",  # content loaded lazily
            doc_type=DocumentType(meta["doc_type"]),
            file_size=meta.get("file_size") or 0,
            created_at=datetime.fromisoformat(meta["created_at"]),
            metadata=meta.get("metadata", {}) or {},
            tags=meta.get("tags", []) or [],
            summary=meta.get("summary"),
            category=meta.get("category"),
            language=meta.get("language"),
        )

    async def load_content(self, document_id: str) -> Optional[str]:
        row = await self.store.fetch_one(
            "SELECT content FROM document_content WHERE id=?", (document_id,)
        )
        return row["content"] if row else None

    def _build_where(self, filters: Optional[Dict[str, Any]]) -> List[Any]:
        if not filters:
            return ("", [])
        clauses: List[str] = []
        params: List[Any] = []
        for key, value in filters.items():
            if key == "doc_type":
                clauses.append("doc_type = ?")
                params.append(value)
            elif key == "category":
                clauses.append("category = ?")
                params.append(value)
            elif key == "language":
                clauses.append("language = ?")
                params.append(value)
            elif key == "filename_contains":
                clauses.append("filename LIKE ? ESCAPE '\\'")
                params.append(f"%{value.replace('%', r'%').replace('_', r'_').lower()}%")
            elif key == "created_after":
                clauses.append("created_at >= ?")
                params.append(self._iso(value))
            elif key == "created_before":
                clauses.append("created_at <= ?")
                params.append(self._iso(value))
            elif key == "tags":
                wanted = value if isinstance(value, list) else [value]
                tag_clauses = []
                for tag in wanted:
                    tag_clauses.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                clauses.append("(" + " OR ".join(tag_clauses) + ")")
            else:
                logger.debug("Unhandled filter key '%s' ignored", key)
        return (" AND ".join(clauses) if clauses else "", params)

    @staticmethod
    def _iso(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return str(value)

    # ------------------------------------------------------------- mutation
    async def update_document_metadata(self, document_id: str, updates: Dict[str, Any]) -> bool:
        row = await self.store.fetch_one(
            "SELECT metadata, tags, summary, category, language, chunk_count"
            " FROM documents WHERE id=?", (document_id,)
        )
        if not row:
            logger.warning("Document %s not found", document_id)
            return False
        current_meta = loads_json(row["metadata"]) or {}
        tags = loads_json(row["tags"]) or []
        summary = row["summary"]
        category = row["category"]
        language = row["language"]
        chunk_count = row["chunk_count"]

        for key, value in updates.items():
            if key == "metadata":
                if isinstance(value, dict) and isinstance(current_meta, dict):
                    current_meta.update(value)
                else:
                    current_meta = value
            elif key == "tags":
                tags = value
            elif key == "summary":
                summary = value
            elif key == "category":
                category = value
            elif key == "language":
                language = value
            elif key == "chunk_count":
                chunk_count = value

        await self.store.execute_many(
            "UPDATE documents SET metadata=?, tags=?, summary=?, category=?,"
            " language=?, chunk_count=? WHERE id=?",
            [(
                dumps_json(current_meta),
                dumps_json(tags or []),
                summary,
                category,
                language,
                chunk_count,
                document_id,
            )],
        )
        cached = self._cache.get(document_id)
        if cached:
            for name in ("tags", "summary", "category", "language"):
                if name in updates:
                    setattr(cached, name, updates[name])
        return True

    async def delete_document(self, document_id: str) -> bool:
        results = await self.store.fetch_all(
            "SELECT COUNT(1) AS c FROM documents WHERE id=?", (document_id,)
        )
        if not results or not results[0]["c"]:
            return False
        await self.store.execute_many(
            "DELETE FROM documents WHERE id = ?", [(document_id,)]
        )
        await self.store.execute_many(
            "DELETE FROM document_content WHERE id = ?", [(document_id,)]
        )
        self._cache.pop(document_id, None)
        return True

    async def search_documents(
        self, query: str, fields: Optional[List[str]] = None, limit: int = 100
    ) -> List[Document]:
        """Substring match on metadata columns - indexed, no full-body scan."""
        if not fields:
            fields = ["filename", "summary", "tags"]
        try:
            clauses, params = [], []
            for field in fields:
                if field == "filename":
                    clauses.append("filename LIKE ?")
                    params.append(f"%{query}%")
                elif field == "summary":
                    clauses.append("summary LIKE ?")
                    params.append(f"%{query}%")
                elif field == "tags":
                    clauses.append("tags LIKE ?")
                    params.append(f'%"{query}"%')
                elif field == "category":
                    clauses.append("category LIKE ?")
                    params.append(f"%{query}%")
            if not clauses:
                return []
            sql = (
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE ("
                + " OR ".join(clauses)
                + ") ORDER BY created_at DESC LIMIT ?"
            )
            params.append(limit)
            rows = await self.store.fetch_all(sql, params)
            return [self._metadata_to_document(self._row_to_metadata(r)) for r in rows]
        except Exception as exc:
            logger.error("Error searching documents: %s", exc)
            return []

    async def get_stats(self) -> Dict[str, Any]:
        row = await self.store.fetch_one("SELECT COUNT(1) AS c FROM documents")
        total = int(row["c"]) if row else 0
        type_counts = {}
        rows = await self.store.fetch_all(
            "SELECT doc_type, COUNT(1) AS c FROM documents GROUP BY doc_type"
        )
        for r in rows:
            type_counts[r["doc_type"]] = int(r["c"])
        chunk_rows = await self.store.fetch_all(
            "SELECT COUNT(1) AS c, SUM(LENGTH(content)) AS total_chars FROM chunks WHERE deleted=0"
        )
        total_chunks = int(chunk_rows[0]["c"]) if chunk_rows else 0
        total_chars = int(chunk_rows[0]["total_chars"] or 0) if chunk_rows else 0
        return {
            "total_documents": total,
            "document_types": type_counts,
            "total_chunks": total_chunks,
            "total_size_bytes": total_chars,
            "cache_size": len(self._cache),
        }

    # ---------------------------------------------------------------- cache
    def _add_to_cache(self, doc_id: str, doc: Document) -> None:
        if len(self._cache) >= self._cache_size_limit:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)
        self._cache[doc_id] = doc
