"""SQLite-backed storage core for scaling to 1M+ documents.

Single WAL-mode database file that holds:
  - documents / document_content  (document store)
  - chunks / chunk_vectors        (vector store metadata + embeddings as BLOBs)
  - meta                          (service-level bookkeeping)

Design notes
------------
* WAL journal -> concurrent readers never block the ingestion writer.
* All access goes through a small thread pool; SQLite handles cross-thread
  concurrency itself (check_same_thread=False + busy_timeout).
* Writes are serialized with an asyncio lock so faiss/BM25 bookkeeping and
  database rows stay consistent without distributed locks.
* Embeddings are stored as BLOBs == source of truth. The FAISS index is a
  rebuildable cache; BM25 snowball index is a rebuildable cache too. Any crash
  can be fully recovered by re-running compaction from SQLite.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class SQLiteWriteLock:
    """Serialize write transactions across asyncio tasks."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class SQLiteStore:
    def __init__(self, db_path: str, executor_max_workers: int = 4):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=executor_max_workers, thread_name_prefix="sqs")
        self._write_lock = asyncio.Lock()
        self._local = threading.local()

    # ------------------------------------------------------------------ util
    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=60.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-65536")  # 64MB page cache
            conn.execute("PRAGMA mmap_size=1073741824")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        return conn

    async def initialize(self) -> None:
        def _init() -> None:
            conn = self._connect()
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        doc_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        metadata TEXT,
                        tags TEXT,
                        summary TEXT,
                        category TEXT,
                        language TEXT,
                        content_length INTEGER NOT NULL DEFAULT 0,
                        chunk_count INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS idx_documents_created
                        ON documents(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_documents_type
                        ON documents(doc_type);
                    CREATE INDEX IF NOT EXISTS idx_documents_category
                        ON documents(category);

                    CREATE TABLE IF NOT EXISTS document_content (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id TEXT UNIQUE,
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL DEFAULT 0,
                        start_pos INTEGER NOT NULL DEFAULT 0,
                        end_pos INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT,
                        deleted INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_doc
                        ON chunks(document_id, deleted);
                    CREATE INDEX IF NOT EXISTS idx_chunks_deleted
                        ON chunks(deleted);

                    CREATE TABLE IF NOT EXISTS chunk_vectors (
                        vector_id INTEGER PRIMARY KEY,
                        embedding BLOB NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    );
                    """
                )

            # cheap cache of current schema version for future migrations
            cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
            if cur.fetchone() is None:
                conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(_SCHEMA_VERSION),))
                conn.commit()
        await asyncio.get_running_loop().run_in_executor(self._executor, _init)

    # -------------------------------------------------------------- executor
    async def _run(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    # ---------------------------------------------------------------- writes
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Single write statement, returns lastrowid."""
        async with self._write_lock:
            def _write():
                conn = self._connect()
                cur = conn.execute(sql, params)
                conn.commit()
                return cur.lastrowid
            return await self._run(_write)

    async def execute_many(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return

        async with self._write_lock:
            def _write_many():
                conn = self._connect()
                with conn:
                    conn.executemany(sql, rows)
            await self._run(_write_many)

    async def script(self, statements: Sequence[str]) -> None:
        async with self._write_lock:
            def _script():
                conn = self._connect()
                with conn:
                    for stmt in statements:
                        conn.execute(stmt)
            await self._run(_script)

    # ---------------------------------------------------------------- reads
    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> List[Any]:
        def _read():
            conn = self._connect()
            return conn.execute(sql, params).fetchall()
        return await self._run(_read)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
        def _read_one():
            conn = self._connect()
            return conn.execute(sql, params).fetchone()
        row = await self._run(_read_one)
        return row

    # ----------------------------------------------------------- stream read
    async def stream(
        self,
        sql: str,
        params: Sequence[Any] = (),
        batch_size: int = 5000,
    ):
        """Async generator that pages through rows ordered by rowid.

        Used by compaction rebuilds / BM25 builds / LlamaIndex sync where we
        must never load the entire corpus into RAM.
        """
        cursor_sql = sql
        last_rowid = 0
        while True:
            rows = await self.fetch_all(
                cursor_sql + f" LIMIT {int(batch_size)}", (*params, last_rowid)
            )
            if not rows:
                return
            for row in rows:
                yield row
                last_rowid = row["rowid"] if isinstance(row, sqlite3.Row) and "rowid" in row.keys() else last_rowid + 1
            if len(rows) < batch_size:
                return

    # ------------------------------------------------------------- maintenance
    async def vacuum(self) -> None:
        def _vacuum():
            self._connect().execute("VACUUM")
        await self._run(_vacuum)

    async def close(self) -> None:
        def _close():
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                try:
                    conn.commit()
                except Exception:
                    pass
                conn.close()
                self._local.conn = None
        # close the thread-local connection for the current thread only; the
        # executor threads' connections close on garbage collection
        await self._run(_close)
        self._executor.shutdown(wait=False, cancel_futures=True)


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def loads_json(raw: Optional[str]) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None
