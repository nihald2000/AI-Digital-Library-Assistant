"""Persistent ingestion queue: split parsing/embedding out of the request path.

Jobs live in SQLite (`ingest_jobs`), so:
  * a crash mid-document (bad PDF, OCR stall) can't stall the API process,
  * restarts resume where they left off (pending / running jobs are retried),
  * progress and failures are auditable per job.

The worker runs on a dedicated thread with its own event loop (same pattern as
the maintenance thread). One job at a time by default
(INGEST_QUEUE_MAX_CONCURRENT_JOBS) so heavy OCR doesn't thrash the store.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.sqlite_store import SQLiteStore
import config

logger = logging.getLogger(__name__)

_JOB_COLUMNS = (
    "id, job_type, target, status, concurrency, progress_done, progress_total,"
    " error, extra, created_at, started_at, finished_at"
)


class IngestionQueueService:
    """Enqueue documents/directories; a worker drains them in the background."""

    def __init__(self, ingestion_tool: Any, store: SQLiteStore):
        self.ingestion_tool = ingestion_tool
        self.store = store
        self.config = config.config
        self._workers: List[asyncio.Task] = []
        self._stopping = False

    # ------------------------------------------------------------------ boot
    async def initialize(self) -> None:
        def _ddl():
            conn = self.store._connect()
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS ingest_jobs (
                        id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,          -- document | directory | url | text
                        target TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        concurrency INTEGER NOT NULL DEFAULT 1,
                        progress_done INTEGER NOT NULL DEFAULT 0,
                        progress_total INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        extra TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_jobs_status
                        ON ingest_jobs(status, created_at);
                    CREATE INDEX IF NOT EXISTS idx_jobs_created
                        ON ingest_jobs(created_at DESC);
                    """
                )
        await asyncio.get_running_loop().run_in_executor(None, _ddl)

    async def recover_stale_running(self) -> int:
        """Crash-recovery: requeue any 'running' job at worker boot.

        Called exactly once per process, before any worker exists, so every
        row in 'running' state belongs to a dead process (the queue is durable
        SQLite, but the old docstring promised restart-safety it never had).
        """
        def _requeue():
            conn = self.store._connect()
            with conn:
                conn.execute(
                    "UPDATE ingest_jobs SET status='pending', started_at=NULL"
                    " WHERE status='running'"
                )
                return conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]
        try:
            requeued = await asyncio.get_running_loop().run_in_executor(None, _requeue)
            if requeued:
                logger.info("Requeued %s stale 'running' ingestion job(s) from a previous run", requeued)
            return requeued
        except Exception as exc:
            logger.warning("Stale-running recovery skipped: %s", exc)
            return 0

    # ------------------------------------------------------------- enqueue
    async def enqueue(
        self, job_type: str, target: str,
        concurrency: int = 1, extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        await self.store.execute_many(
            "INSERT INTO ingest_jobs (id, job_type, target, status, concurrency, extra, created_at)"
            " VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            [(
                job_id,
                job_type,
                target,
                max(1, int(concurrency)),
                json.dumps(extra or {}) if extra else None,
                datetime.utcnow().isoformat(),
            )],
        )
        logger.info("Queued %s job %s -> %s", job_type, job_id[:8], target)
        return {"success": True, "job_id": job_id, "status": "pending", "job_type": job_type, "target": target}

    # -------------------------------------------------------------- claims
    async def claim_next(self) -> Optional[Dict[str, Any]]:
        """Atomically claim the oldest pending job (restart-safe retries)."""
        rows = await self.store.fetch_all(
            "SELECT * FROM ingest_jobs WHERE status='pending' ORDER BY created_at LIMIT 1"
        )
        if not rows:
            return None
        job = dict(rows[0])
        await self.store.execute_many(
            "UPDATE ingest_jobs SET status='running', started_at=?, progress_done=0"
            " WHERE id=? AND status='pending'",
            [(datetime.utcnow().isoformat(), job["id"])],
        )
        job["status"] = "running"
        return job

    async def _complete(self, job_id: str, result: Dict[str, Any]) -> None:
        extra = {"result_summary": {
            key: result.get(key) for key in (
                "total_files", "successful", "failed", "chunks_created", "chosen_method"
            ) if key in result
        }}
        await self.store.execute_many(
            "UPDATE ingest_jobs SET status='done', progress_done=progress_total,"
            " finished_at=?, extra=? WHERE id=?",
            [(datetime.utcnow().isoformat(), json.dumps(extra), job_id)],
        )

    async def _fail(self, job_id: str, error: str) -> None:
        await self.store.execute_many(
            "UPDATE ingest_jobs SET status='failed', error=?, finished_at=? WHERE id=?",
            [(
                error[:2000],
                datetime.utcnow().isoformat(),
                job_id,
            )],
        )

    # ------------------------------------------------------------- worker
    async def process_claimed(self, job: Dict[str, Any]) -> None:
        """Run one claimed job (document | directory | url | text)."""
        job_id, target, job_type = job["id"], job["target"], job["job_type"]
        try:
            if job_type == "directory":
                concurrency = int(job.get("concurrency") or self.config.INGEST_DEFAULT_CONCURRENCY)
                result = await self.ingestion_tool.batch_process_directory(
                    target, task_id=job_id, concurrency=concurrency
                )
            elif job_type == "document":
                result = await self.ingestion_tool.process_document(target, Path(target).suffix)
            elif job_type == "url":
                result = await self.ingestion_tool.process_url(target)
            elif job_type == "text":
                extra = json.loads(job.get("extra") or "{}")
                result = await self.ingestion_tool.process_text_content(
                    extra.get("content", ""), filename=extra.get("filename") or "queued_text.txt"
                )
            else:
                raise ValueError(f"Unknown job_type {job_type}")
            if result.get("success"):
                await self._complete(job_id, result)
            else:
                await self._fail(job_id, result.get("error") or "Ingestion failed")
        except Exception as exc:
            logger.exception("Queue job %s failed: %s", job_id[:8], exc)
            await self._fail(job_id, str(exc))

    async def run_worker_loop(self) -> None:
        """Poll the queue; durable across restarts."""
        interval = max(0.2, self.config.INGEST_QUEUE_POLL_SECONDS)
        logger.info("Ingestion queue worker started (poll %.0fms)", interval * 1000)
        while not self._stopping:
            try:
                job = await self.claim_next()
                if job is None:
                    await asyncio.sleep(interval)
                    continue
                logger.info("Worker picked up %s job %s", job["job_type"], job["id"][:8])
                await self.process_claimed(job)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # keep the worker alive no matter what
                logger.exception("Queue worker loop error: %s", exc)
                await asyncio.sleep(min(5.0, interval * 5))

    # ----------------------------------------------------------- reporting
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = await self.store.fetch_one(
            f"SELECT {_JOB_COLUMNS} FROM ingest_jobs WHERE id=?", (job_id,)
        )
        return dict(row) if row else None

    async def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = await self.store.fetch_all(
            f"SELECT {_JOB_COLUMNS} FROM ingest_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        rows = await self.store.fetch_all(
            "SELECT status, COUNT(1) AS c FROM ingest_jobs GROUP BY status"
        )
        counts = {row["status"]: int(row["c"]) for row in rows}
        return {
            "enabled": self.config.INGEST_QUEUE_ENABLED,
            "worker_alive": bool(self._workers) and not all(t.done() for t in self._workers),
            "jobs_by_status": counts,
            "poll_seconds": self.config.INGEST_QUEUE_POLL_SECONDS,
        }
