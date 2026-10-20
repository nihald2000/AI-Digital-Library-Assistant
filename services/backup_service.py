"""Data persistence across HF Space restarts/rebuilds.

Space containers are wiped whenever the Space restarts (upgrade, code push,
hardware change). Everything the library depends on lives under ``DATA_DIR``
(SQLite ``library.db``, FAISS index, BM25 arrays, uploaded documents), so two
survival strategies exist:

1. **Persistent Storage** (paid add-on): auto-detected at ``/data`` in
   ``config.py`` - no code needed here.
2. **Free snapshot fallback** (this file): a ``tar.gz`` of the data tree is
   uploaded to a *private* HF dataset repo (``PERSIST_BACKUP_REPO`` +
   ``HF_TOKEN`` secrets). On boot ``restore()`` rehydrates when the local
   data dir is empty; a debounced ``flush_if_due()`` runs in the maintenance
   loop whenever ``library.db`` changes.

Offline-testable: the network calls (``_upload`` / snapshot fetch) are tiny
methods, stubbed in tests; tar creation/extraction and the path-traversal
guard are fully local.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_SNAPSHOT_NAME = "library-snapshot.tar.gz"


class BackupService:
    """Snapshot/restore the whole data tree via a private HF dataset repo."""

    def __init__(self, cfg: Any = None):
        cfg = cfg or config.config
        self.config = cfg
        self.data_dir = Path(getattr(cfg, "DATA_DIR", "./data"))
        self.repo = getattr(cfg, "PERSIST_BACKUP_REPO", None)
        self.token = getattr(cfg, "HUGGINGFACE_API_KEY", None) or getattr(cfg, "HF_TOKEN", None)
        self.max_file_bytes = max(0, int(getattr(cfg, "PERSIST_BACKUP_MAX_FILE_MB", 200))) * 1024 * 1024
        self.enabled = bool(
            getattr(cfg, "PERSIST_BACKUP_ENABLED", True) and self.repo and self.token
        )
        self._last_flush_mtime: float = -1.0
        self._last_flush_time: float = 0.0
        self._flush_inflight = False
        if getattr(cfg, "PERSIST_BACKUP_ENABLED", True) and not self.enabled:
            missing = "PERSIST_BACKUP_REPO secret" if not self.repo else "HF_TOKEN secret"
            logger.warning(
                "Data backup disabled (missing %s) - Space restarts will reset the library; "
                "consider Persistent Storage or set both secrets.", missing
            )

    # ------------------------------------------------------------- restore
    def restore(self) -> bool:
        """Rehydrate the data dir from the repo snapshot on a fresh container.

        Runs synchronously in ``AppServices.__init__`` *before* the stores
        open their files. Only restores into an empty dir, so newer local
        data can never be clobbered by an older snapshot.
        """
        if not self.enabled or (self.data_dir / "library.db").exists():
            return False
        archive = self._fetch_snapshot()
        if archive is None:
            logger.info("No snapshot in %s yet - starting with an empty library", self.repo)
            return False
        try:
            self._extract(archive)
            self._last_flush_mtime = (self.data_dir / "library.db").stat().st_mtime
            logger.info("Restored library snapshot from %s into %s", self.repo, self.data_dir)
            return True
        finally:
            Path(archive).unlink(missing_ok=True)

    def _fetch_snapshot(self) -> Optional[str]:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            logger.warning("huggingface_hub unavailable; restore skipped: %s", exc)
            return None
        try:
            return hf_hub_download(
                repo_id=self.repo,
                filename=_SNAPSHOT_NAME,
                repo_type="dataset",
                token=self.token,
                cache_dir=tempfile.gettempdir(),
            )
        except Exception as exc:
            logger.info("Snapshot fetch failed (%s: %s)", exc.__class__.__name__, str(exc)[:120])
            return None

    def _extract(self, archive: Path) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        base = os.path.realpath(self.data_dir)
        with tarfile.open(archive, "r:gz") as tar:
            # path-traversal guard before touching the disk
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(base, member.name))
                if os.path.commonpath([target, base]) != base:
                    raise ValueError(f"unsafe snapshot path rejected: {member.name}")
            tar.extractall(self.data_dir, filter="data")

    # ------------------------------------------------------------- flush
    async def flush_if_due(self) -> str:
        """Debounced push when ``library.db`` changed since the last snapshot."""
        if not self.enabled or self._flush_inflight:
            return ""
        db_path = self.data_dir / "library.db"
        try:
            mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
        except OSError:
            return ""
        if mtime == self._last_flush_mtime:
            return ""
        if time.time() - self._last_flush_time < float(self.config.PERSIST_BACKUP_INTERVAL_SECONDS):
            return "pending"
        self._flush_inflight = True
        loop = asyncio.get_running_loop()
        try:
            archive = await loop.run_in_executor(None, self.snapshot)
            if archive is None:
                return ""
            await loop.run_in_executor(None, self._upload, str(archive))
            self._last_flush_mtime = mtime
            self._last_flush_time = time.time()
            return f"snapshot -> {self.repo}"
        except Exception as exc:
            logger.warning("Snapshot flush failed (%s: %.150s)", exc.__class__.__name__, str(exc))
            return ""
        finally:
            self._flush_inflight = False

    def snapshot(self) -> Optional[Path]:
        """Checkpoint SQLite WAL, then tar the data tree to a temp archive.

        Files above ``PERSIST_BACKUP_MAX_FILE_MB`` (podcast audio blobs, say)
        are skipped so a repo upload can never balloon.
        """
        if not self.enabled:
            return None
        self._checkpoint_wal()
        fd, tmp = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        skipped: list[str] = []
        with tarfile.open(tmp, "w:gz") as tar:
            for path in sorted(self.data_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.data_dir).as_posix()
                if "__pycache__" in path.parts or path.name.startswith("multi_probe") \
                        or path.name == _SNAPSHOT_NAME:
                    continue
                try:
                    if path.stat().st_size > self.max_file_bytes:
                        skipped.append(rel)
                        continue
                except OSError:
                    continue
                tar.add(str(path), arcname=rel)
        if skipped:
            logger.info("Snapshot skipped %d oversize file(s), e.g. %s", len(skipped), skipped[0])
        return Path(tmp)

    def _checkpoint_wal(self) -> None:
        """Flush the WAL into the main db so the tar is self-contained."""
        try:
            conn = sqlite3.connect(str(self.data_dir / "library.db"), timeout=10)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("WAL checkpoint skipped: %s", exc)

    def _upload(self, archive_path: str) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        api.create_repo(self.repo, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=archive_path,
            path_in_repo=_SNAPSHOT_NAME,
            repo_id=self.repo,
            repo_type="dataset",
        )
        Path(archive_path).unlink(missing_ok=True)
