"""Offline tests for the Space-restart backup service (snapshot/restore).

Run: python -m pytest tests/test_backup_service.py -q
Network calls are stubbed - everything here is local tar/extraction logic.
"""
from __future__ import annotations

import asyncio
import io
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from services.backup_service import BackupService


def _cfg(data_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        DATA_DIR=str(data_dir),
        PERSIST_BACKUP_ENABLED=True,
        PERSIST_BACKUP_REPO="owner/library-backup",
        PERSIST_BACKUP_INTERVAL_SECONDS=0.0,
        PERSIST_BACKUP_MAX_FILE_MB=1,
        HUGGINGFACE_API_KEY="fake-token",
    )


def _seed_library(data_dir: Path) -> None:
    """Fake library layout: sqlite-ish files, faiss, documents."""
    (data_dir / "faiss").mkdir(parents=True, exist_ok=True)
    (data_dir / "documents" / "uploads").mkdir(parents=True, exist_ok=True)
    (data_dir / "library.db").write_bytes(b"sqlite-magic" * 10)
    (data_dir / "faiss" / "content_index.faiss").write_bytes(b"faiss-index" * 5)
    (data_dir / "documents" / "uploads" / "paper.txt").write_text("hello library")


def test_snapshot_round_trip(tmp_path, monkeypatch):
    src, staging = tmp_path / "src_data", tmp_path / "hub"
    staging.mkdir()
    _seed_library(src)

    svc = BackupService(_cfg(src))
    assert svc.enabled, "repo + token + enabled must arm the service"

    # stub the upload: move the archive into a fake hub dir
    def _fake_upload(archive_path: str) -> None:
        Path(staging / "library-snapshot.tar.gz").write_bytes(Path(archive_path).read_bytes())

    monkeypatch.setattr(svc, "_upload", _fake_upload)
    assert svc.snapshot() is not None

    flushed = asyncio.run(svc.flush_if_due())
    assert flushed == "snapshot -> owner/library-backup", flushed
    assert (staging / "library-snapshot.tar.gz").exists()

    # second flush with no db change is a no-op
    svc._last_flush_mtime = (src / "library.db").stat().st_mtime
    assert asyncio.run(svc.flush_if_due()) == ""

    # ---- restore into a fresh container dir ----
    fresh = tmp_path / "fresh_data"
    fresh_svc = BackupService(_cfg(fresh))

    def _fake_fetch(name=None):
        return str(staging / "library-snapshot.tar.gz")

    monkeypatch.setattr(fresh_svc, "_fetch_snapshot", lambda: str(staging / "library-snapshot.tar.gz"))
    assert fresh_svc.restore() is True
    assert (fresh / "library.db").read_bytes() == b"sqlite-magic" * 10
    assert (fresh / "faiss" / "content_index.faiss").exists()
    assert (fresh / "documents" / "uploads" / "paper.txt").read_text() == "hello library"

    # restore is refused when local data already exists (never clobber)
    monkeypatch.setattr(svc, "_fetch_snapshot", lambda: str(staging / "library-snapshot.tar.gz"))
    assert svc.restore() is False


def test_restore_refuses_path_traversal(tmp_path):
    data_dir = tmp_path / "victim"
    data_dir.mkdir()
    outside = tmp_path / "evil.txt"

    archive = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../evil.txt")
        payload = io.BytesIO(b"boom")
        info.size = payload.getbuffer().nbytes
        tar.addfile(info, payload)

    svc = BackupService(_cfg(data_dir))
    with pytest.raises(ValueError):
        svc._extract(archive)
    assert not outside.exists()


def test_disabled_when_secrets_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.PERSIST_BACKUP_REPO = None
    assert BackupService(cfg).enabled is False
    cfg2 = _cfg(tmp_path)
    cfg2.HUGGINGFACE_API_KEY = None
    assert BackupService(cfg2).enabled is False


def test_snapshot_respects_max_file_mb(tmp_path):
    _seed_library(tmp_path)
    big = tmp_path / "podcasts" / "big.mp3"
    big.parent.mkdir(exist_ok=True)
    big.write_bytes(b"x" * (2 * 1024 * 1024))  # > 1MB cap

    svc = BackupService(_cfg(tmp_path))
    archive = svc.snapshot()
    assert archive is not None
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "library.db" in names and "podcasts/big.mp3" not in names
    Path(archive).unlink()
