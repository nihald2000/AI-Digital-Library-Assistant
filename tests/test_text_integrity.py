"""Chunks must preserve the characters PII detection and checksums live on.

Run: python -m pytest tests/test_text_integrity.py -q

Regressions this guards: the old "aggressive cleaning for embeddings" chain
destroyed '@' and '+' in stored chunk text - emails/phones/country codes got
silently corrupted, which degraded both retrieval and the PII mask boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chunker import TextChunker


def test_chunker_preserves_emails_phones_and_identifiers():
    chunker = TextChunker()
    text = (
        "Library outreach letter: contact Priya Sharma at priya.sharma@govt.in "
        "or +91 9810012345. Her PAN is ABCDE1234F and the officer writes from "
        "ravi.branch@bank.co.in on behalf of the branch."
    )
    chunks = chunker.optimize_chunks_for_embedding(chunker.chunk_document("doc1", text))
    assert chunks, "expected at least one chunk"
    joined = " ".join(c.content for c in chunks)
    for token in ("priya.sharma@govt.in", "+91 9810012345", "ABCDE1234F", "ravi.branch@bank.co.in"):
        assert token in joined, f"ingestion corrupted {token!r}: {joined[:200]}"

    # whitespace collapses that used to matter still happen
    assert "\n" not in joined and "  " not in joined


def test_optimize_is_idempotent_and_metadata_tagged():
    chunker = TextChunker()
    chunks = chunker.optimize_chunks_for_embedding(
        chunker.chunk_document("doc2", "Simple sentence one is here plus short second line.")
    )
    assert all(c.metadata.get("optimized_for_embedding") for c in chunks)
    double = chunker.optimize_chunks_for_embedding(chunks)
    assert [c.content for c in double] == [c.content for c in chunks]
