"""Offline tests for the MCP-boundary PII protocol (services/mcp_pii.py).

Run: python -m pytest tests/test_mcp_pii.py -q
Uses a stub PIIService - no model boot, no network.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import mcp_pii
from services.mcp_pii import mask_for_mcp, masked_source_dicts


@dataclass
class CannedFinding:
    entity: str = "IN_PAN"
    placeholder: str = "[IN_PAN_1]"


@dataclass
class CannedMasked:
    masked_text: str
    masked_count: int
    findings: list = field(default_factory=list)


class StubPiiService:
    """Mimics the PIIService contract: mask_for_agent / restore_for_user."""

    def __init__(self, masked_text: str = "", count: int = 1, raise_on_mask: bool = False):
        self.masked_text = masked_text
        self.count = count
        self.raise_on_mask = raise_on_mask
        self.namespaces: Dict[str, Dict[str, str]] = {}
        self.calls: list = []

    def mask_for_agent(self, text: str, namespace: str, register: bool = True,
                       disabled_types: Optional[Sequence[str]] = None):
        self.calls.append((namespace, register))
        if self.raise_on_mask:
            raise RuntimeError("boom")
        assert namespace == mcp_pii.MCP_PII_NAMESPACE
        return CannedMasked(masked_text=self.masked_text, masked_count=self.count)

    def restore_for_user(self, text: str, namespace: str) -> str:
        return text


class SentenceResult:
    def __init__(self, content: str, meta: Dict):
        self._payload = {"content": content, "metadata": dict(meta)}

    def to_dict(self):
        return self._payload


def test_masked_source_dicts_masks_content_and_metadata():
    stub = StubPiiService(masked_text="[IN_PAN_1]", count=1)
    rows, findings = masked_source_dicts(
        [SentenceResult("PAN ABCDE1234F user", {"filename": "mailto:x@y.in"})],
        stub,
    )
    assert rows[0]["content"] == "[IN_PAN_1]"
    assert rows[0]["metadata"]["filename"] == "[IN_PAN_1]"
    assert findings == 2
    # every mask call went through the shared MCP namespace
    assert all(ns == mcp_pii.MCP_PII_NAMESPACE for ns, _ in stub.calls)


def test_masked_source_dicts_empty_content_and_nonstring_meta():
    stub = StubPiiService()
    rows, findings = masked_source_dicts(
        [SentenceResult("", {"size": 42, "score": 0.9})],
        stub,
    )
    assert rows[0]["content"] == ""
    assert findings == 0 and stub.calls == []  # nothing to mask -> no calls


def test_mask_failure_passes_text_through():
    stub = StubPiiService(raise_on_mask=True)
    out, findings = mask_for_mcp("raw snippet", stub)
    assert out == "raw snippet" and findings == 0


def test_no_pii_service_passthrough():
    out, findings = mask_for_mcp("raw", None)
    assert out == "raw" and findings == 0
