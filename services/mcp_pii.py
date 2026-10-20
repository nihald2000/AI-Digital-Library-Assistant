"""PII boundary helpers for MCP tool output.

External MCP clients (Claude Desktop, ChatGPT, any agent host) treat this
server as a portable knowledge base. No sensitive value may ever cross the
tool boundary in raw form, so:

1. Retrieval outputs (search results, RAG sources) are rewritten to
   ``[EMAIL_1]``-style placeholders before the tool returns them.
2. Generation tools mask what *their* LLM sees and restore real values in the
   reply inside this process (memory-only TTL mappings, never persisted).
3. For snippets an agent composes itself (it called ``semantic_search``), the
   agent writes placeholders through its reply, then calls
   ``pii_restore_placeholders`` to unmask the final user-facing text.

Placeholders are stable per value inside the TTL window (per-process, shared
namespace), so identical values map to identical placeholders across calls.

Kept dependency-light (no FastMCP / no service boot) for offline testing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Shared namespace: one mapping cache for all MCP persona calls. TTL-pruned by
# PIIService (_prune_mappings, PII_MAPPING_TTL_SECONDS).
MCP_PII_NAMESPACE = "mcp-knowledge-base"


def mask_for_mcp(text: str, pii_service: Any, register: bool = True) -> Tuple[str, int]:
    """Mask raw text at the MCP boundary. Returns (masked_text, finding_count).

    Never raises: masking is a security boundary, so on failure we log and
    pass the text through *masked or not* is a caller decision - default here
    matches the rest of the pipeline (log + pass-through, PII_MASKING_ENABLED
    is the master switch).
    """
    if not text or not pii_service:
        return text, 0
    try:
        masked = pii_service.mask_for_agent(text, namespace=MCP_PII_NAMESPACE, register=register)
        return masked.masked_text, masked.masked_count
    except Exception as exc:
        logger.warning("MCP PII masking failed (%s: %.150s)", exc.__class__.__name__, exc)
        return text, 0


def masked_source_dicts(results: List[Any], pii_service: Any) -> Tuple[List[Dict[str, Any]], int]:
    """Mask every snippet/metadata string coming out of search results.

    Returns (masked_row_dicts, total_findings). Raw ``.to_dict()`` payloads
    must never be returned to an MCP client directly.
    """
    rows: List[Dict[str, Any]] = []
    findings = 0
    for result in results:
        data = dict(result.to_dict() if hasattr(result, "to_dict") else result)
        masked, count = mask_for_mcp(str(data.get("content") or ""), pii_service)
        data["content"] = masked
        findings += count
        for key, value in list((data.get("metadata") or {}).items()):
            if isinstance(value, str) and value:
                data["metadata"][key], count = mask_for_mcp(value, pii_service)
                findings += count
        rows.append(data)
    return rows, findings
