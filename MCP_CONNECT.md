# Attaching the Digital Library knowledge base anywhere (MCP)

The same server backs the Gradio app **and** a standard Model Context Protocol
endpoint, so any MCP-capable agent can use your library as a portable private
knowledge base.

## Endpoints

| Client type | URL / command |
|---|---|
| Space (live) | `https://nihal2000-ai-digital-library-assistant-agent.hf.space/mcp` |
| Claude / GPT HTTP+Remote hosts | the Space URL above |
| Local stdio | `python mcp_server.py` (FastMCP stdio) |

## What agents get

| Tool | Purpose | PII behavior |
|---|---|---|
| `semantic_search` | hybrid dense+sparse retrieval | snippets arrive **masked**: `[EMAIL_1]`/`[PAN_1]`-style placeholders |
| `answer_question` | RAG answer (LLM runs in *our* process) | question + context masked before the LLM; real values restored in the returned answer; sources masked |
| `pii_restore_placeholders` | unmask the agent's final reply | swaps placeholders back to real values, memory-only (TTL 1 h), never on disk |
| `summarize_content`, `generate_tags`, `scan_pii`, `pii_add_custom_rule` … | generative + PII utilities | masked inputs everywhere |
| `ingest_*`, `ingest_queued`, `job_status`, `list_documents_for_ui`, `library_stats` | library management | n/a |

## The PII contract (what any client should know)

1. Personal data never crosses the tool boundary raw. Retrieval output is
   rewritten to stable `[TYPE_N]` placeholders per real value.
2. When an agent composes its own reply from `semantic_search` snippets, it
   writes the placeholders as-is, then calls `pii_restore_placeholders` with
   its final user-facing text. Resolved placeholders return as real values.
3. Mappings live only in process memory with a 1-hour TTL — nothing is
   persisted, logged, or exported.
4. `semantic_search` return carries `pii_placeholders` (count) so a client can
   notice masking happened.

## Example: Claude Desktop (HTTP transport)

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "digital-library": {
      "url": "https://nihal2000-ai-digital-library-assistant-agent.hf.space/mcp"
    }
  }
}
```

Then just ask: *"Using digital-library, what does the outreach letter say
about the officer's email?"* — Claude searches (sees `[EMAIL_2]`), composes,
and calls `pii_restore_placeholders` before showing you the address.

## Known bounds

- Person/organization **names** are masked only when optional NER is enabled
  (`PII_NER_ENABLED=true`, needs `presidio-analyzer`). Everything with a
  structure (emails, phones, Aadhaar/PAN/SSN/IBAN/cards…) is masked always.
- The `voice_qa` / `generate_podcast` LlamaIndex path predates the boundary
  protocol and is the one remaining raw-text leg (documented in SCALING.md).
- PII detection is regex+checksum based: checksum-bearing government ids are
  high-precision; free-text names are heuristic. Treat the mask as defense,
  not a guarantee.
