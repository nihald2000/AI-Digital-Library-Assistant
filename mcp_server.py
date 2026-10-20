"""FastMCP server for the AI Digital Library Assistant.

Wired through services.bootstrap so the same storage core + hybrid search
pipeline backs both the Gradio app and this MCP entry point.
"""
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from services.bootstrap import AppServices
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing services for FastMCP (SQLite + hybrid retrieval)...")
services = AppServices()
vector_store_service = services.vector_store
document_store_service = services.document_store
mcp = FastMCP("")
logger.info("FastMCP server initialized.")


# --------------------------------------------------------------------------
# PII protocol for external MCP clients (Claude Desktop, ChatGPT, ...)
# See services/mcp_pii.py for the full contract: retrieval outputs are masked
# to [TYPE_N] placeholders (stable per value, TTL-bound, memory-only), and
# pii_restore_placeholders unmask the agent's final user-facing reply.
# --------------------------------------------------------------------------
from services.mcp_pii import MCP_PII_NAMESPACE as _MCP_PII_NS
from services.mcp_pii import mask_for_mcp, masked_source_dicts  # noqa: E402


@mcp.tool()
async def ingest_document(file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Process and index a document from a local file path for searching.
    Automatically determines file_type if not provided.
    """
    logger.info(f"Tool 'ingest_document' called with file_path: {file_path}, file_type: {file_type}")
    try:
        actual_file_type = file_type
        if not actual_file_type:
            actual_file_type = Path(file_path).suffix.lower().strip('.')
            logger.info(f"Inferred file_type: {actual_file_type}")
        result = await services.ingestion_tool.process_document(file_path, actual_file_type)
        logger.info(f"Ingestion result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in 'ingest_document' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def ingest_directory(directory_path: str, concurrency: int = 4) -> Dict[str, Any]:
    """
    Bulk ingest every supported file under a directory, synchronously.
    For large bootstraps prefer ingest_directory_queue (worker queue).
    """
    logger.info(f"Tool 'ingest_directory' called with {directory_path}, concurrency={concurrency}")
    try:
        return await services.ingestion_tool.batch_process_directory(
            directory_path, concurrency=concurrency
        )
    except Exception as e:
        logger.error(f"Error in 'ingest_directory' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def ingest_directory_queue(directory_path: str, concurrency: int = 4) -> Dict[str, Any]:
    """
    Enqueue a directory for background ingestion (persistent worker queue).
    Returns a job_id; check progress with job_status or list_ingest_jobs.
    Recommended for bootstrapping large libraries.
    """
    logger.info(f"Tool 'ingest_directory_queue' called with {directory_path}")
    try:
        from pathlib import Path as _Path
        path = _Path(directory_path)
        if not path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory_path}"}
        return await services.ingest_queue.enqueue(
            "directory", str(path), concurrency=concurrency
        )
    except Exception as e:
        logger.error(f"Error in 'ingest_directory_queue' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def ingest_queued(
    target: str,
    job_type: str = "document",
    content: str = "",
    filename: str = "queued_text.txt",
    concurrency: int = 1,
) -> Dict[str, Any]:
    """
    Queue a single ingestion job. job_type: document (target=file path),
    directory, url, or text (pass the raw text via 'content').
    """
    try:
        extra = None
        if job_type == "text":
            if not content:
                return {"success": False, "error": "text jobs require 'content'"}
            extra = {"content": content, "filename": filename}
        return await services.ingest_queue.enqueue(
            job_type, target, concurrency=concurrency, extra=extra,
        )
    except Exception as e:
        logger.error(f"Error in 'ingest_queued' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def job_status(job_id: str) -> Dict[str, Any]:
    """Status of one background ingestion job (pending/running/done/failed)."""
    try:
        job = await services.ingest_queue.get_job(job_id)
        return {"success": job is not None, "job": job}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_ingest_jobs(limit: int = 25) -> Dict[str, Any]:
    """List recent background ingestion jobs with status and progress."""
    try:
        jobs = await services.ingest_queue.list_jobs(limit)
        stats = await services.ingest_queue.get_stats()
        return {"success": True, "jobs": jobs, "stats": stats}
    except Exception as e:
        return {"success": False, "error": str(e), "jobs": []}


@mcp.tool()
async def scan_pii(text: str) -> Dict[str, Any]:
    """
    Auto-detect sensitive data (India/US/global: Aadhaar, PAN, IFSC, GSTIN,
    SSN, ITIN, phone, email, cards, IPs) and return the masked placeholder
    view plus the findings table.
    """
    try:
        result = services.pii_service.mask_for_agent(
            text or "", namespace=f"mcp:{uuid.uuid4().hex}", register=False
        )
        return {
            "success": True,
            "masked_text": result.masked_text,
            "masked_count": result.masked_count,
            "findings": [
                {
                    "entity": f.entity, "label": f.label, "category": f.category,
                    "placeholder": f.placeholder, "start": f.start, "end": f.end,
                }
                for f in result.findings
            ],
        }
    except Exception as e:
        logger.error(f"Error in 'scan_pii' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def pii_restore_placeholders(text: str) -> Dict[str, Any]:
    """
    Replace [TYPE_N]-style PII placeholders with the real values stored
    in-memory from recent masked tool results, then return the text ready
    for the end user.

    Call this on your FINAL user-facing reply (after composing it with the
    placeholders you received). Placeholders expire after one hour and never
    touch disk. If nothing resolves, pass the text through unchanged.
    """
    try:
        import re as _re
        raw = text or ""
        restored = services.pii_service.restore_for_user(raw, _MCP_PII_NS)
        placeholders = _re.findall(r"\[(?:[A-Z_]+)_\d+\]", raw)
        # a placeholder is resolved when it no longer appears in the output
        unresolved = sum(1 for token in set(placeholders) if token in restored)
        return {
            "success": True,
            "text": restored,
            "placeholders_seen": len(set(placeholders)),
            "unresolved": unresolved,
            "changed": restored != raw,
        }
    except Exception as e:
        logger.error(f"Error in 'pii_restore_placeholders' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "text": text}


@mcp.tool()
async def pii_add_custom_rule(
    name: str, rule: str, rule_type: str = "keyword", tag: str = "KEYWORD"
) -> Dict[str, Any]:
    """
    Add a custom PII filter rule that is applied in every masking pass.
    rule_type: 'keyword' (exact term) or 'regex' (Python regex, case-insensitive).
    """
    try:
        return await services.pii_service.add_custom_rule(name, rule, rule_type, tag)
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def pii_remove_custom_rule(name: str) -> Dict[str, Any]:
    """Remove a custom PII filter rule by its name."""
    try:
        return await services.pii_service.remove_custom_rule(name)
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def pii_list_custom_rules() -> Dict[str, Any]:
    """List the saved custom PII filter rules."""
    try:
        return {"success": True, "rules": services.pii_service.list_custom_rules()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def ingest_url(url: str) -> Dict[str, Any]:
    """
    Download a file over HTTP(S) and ingest it.
    """
    logger.info(f"Tool 'ingest_url' called with {url}")
    try:
        return await services.ingestion_tool.process_url(url)
    except Exception as e:
        logger.error(f"Error in 'ingest_url' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def ingest_text(content: str, filename: str = "text_content.txt") -> Dict[str, Any]:
    """
    Ingest raw text content (e.g. pasted notes) directly.
    """
    logger.info(f"Tool 'ingest_text' called with filename: {filename}")
    try:
        return await services.ingestion_tool.process_text_content(content, filename)
    except Exception as e:
        logger.error(f"Error in 'ingest_text' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def semantic_search(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Hybrid search (dense FAISS + BM25 sparse, Reciprocal Rank Fusion) through
    the indexed knowledge base. 'filters' can narrow the search (document_id,
    doc_type).

    PII protocol: returned snippets have sensitive values rewritten to
    [TYPE_N] placeholders. Use the placeholders as-is when composing answers
    and call pii_restore_placeholders on the final user-facing reply.
    """
    logger.info(f"Tool 'semantic_search' called with query: {query}, top_k: {top_k}, filters: {filters}")
    try:
        results = await services.search_tool.search(query, top_k, filters)
        masked_results, findings = masked_source_dicts(results, services.pii_service)
        if findings:
            logger.info("MCP boundary: masked %s PII finding(s) in search output", findings)
        return {
            "success": True,
            "query": query,
            "results": masked_results,
            "total_results": len(results),
            "pii_placeholders": findings,
        }
    except Exception as e:
        logger.error(f"Error in 'semantic_search' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "results": []}


@mcp.tool()
async def summarize_content(
    content: Optional[str] = None,
    document_id: Optional[str] = None,
    style: str = "concise"
) -> Dict[str, Any]:
    """
    Generate a summary of provided content or a document_id.
    Available styles: concise, detailed, bullet_points, executive.
    """
    logger.info(f"Tool 'summarize_content' called. doc_id: {document_id}, style: {style}, has_content: {content is not None}")
    try:
        text_to_summarize = content
        if document_id and not text_to_summarize:
            doc = await document_store_service.get_document(document_id)
            if not doc:
                return {"success": False, "error": f"Document {document_id} not found"}
            text_to_summarize = await document_store_service.load_content(document_id) or ""
        if not text_to_summarize:
            return {"success": False, "error": "No content provided for summarization"}
        max_length = 10000
        if len(text_to_summarize) > max_length:
            logger.warning(f"Content for summarization is long ({len(text_to_summarize)} chars), truncating to {max_length}")
            text_to_summarize = text_to_summarize[:max_length] + "..."
        summary = await services.generative_tool.summarize(text_to_summarize, style)
        return {
            "success": True,
            "summary": summary,
            "original_length": len(text_to_summarize),
            "summary_length": len(summary),
            "style": style
        }
    except Exception as e:
        logger.error(f"Error in 'summarize_content' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def generate_tags(
    content: Optional[str] = None,
    document_id: Optional[str] = None,
    max_tags: int = 5
) -> Dict[str, Any]:
    """
    Generate relevant tags for content or a document_id.
    Saves tags to document metadata if document_id is provided.
    """
    logger.info(f"Tool 'generate_tags' called. doc_id: {document_id}, max_tags: {max_tags}, has_content: {content is not None}")
    try:
        text_for_tags = content
        if document_id and not text_for_tags:
            text_for_tags = await document_store_service.load_content(document_id) or ""
            if not text_for_tags:
                return {"success": False, "error": f"Document {document_id} not found"}
        if not text_for_tags:
            return {"success": False, "error": "No content provided for tag generation"}
        # the LLM only ever sees a masked view of the content
        masked_for_tags, _ = mask_for_mcp(text_for_tags, services.pii_service)
        tags = await services.generative_tool.generate_tags(masked_for_tags, max_tags)
        if document_id and tags:
            await document_store_service.update_document_metadata(document_id, {"tags": tags})
            logger.info(f"Tags {tags} saved for document {document_id}")
        return {
            "success": True,
            "tags": tags,
            "content_length": len(text_for_tags),
            "document_id": document_id
        }
    except Exception as e:
        logger.error(f"Error in 'generate_tags' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def answer_question(question: str, context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Answer questions using hybrid RAG over the knowledge base.
    'context_filter' can narrow the context search (document_id, doc_type).

    PII protocol (end-to-end): the question and retrieval context are masked
    to [TYPE_N] placeholders before the LLM sees them, generation happens in
    this process, and real values are restored in the returned answer. Sources
    are returned masked - use pii_restore_placeholders on any text you show
    the user if you compose your own reply.
    """
    logger.info(f"Tool 'answer_question' called with question: {question}, context_filter: {context_filter}")
    try:
        search_results = await services.search_tool.search(question, top_k=5, filters=context_filter)
        if not search_results:
            return {
                "success": False,
                "error": "No relevant context found. Please upload relevant documents.",
                "question": question,
                "answer": "I could not find enough information in the documents to answer your question."
            }
        answer = await services.generative_tool.answer_question(question, search_results)
        masked_sources, findings = masked_source_dicts(search_results, services.pii_service)
        if findings:
            logger.info("MCP boundary: masked %s PII finding(s) in source list", findings)
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": masked_sources,
            "pii_placeholders": findings,
            "confidence": "high" if len(search_results) >= 3 else "medium"
        }
    except Exception as e:
        logger.error(f"Error in 'answer_question' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def voice_qa(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ask a question using the AI voice assistant with RAG capabilities.
    Provides text-based Q&A powered by LlamaIndex agentic search.
    """
    logger.info(f"Tool 'voice_qa' called with question: {question}")
    try:
        result = await services.voice_tool.voice_qa(question, session_id)
        return result
    except Exception as e:
        logger.error(f"Error in 'voice_qa' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def generate_podcast(
    document_ids: List[str],
    style: str = "conversational",
    duration_minutes: int = 10,
    host1_voice: str = "Rachel",
    host2_voice: str = "Adam"
) -> Dict[str, Any]:
    """
    Generate a podcast from selected documents.
    Styles: conversational, educational, technical, casual.
    Recommend 5-15 minutes; voices: Rachel, Adam, Domi, Bella, ...
    """
    logger.info(f"Tool 'generate_podcast' called with {len(document_ids)} docs, style: {style}")
    try:
        result = await services.podcast_tool.generate_podcast(
            document_ids=document_ids,
            style=style,
            duration_minutes=duration_minutes,
            host1_voice=host1_voice,
            host2_voice=host2_voice
        )
        return result
    except Exception as e:
        logger.error(f"Error in 'generate_podcast' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_documents_for_ui(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """
    (UI Helper) Paginated document listing backed by SQLite indexes -
    O(limit) at 1M+ documents instead of full filesystem scans.
    """
    logger.info(f"Tool 'list_documents_for_ui' called limit: {limit}, offset: {offset}")
    try:
        documents = await document_store_service.list_documents(limit, offset)
        total = await document_store_service.count_documents()
        return {
            "success": True,
            "documents": [doc.to_dict() for doc in documents],
            "total": total,
        }
    except Exception as e:
        logger.error(f"Error in 'list_documents_for_ui' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "documents": []}


@mcp.tool()
async def library_stats() -> Dict[str, Any]:
    """
    Capacity/health view: FAISS vector stats, BM25 sparse index stats,
    SQLite document counts, embedding model info.
    """
    try:
        vector_stats = await vector_store_service.get_stats()
        doc_stats = await document_store_service.get_stats()
        bm25_stats = await services.bm25_service.get_stats()
        embed_info = services.embedding_service.get_model_info()
        return {
            "success": True,
            "vector": vector_stats,
            "documents": doc_stats,
            "bm25": bm25_stats,
            "embedding": embed_info,
            "hybrid": services.hybrid_search.stats(),
        }
    except Exception as e:
        logger.error(f"Error in 'library_stats' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _startup() -> None:
    """Boot the storage core + retrieval tiers before serving requests."""
    await services.start()


if __name__ == "__main__":
    logger.info("Starting FastMCP server...")
    asyncio.run(_startup())
    services.start_maintenance_in_thread()
    try:
        asyncio.run(mcp.run())
    finally:
        logger.info("FastMCP server loop exited.")
