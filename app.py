import gradio as gr
import os
import asyncio
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.asyncio_compat import apply as apply_asyncio_compat
from core.asyncio_compat import ensure_utf8_stdio

# Patched asyncio: nested loops for Gradio callbacks AND tolerance for the
# `loop_factory` kwarg that modern uvicorn passes to asyncio.run (uvicorn>=0.35
# + unpatched nest_asyncio crashes the server thread -> "cannot find empty
# port" on single-port hosts like HF Spaces).
apply_asyncio_compat()
ensure_utf8_stdio()

# Import our custom modules
from services.bootstrap import get_services
from core.models import SearchResult, Document
import config
from services.llamaindex_service import LlamaIndexService
from services.elevenlabs_service import ElevenLabsService
from services.podcast_generator_service import PodcastGeneratorService
from mcp_tools.voice_tool import VoiceTool
from mcp_tools.podcast_tool import PodcastTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _boot_services(services) -> None:
    """One-shot boot of storage core + retrieval tiers, loop-safe.

    Runs `services.start()` on an isolated event loop thread so this works
    both at module import (no ambient loop) and from inside a running one
    (Gradio worker threads).
    """
    if getattr(services, "_booted", False):
        return
    import threading

    done = threading.Event()

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(services.start())
            services._booted = True
        except Exception as exc:
            logger.exception("Service boot failed: %s", exc)
        finally:
            loop.close()
            done.set()

    threading.Thread(target=_runner, name="rag-boot", daemon=True).start()
    done.wait(timeout=120)


def _ensure_maintenance_thread(services) -> None:
    if not getattr(services, "_maintenance_started", False):
        services.start_maintenance_in_thread()
        services._maintenance_started = True

class ContentOrganizerMCPServer:
    def __init__(self):
        # Initialize services via the shared bootstrap (SQLite core +
        # FAISS vector tier + BM25 sparse leg + hybrid RRF pipeline)
        logger.info("Initializing Content Organizer MCP Server...")
        services = get_services()
        self.vector_store = services.vector_store
        self.document_store = services.document_store
        self.embedding_service = services.embedding_service
        self.llm_service = services.llm_service
        self.ocr_service = services.ocr_service
        self.llamaindex_service = services.llamaindex_service
        self.elevenlabs_service = services.elevenlabs_service
        self.podcast_generator = services.podcast_generator
        self.ingestion_tool = services.ingestion_tool
        self.search_tool = services.search_tool
        self.generative_tool = services.generative_tool
        self.pii_service = services.pii_service
        self.ingest_queue = services.ingest_queue
        self.voice_tool = services.voice_tool
        self.podcast_tool = services.podcast_tool

        # Track processing status
        self.processing_status = {}

        # Document cache for quick access
        self.document_cache = {}

        # Boot storage core + retrieval tiers once per process
        _boot_services(services)
        _ensure_maintenance_thread(services)
        logger.info("Content Organizer MCP Server initialized successfully!")

    def run_async(self, coro):
        """Helper to run async functions in Gradio"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # If loop is already running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    async def ingest_document_async(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """MCP Tool: Ingest and process a document"""
        try:
            task_id = str(uuid.uuid4())
            self.processing_status[task_id] = {"status": "processing", "progress": 0}
            result = await self.ingestion_tool.process_document(file_path, file_type, task_id)
            if result.get("success"):
                self.processing_status[task_id] = {"status": "completed", "progress": 100}
                doc_id = result.get("document_id")
                if doc_id:
                    doc = await self.document_store.get_document(doc_id)
                    if doc:
                        self.document_cache[doc_id] = doc
                return result
            else:
                self.processing_status[task_id] = {"status": "failed", "error": result.get("error")}
                return result
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {"success": False, "error": str(e), "message": "Failed to process document"}

    # ------------------------------------------------------- PII protection
    async def pii_scan_async(self, text: str, disabled_types_json: Optional[str] = None) -> Dict[str, Any]:
        """MCP Tool: auto-detect PII and return the masked placeholder view."""
        try:
            disabled = json.loads(disabled_types_json) if disabled_types_json else None
            result = self.pii_service.mask_for_agent(
                text or "", namespace=f"ui:{uuid.uuid4().hex}", disabled_types=disabled
            )
            rows = [f.to_table_row() for f in result.findings]
            return {
                "success": True,
                "masked_text": result.masked_text,
                "findings": rows,
                "masked_count": result.masked_count,
                "enabled": config.config.PII_MASKING_ENABLED,
            }
        except Exception as e:
            logger.error(f"PII scan failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def pii_add_rule_async(self, name: str, rule: str, rule_type: str, tag: str) -> Dict[str, Any]:
        try:
            return await self.pii_service.add_custom_rule(name, rule, rule_type, tag)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def pii_remove_rule_async(self, name: str) -> Dict[str, Any]:
        try:
            return await self.pii_service.remove_custom_rule(name)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def pii_list_rules_async(self) -> Dict[str, Any]:
        try:
            return {"success": True, "rules": self.pii_service.list_custom_rules()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --------------------------------------------------- ingestion queue
    async def enqueue_directory_async(self, directory_path: str, concurrency: int = 4) -> Dict[str, Any]:
        try:
            path = Path(directory_path)
            if not path.is_dir():
                return {"success": False, "error": f"Not a directory: {directory_path}"}
            return await self.ingest_queue.enqueue(
                "directory", str(path), concurrency=concurrency
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def job_status_async(self, job_id: str) -> Dict[str, Any]:
        try:
            job = await self.ingest_queue.get_job((job_id or "").strip())
            return {"success": job is not None, "job": job}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_jobs_async(self, limit: int = 25) -> Dict[str, Any]:
        try:
            jobs = await self.ingest_queue.list_jobs(limit)
            stats = await self.ingest_queue.get_stats()
            return {"success": True, "jobs": jobs, "stats": stats}
        except Exception as e:
            return {"success": False, "error": str(e)}


    async def get_document_content_async(self, document_id: str) -> Optional[str]:
        """Get document content by ID"""
        try:
            # Check cache first
            if document_id in self.document_cache:
                return self.document_cache[document_id].content
            
            # Get from store
            doc = await self.document_store.get_document(document_id)
            if doc:
                self.document_cache[document_id] = doc
                return doc.content
            return None
        except Exception as e:
            logger.error(f"Error getting document content: {str(e)}")
            return None

    async def semantic_search_async(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """MCP Tool: Perform semantic search"""
        try:
            results = await self.search_tool.search(query, top_k, filters)
            return {"success": True, "query": query, "results": [result.to_dict() for result in results], "total_results": len(results)}
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return {"success": False, "error": str(e), "query": query, "results": []}

    async def summarize_content_async(self, content: str = None, document_id: str = None, style: str = "concise") -> Dict[str, Any]:
        try:
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for summarization"}
            max_content_length = 4000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            summary = await self.generative_tool.summarize(content, style)
            return {"success": True, "summary": summary, "original_length": len(content), "summary_length": len(summary), "style": style, "document_id": document_id}
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_tags_async(self, content: str = None, document_id: str = None, max_tags: int = 5) -> Dict[str, Any]:
        """MCP Tool: Generate tags for content"""
        try:
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for tag generation"}
            tags = await self.generative_tool.generate_tags(content, max_tags)
            if document_id and document_id != "none" and tags:
                await self.document_store.update_document_metadata(document_id, {"tags": tags})
            return {"success": True, "tags": tags, "content_length": len(content), "document_id": document_id}
        except Exception as e:
            logger.error(f"Tag generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    async def generate_podcast_async(
        self,
        document_ids: List[str],
        style: str = "conversational",
        duration_minutes: int = 10,
        host1_voice: str = "Rachel",
        host2_voice: str = "Adam"
    ) -> Dict[str, Any]:
        """Generate podcast from documents"""
        try:
            result = await self.podcast_tool.generate_podcast(
                document_ids=document_ids,
                style=style,
                duration_minutes=duration_minutes,
                host1_voice=host1_voice,
                host2_voice=host2_voice
            )
            return result
        except Exception as e:
            logger.error(f"Podcast generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def answer_question_async(self, question: str, context_filter: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            search_results = await self.search_tool.search(question, top_k=5, filters=context_filter)
            if not search_results:
                return {"success": False, "error": "No relevant context found in your documents. Please make sure you have uploaded relevant documents.", "question": question}
            answer = await self.generative_tool.answer_question(question, search_results)
            return {"success": True, "question": question, "answer": answer, "sources": [result.to_dict() for result in search_results], "confidence": "high" if len(search_results) >= 3 else "medium"}
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return {"success": False, "error": str(e), "question": question}

    async def generate_outline_async(self, topic: str, num_sections: int = 5, detail_level: str = "medium") -> Dict[str, Any]:
        try:
            outline = await self.generative_tool.generate_outline(topic, num_sections, detail_level)
            return {"success": True, "result": outline}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def explain_concept_async(self, concept: str, audience: str = "general", length: str = "medium") -> Dict[str, Any]:
        try:
            explanation = await self.generative_tool.explain_concept(concept, audience, length)
            return {"success": True, "result": explanation}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def paraphrase_text_async(self, text: str, style: str = "formal") -> Dict[str, Any]:
        try:
            paraphrase = await self.generative_tool.paraphrase_text(text, style)
            return {"success": True, "result": paraphrase}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def categorize_content_async(self, content: str, categories: List[str]) -> Dict[str, Any]:
        try:
            category = await self.generative_tool.categorize(content, categories)
            return {"success": True, "result": category}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def extract_key_insights_async(self, content: str, num_insights: int = 5) -> Dict[str, Any]:
        try:
            insights = await self.generative_tool.extract_key_insights(content, num_insights)
            return {"success": True, "result": "\n".join([f"- {insight}" for insight in insights])}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_questions_async(self, content: str, question_type: str = "comprehension", num_questions: int = 5) -> Dict[str, Any]:
        try:
            questions = await self.generative_tool.generate_questions(content, question_type, num_questions)
            return {"success": True, "result": "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def extract_key_information_async(self, content: str) -> Dict[str, Any]:
        try:
            info = await self.llm_service.extract_key_information(content)
            return {"success": True, "result": json.dumps(info, indent=2)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_documents_sync(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        try:
            documents = self.run_async(self.document_store.list_documents(limit, offset))
            return {"success": True, "documents": [doc.to_dict() for doc in documents], "total": len(documents)}
        except Exception as e:
            return {"success": False, "error": str(e)}

mcp_server = ContentOrganizerMCPServer()

def get_document_list():
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"]:
            if result["documents"]:
                doc_list_str = "📚 Documents in Library:\n\n"
                for i, doc_item in enumerate(result["documents"], 1):
                    doc_list_str += f"{i}. {doc_item['filename']} (ID: {doc_item['id'][:8]}...)\n"
                    doc_list_str += f"   Type: {doc_item['doc_type']}, Size: {doc_item['file_size']} bytes\n"
                    if doc_item.get('tags'):
                        doc_list_str += f"   Tags: {', '.join(doc_item['tags'])}\n"
                    doc_list_str += f"   Created: {doc_item['created_at'][:10]}\n\n"
                return doc_list_str
            else:
                return "No documents in library yet. Upload some documents to get started!"
        else:
            return f"Error loading documents: {result['error']}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_document_choices():
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"] and result["documents"]:
            choices = [(f"{doc['filename']} ({doc['id'][:8]}...)", doc['id']) for doc in result["documents"]]
            logger.info(f"Generated {len(choices)} document choices")
            return choices
        return []
    except Exception as e:
        logger.error(f"Error getting document choices: {str(e)}")
        return []

def refresh_library():
    """First half of every refresh: stats list + fresh choices.

    Callers write the choices into ``doc_choices_state``; the state's
    .change handler broadcasts gr.update to every document dropdown
    (library, content, tags, delete, PII scan, podcast) so new/deleted
    documents appear everywhere with one code path.
    """
    doc_list_refreshed = get_document_list()
    doc_choices_refreshed = get_document_choices()
    logger.info(f"Refreshing library. Found {len(doc_choices_refreshed)} choices.")
    return doc_list_refreshed, doc_choices_refreshed

def upload_and_process_file(file):
    if file is None:
        doc_list_initial = get_document_list()
        doc_choices_initial = get_document_choices()
        return ("No file uploaded", "", doc_list_initial, doc_choices_initial)
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_type = Path(file_path).suffix.lower().strip('.') # Ensure suffix is clean
        logger.info(f"Processing file: {file_path}, type: {file_type}")
        result = mcp_server.run_async(mcp_server.ingest_document_async(file_path, file_type))

        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()

        if result["success"]:
            return (
                f"✅ Success: {result['message']}\nDocument ID: {result['document_id']}\nChunks created: {result['chunks_created']}",
                result["document_id"],
                doc_list_updated,
                doc_choices_updated,
            )
        else:
            return (
                f"❌ Error: {result.get('error', 'Unknown error')}", "",
                doc_list_updated,
                doc_choices_updated,
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (
            f"❌ Error: {str(e)}", "",
            doc_list_error,
            doc_choices_error,
        )

def perform_search(query, top_k):
    if not query.strip():
        return "Please enter a search query"
    try:
        result = mcp_server.run_async(mcp_server.semantic_search_async(query, int(top_k)))
        if result["success"]:
            if result["results"]:
                output_str = f"🔍 Found {result['total_results']} results for: '{query}'\n\n"
                for i, res_item in enumerate(result["results"], 1):
                    output_str += f"Result {i}:\n"
                    output_str += f"📊 Relevance Score: {res_item['score']:.3f}\n"
                    output_str += f"📄 Content: {res_item['content'][:300]}...\n"
                    if 'document_filename' in res_item.get('metadata', {}):
                        output_str += f"📁 Source: {res_item['metadata']['document_filename']}\n"
                    output_str += f"🔗 Document ID: {res_item.get('document_id', 'Unknown')}\n"
                    output_str += "-" * 80 + "\n\n"
                return output_str
            else:
                return f"No results found for: '{query}'\n\nMake sure you have uploaded relevant documents first."
        else:
            return f"❌ Search failed: {result['error']}"
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Error: {str(e)}"


# ------------------------------------------------- PII Protection tab handlers

PII_ENTITY_CHOICES = [
    "EMAIL", "PHONE", "IN_AADHAAR", "IN_AADHAAR_UNVERIFIED", "IN_PAN", "IN_GSTIN", "IN_IFSC",
    "IN_PASSPORT", "US_SSN", "US_ITIN", "US_ZIP", "US_ROUTING", "US_DL",
    "CREDIT_CARD", "IBAN", "IP_ADDRESS", "URL",
]

def pii_detect_and_mask(text, include_india, include_us, include_ner, disabled_types):
    """Auto-detect all PII types, show the masked placeholder view + findings."""
    if not text or not text.strip():
        return "", [], "Paste some text or load a library document, then click Detect & Mask."
    try:
        disabled = list(disabled_types or [])
        if not include_india:
            disabled.extend(["IN_AADHAAR", "IN_AADHAAR_UNVERIFIED", "IN_PAN", "IN_GSTIN", "IN_IFSC", "IN_PASSPORT", "IN_PHONE"])
        if not include_us:
            disabled.extend(["US_SSN", "US_ITIN", "US_PHONE", "US_ZIP", "US_ROUTING", "US_DL"])
        config.config.PII_NER_ENABLED = bool(include_ner)
        result = mcp_server.run_async(
            mcp_server.pii_scan_async(text, json.dumps(disabled))
        )
        if result["success"]:
            status = (
                f"✅ Masked {result['masked_count']} item(s). Same values always share the same "
                f"placeholder, and the agent only ever sees the masked view."
            )
            return result["masked_text"], result["findings"], status
        return text, [], f"❌ PII scan failed: {result['error']}"
    except Exception as e:
        logger.error(f"PII detect error: {str(e)}")
        return text, [], f"❌ Error: {str(e)}"

def pii_scan_document(doc_choice, include_in, include_us, include_ner, disabled_types):
    """Load a library document and auto-detect its PII (respects toggles)."""
    try:
        if not doc_choice:
            return "", [], "Select a document from your library first."
        doc_id = doc_choice
        content = mcp_server.run_async(mcp_server.get_document_content_async(doc_id)) or ""
        if not content.strip():
            return "", [], "Document has no extractable text content."
        return pii_detect_and_mask(content[:50000], include_in, include_us, include_ner, disabled_types)
    except Exception as e:
        logger.error(f"PII document scan error: {str(e)}")
        return "", [], f"❌ Error: {str(e)}"

def _rules_to_rows(rules):
    return [[r.get("name", ""), r.get("type", ""), r.get("tag", ""), r.get("rule", "")] for r in (rules or [])]

def pii_add_custom_rule(name, rule, rule_type, tag):
    if not rule or not rule.strip():
        return "Enter a keyword or regex to filter.", gr.update()
    try:
        result = mcp_server.run_async(
            mcp_server.pii_add_rule_async(name or "", rule.strip(), rule_type, tag or "KEYWORD")
        )
        if result.get("success"):
            rules_result = mcp_server.run_async(mcp_server.pii_list_rules_async())
            rows = _rules_to_rows(rules_result.get("rules"))
            return (
                f"✅ Rule added ({result['rule']['type']}: {result['rule']['rule'][:40]}). "
                f"{result['total']} rule(s) saved.",
                rows,
            )
        return f"❌ {result.get('error', 'Could not add rule')}", gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()

def pii_remove_custom_rule(name):
    if not name or not name.strip():
        return "Enter the rule name to remove.", gr.update()
    try:
        result = mcp_server.run_async(mcp_server.pii_remove_rule_async(name.strip()))
        rules_result = mcp_server.run_async(mcp_server.pii_list_rules_async())
        rows = _rules_to_rows(rules_result.get("rules"))
        if result.get("success"):
            return f"🗑️ Removed {result['removed']} rule(s).", rows
        return f"❌ No rule named '{name.strip()}' found.", rows
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()

def pii_refresh_rules():
    try:
        rules_result = mcp_server.run_async(mcp_server.pii_list_rules_async())
        return _rules_to_rows(rules_result.get("rules"))
    except Exception:
        return []

# --------------------------------------------- Ingestion Queue tab handlers

def enqueue_directory_ui(directory_path, concurrency):
    if not directory_path or not str(directory_path).strip():
        return "Enter a directory path to ingest.", gr.update(), gr.update()
    try:
        result = mcp_server.run_async(
            mcp_server.enqueue_directory_async(str(directory_path).strip(), int(concurrency or 4))
        )
        if result["success"]:
            status = f"✅ Queued job {result['job_id'][:12]}… ({result['target']}). The worker picks it up automatically."
            return status, *refresh_jobs_tables()
        return f"❌ {result.get('error')}", gr.update(), gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update()

def _jobs_rows(jobs):
    rows = []
    for job in jobs or []:
        rows.append([
            (job.get("id") or "")[:12],
            job.get("job_type", ""),
            job.get("target", "")[:60],
            job.get("status", ""),
            f"{job.get('progress_done') or 0}/{job.get('progress_total') or 0}",
            (job.get("error") or "")[:80],
            (job.get("created_at") or "")[:19],
        ])
    return rows

def refresh_jobs_tables():
    try:
        result = mcp_server.run_async(mcp_server.list_jobs_async(25))
        if not result["success"]:
            return [], f"❌ {result.get('error')}"
        rows = _jobs_rows(result["jobs"])
        stats = result.get("stats", {})
        by_status = stats.get("jobs_by_status", {})
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) or "no jobs yet"
        worker = "🟢 alive" if stats.get("worker_alive") else "🔴 stopped"
        return rows, f"Worker {worker} | jobs: {summary}"
    except Exception as e:
        return [], f"❌ Error: {str(e)}"

def refresh_jobs_ui():
    """Job table + worker status; also refreshes document dropdowns because
    finishing queue jobs add new library documents in the background."""
    rows, summary = refresh_jobs_tables()
    return rows, summary, get_document_choices()

def lookup_job_ui(job_id):
    if not job_id or not job_id.strip():
        return "Paste a job ID to check its status."
    try:
        result = mcp_server.run_async(mcp_server.job_status_async(job_id.strip()))
        job = result.get("job")
        if not job:
            return f"❌ No job with ID {job_id.strip()}"
        return (
            f"Job {job['id'][:12]}… | {job['job_type']} | {job['status']} | "
            f"progress {job.get('progress_done') or 0}/{job.get('progress_total') or 0}"
            + (f"\nError: {job['error']}" if job.get("error") else "")
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"

def update_options_visibility(task):
    """Update visibility of options based on selected task"""
    return (
        gr.update(visible=task == "Summarize"),          # summary_style
        gr.update(visible=task == "Generate Outline"),   # outline_sections
        gr.update(visible=task == "Generate Outline"),   # outline_detail
        gr.update(visible=task == "Explain Concept"),    # explain_audience
        gr.update(visible=task == "Explain Concept"),    # explain_length
        gr.update(visible=task == "Paraphrase"),         # paraphrase_style
        gr.update(visible=task == "Categorize"),         # categories_input
        gr.update(visible=task in ["Key Insights", "Generate Questions"]), # num_items
        gr.update(visible=task == "Generate Questions")  # question_type
    )

def execute_content_task(task, doc_choice, custom_text, 
                        summary_style, outline_sections, outline_detail,
                        explain_audience, explain_length,
                        paraphrase_style, categories_input,
                        num_items, question_type):
    try:
        # Get content
        content = ""
        if custom_text and custom_text.strip():
            content = custom_text
        elif doc_choice and doc_choice != "none":
            content = mcp_server.run_async(mcp_server.get_document_content_async(doc_choice))
            if not content:
                return "❌ Error: Document not found or empty"
        else:
            if task == "Generate Outline":
                content = custom_text # Topic is passed as text
            else:
                return "⚠️ Please select a document or enter text"

        # Execute task
        result = {"success": False, "error": "Unknown task"}
        
        if task == "Summarize":
            result = mcp_server.run_async(mcp_server.summarize_content_async(content=content, style=summary_style))
            if result["success"]:
                return f"📝 Summary ({summary_style}):\n\n{result['summary']}"
                
        elif task == "Generate Outline":
            # For outline, content is the topic
            result = mcp_server.run_async(mcp_server.generate_outline_async(content, int(outline_sections), outline_detail))
            if result["success"]:
                return f"📝 Outline for '{content}':\n\n{result['result']}"
                
        elif task == "Explain Concept":
            # For explain, content is the concept
            result = mcp_server.run_async(mcp_server.explain_concept_async(content, explain_audience, explain_length))
            if result["success"]:
                return f"💡 Explanation ({explain_audience}):\n\n{result['result']}"
                
        elif task == "Paraphrase":
            result = mcp_server.run_async(mcp_server.paraphrase_text_async(content, paraphrase_style))
            if result["success"]:
                return f"🔄 Paraphrased Text ({paraphrase_style}):\n\n{result['result']}"
                
        elif task == "Categorize":
            categories = [c.strip() for c in categories_input.split(',')] if categories_input else []
            result = mcp_server.run_async(mcp_server.categorize_content_async(content, categories))
            if result["success"]:
                return f"🏷️ Category:\n\n{result['result']}"
                
        elif task == "Key Insights":
            result = mcp_server.run_async(mcp_server.extract_key_insights_async(content, int(num_items)))
            if result["success"]:
                return f"🔍 Key Insights:\n\n{result['result']}"
                
        elif task == "Generate Questions":
            result = mcp_server.run_async(mcp_server.generate_questions_async(content, question_type, int(num_items)))
            if result["success"]:
                return f"❓ Generated Questions ({question_type}):\n\n{result['result']}"
                
        elif task == "Extract Key Info":
            result = mcp_server.run_async(mcp_server.extract_key_information_async(content))
            if result["success"]:
                return f"📊 Key Information:\n\n{result['result']}"

        if not result["success"]:
            return f"❌ Error: {result.get('error', 'Unknown error')}"
            
        return "✅ Task completed"

    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        return f"❌ Error: {str(e)}"

def generate_tags_for_document(doc_choice, custom_text, max_tags):
    try:
        logger.info(f"Generate tags called with doc_choice: {doc_choice}, type: {type(doc_choice)}")
        document_id = doc_choice if doc_choice and doc_choice != "none" and doc_choice != "" else None

        if custom_text and custom_text.strip():
            logger.info("Using custom text for tag generation")
            result = mcp_server.run_async(mcp_server.generate_tags_async(content=custom_text, max_tags=int(max_tags)))
        elif document_id:
            logger.info(f"Generating tags for document: {document_id}")
            result = mcp_server.run_async(mcp_server.generate_tags_async(document_id=document_id, max_tags=int(max_tags)))
        else:
            return "Please select a document from the dropdown or enter text to generate tags"
        
        if result["success"]:
            tags_str = ", ".join(result["tags"])
            output_str = f"🏷️ Generated Tags:\n\n{tags_str}\n\n"
            output_str += f"📊 Statistics:\n"
            output_str += f"- Content length: {result['content_length']} characters\n"
            output_str += f"- Number of tags: {len(result['tags'])}\n"
            if result.get('document_id'):
                output_str += f"- Document ID: {result['document_id']}\n"
                output_str += f"\n✅ Tags have been saved to the document."
            return output_str
        else:
            return f"❌ Tag generation failed: {result['error']}"
    except Exception as e:
        logger.error(f"Tag generation error: {str(e)}")
        return f"❌ Error: {str(e)}"

def ask_question(question):
    if not question.strip():
        return "Please enter a question"
    try:
        result = mcp_server.run_async(mcp_server.answer_question_async(question))
        if result["success"]:
            output_str = f"❓ Question: {result['question']}\n\n"
            output_str += f"💡 Answer:\n{result['answer']}\n\n"
            output_str += f"🎯 Confidence: {result['confidence']}\n\n"
            output_str += f"📚 Sources Used ({len(result['sources'])}):\n"
            for i, source_item in enumerate(result['sources'], 1):
                filename = source_item.get('metadata', {}).get('document_filename', 'Unknown')
                output_str += f"\n{i}. 📄 {filename}\n"
                output_str += f"   📝 Excerpt: {source_item['content'][:150]}...\n"
                output_str += f"   📊 Relevance: {source_item['score']:.3f}\n"
            return output_str
        else:
            return f"❌ {result.get('error', 'Failed to answer question')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def delete_document_from_library(document_id):
    if not document_id:
        doc_list_current = get_document_list()
        doc_choices_current = get_document_choices()
        return ("No document selected to delete.", doc_list_current, doc_choices_current)
    try:
        delete_doc_store_result = mcp_server.run_async(mcp_server.document_store.delete_document(document_id))
        delete_vec_store_result = mcp_server.run_async(mcp_server.vector_store.delete_document(document_id))

        msg = ""
        if delete_doc_store_result:
            msg += f"🗑️ Document {document_id[:8]}... deleted from document store. "
        else:
            msg += f"❌ Failed to delete document {document_id[:8]}... from document store. "

        if delete_vec_store_result:
             msg += "Embeddings deleted from vector store."
        else:
             msg += "Failed to delete embeddings from vector store (or no embeddings existed)."


        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()
        return (msg, doc_list_updated, doc_choices_updated)
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (f"❌ Error deleting document: {str(e)}", doc_list_error, doc_choices_error)

# Voice conversation state - global scope
voice_conversation_state = {
    "session_id": None,
    "active": False,
    "transcript": []
}

def start_voice_conversation():
    """Start a new voice conversation session"""
    try:
        if not mcp_server.elevenlabs_service.is_available():
            return (
                "⚠️ Voice assistant not configured. Please set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in .env",
                gr.update(interactive=False),
                gr.update(interactive=True),
                ""
            )
        
        session_id = str(uuid.uuid4())
        result = mcp_server.run_async(mcp_server.elevenlabs_service.start_conversation(session_id))
        
        if result.get("success"):
            voice_conversation_state["session_id"] = session_id
            voice_conversation_state["active"] = True
            voice_conversation_state["transcript"] = []
            
            return (
                "🎙️ Voice assistant is ready. Type your question below.",
                gr.update(interactive=False),
                gr.update(interactive=True),
                []
            )
        else:
            return (
                f"❌ Failed to start conversation: {result.get('error')}",
                gr.update(interactive=True),
                gr.update(interactive=False),
                []
            )
    except Exception as e:
        logger.error(f"Error starting voice conversation: {str(e)}")
        return (
            f"❌ Error: {str(e)}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            []
        )


def stop_voice_conversation():
    """Stop active voice conversation"""
    try:
        if not voice_conversation_state["active"]:
            return (
                "No active conversation",
                gr.update(interactive=True),
                gr.update(interactive=False),
                voice_conversation_state["transcript"]
            )
        
        session_id = voice_conversation_state["session_id"]
        if session_id:
            mcp_server.run_async(mcp_server.elevenlabs_service.end_conversation(session_id))
        
        voice_conversation_state["active"] = False
        voice_conversation_state["session_id"] = None
        
        return (
            "✅ Conversation ended",
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_conversation_state["transcript"]
        )
    except Exception as e:
        logger.error(f"Error stopping conversation: {str(e)}")
        return (
            f"❌ Error: {str(e)}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_conversation_state["transcript"]
        )


def send_voice_message(message):
    """Send a text message in voice conversation"""
    try:
        if not voice_conversation_state["active"]:
            return ("Please start a conversation first", "", format_transcript(voice_conversation_state["transcript"]))
        
        if not message or not message.strip():
            return ("Please enter a message", message, format_transcript(voice_conversation_state["transcript"]))
        
        session_id = voice_conversation_state["session_id"]
        voice_conversation_state["transcript"].append({"role": "user", "content": message})
        
        result = mcp_server.run_async(mcp_server.voice_tool.voice_qa(message, session_id))
        
        if result.get("success"):
            answer = result.get("answer", "No response")
            voice_conversation_state["transcript"].append({"role": "assistant", "content": answer})
            return ("✅ Response received", "", format_transcript(voice_conversation_state["transcript"]))
        else:
            return (f"❌ Error: {result.get('error')}", message, format_transcript(voice_conversation_state["transcript"]))
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        return (f"❌ Error: {str(e)}", message, format_transcript(voice_conversation_state["transcript"]))

def format_transcript(transcript):
    """Format conversation transcript for display"""
    if not transcript:
        return "No conversation yet. Start talking to the AI librarian!"
    
    formatted = ""
    for msg in transcript:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"👤 **You:** {content}\n\n"
        else:
            formatted += f"🤖 **AI Librarian:** {content}\n\n"
        formatted += "---\n\n"
    return formatted

def clear_voice_transcript():
    """Clear conversation transcript"""
    voice_conversation_state["transcript"] = []
    return ""

def send_voice_message_v6(message, chat_history):
    """Send message in voice conversation - Gradio 6 format"""
    try:
        if not voice_conversation_state["active"]:
            return chat_history, ""
        
        if not message or not message.strip():
            return chat_history, message
        
        session_id = voice_conversation_state["session_id"]
        
        # Add user message in Gradio 6 format
        chat_history.append({"role": "user", "content": message})
        
        # Get AI response
        result = mcp_server.run_async(mcp_server.voice_tool.voice_qa(message, session_id))
        
        if result.get("success"):
            answer = result.get("answer", "No response")
            chat_history.append({"role": "assistant", "content": answer})
        else:
            chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {result.get('error')}"
            })
        
        return chat_history, ""
    except Exception as e:
        logger.error(f"Error in voice message: {str(e)}")
        chat_history.append({
            "role": "assistant",
            "content": f"❌ Error: {str(e)}"
        })
        return chat_history, ""

def generate_podcast_ui(doc_ids, style, duration, voice1, voice2):
    """UI wrapper for podcast generation"""
    try:
        if not doc_ids or len(doc_ids) == 0:
            return ("⚠️ Please select at least one document", None, "No documents selected", "")
        
        logger.info(f"Generating podcast: {len(doc_ids)} docs, {style}, {duration}min")
        
        result = mcp_server.run_async(
            mcp_server.generate_podcast_async(
                document_ids=doc_ids,
                style=style,
                duration_minutes=int(duration),
                host1_voice=voice1,
                host2_voice=voice2
            )
        )
        
        if result.get("success"):
            audio_file = result.get("audio_file")
            transcript = result.get("transcript", "Transcript not available")
            message = result.get("message", "Podcast generated!")
            formatted_transcript = f"## Podcast Transcript\n\n{transcript}"
            
            return (
                f"✅ {message}",
                audio_file,
                formatted_transcript,
                result.get("podcast_id", "")
            )
        else:
            error = result.get("error", "Unknown error")
            return (f"❌ Error: {error}", None, "Generation failed", "")
    except Exception as e:
        logger.error(f"Podcast UI error: {str(e)}")
        return (f"❌ Error: {str(e)}", None, "An error occurred", "")

def load_dashboard_stats():
    """Load dashboard statistics for the UI"""
    try:
        # Get document list
        docs_result = mcp_server.list_documents_sync(limit=1000)
        doc_count = 0
        total_chunks = 0
        total_size = 0
        recent_data = []
        
        if docs_result.get("success"):
            documents = docs_result.get("documents", [])
            doc_count = len(documents)
            total_chunks = sum(doc.get("metadata", {}).get("chunk_count", 0) for doc in documents)
            total_size = sum(doc.get("file_size", 0) for doc in documents)
            storage_mb = round(total_size / (1024 * 1024), 2) if total_size > 0 else 0.0
            
            # Get recent 5 documents
            recent = documents[:5]
            recent_data = [
                [
                    doc.get("filename", "Unknown"), 
                    doc.get("doc_type", "unknown"), 
                    doc.get("created_at", "")[:10] if doc.get("created_at") else "N/A", 
                    f"{doc.get('file_size', 0)} bytes"
                ]
                for doc in recent
            ]
        else:
            storage_mb = 0.0
        
        # Service status indicators
        vector_stat = "✅ Online" if getattr(mcp_server, "vector_store", None) else "❌ Offline"
        llm_stat = "✅ Ready" if getattr(mcp_server, "llm_service", None) else "❌ Offline"
        voice_stat = "✅ Ready" if (getattr(mcp_server, "elevenlabs_service", None) and mcp_server.elevenlabs_service.is_available()) else "⚠️ Configure API Key"
        
        return (
            doc_count,
            total_chunks,
            storage_mb,
            recent_data,
            vector_stat,
            llm_stat,
            voice_stat,
        )
    except Exception as e:
        logger.error(f"Error loading dashboard stats: {str(e)}")
        return (0, 0, 0.0, [], "❌ Error", "❌ Error", "❌ Error")

def create_gradio_interface():
    # Create custom theme with modern aesthetics
    custom_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"],
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        block_title_text_weight="600",
        block_label_text_size="sm",
        block_label_text_weight="500",
    )
    
    with gr.Blocks(title="🧠 AI Digital Library Assistant", theme=custom_theme) as interface:
        with gr.Tabs():
            # Dashboard Tab - New Landing Page
            with gr.Tab("🏠 Dashboard"):
                gr.Markdown("# Welcome to Your AI Library Assistant")
                gr.Markdown("*Your intelligent document management and analysis platform powered by AI*")
                
                # Quick Stats Section
                gr.Markdown("## 📊 Quick Stats")
                with gr.Row():
                    total_docs = gr.Number(
                        label="📚 Total Documents",
                        value=0,
                        interactive=False,
                        container=True
                    )
                    total_chunks = gr.Number(
                        label="🧩 Vector Chunks",
                        value=0,
                        interactive=False,
                        container=True
                    )
                    storage_size = gr.Number(
                        label="💾 Storage (MB)",
                        value=0,
                        interactive=False,
                        container=True
                    )
                
                # Recent Activity Section
                gr.Markdown("## 📊 Recent Activity")
                with gr.Group():
                    recent_docs = gr.Dataframe(
                        headers=["Document", "Type", "Date", "Size"],
                        datatype=["str", "str", "str", "str"],
                        row_count=(5, "fixed"),
                        col_count=(4, "fixed"),
                        interactive=False,
                        label="Recently Added Documents"
                    )
                
                # System Status Section
                gr.Markdown("## � System Status")
                with gr.Row():
                    vector_status = gr.Textbox(
                        label="Vector Store",
                        value="✅ Online",
                        interactive=False,
                        container=True
                    )
                    llm_status = gr.Textbox(
                        label="LLM Service",
                        value="✅ Ready",
                        interactive=False,
                        container=True
                    )
                    voice_status = gr.Textbox(
                        label="Voice Service",
                        value="⚠️ Configure API Key",
                        interactive=False,
                        container=True
                    )
            
            with gr.Tab("📚 Document Library"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Your Document Collection")
                        document_list_display = gr.Textbox(label="Documents in Library", value=get_document_list(), lines=20, interactive=False)
                        refresh_btn_library = gr.Button("🔄 Refresh Library", variant="secondary")
                        delete_doc_dropdown_visible = gr.Dropdown(label="Select Document to Delete", choices=get_document_choices(), value=None, interactive=True, allow_custom_value=False)
                        delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")
                        delete_output_display = gr.Textbox(label="Delete Status", visible=True)
            
            with gr.Tab("📄 Upload Documents"):
                gr.Markdown("""
                ### 📥 Add Documents to Library
                Upload PDFs, Word documents, text files, or images. OCR will extract text from images automatically.
                """)
                
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("**Supported formats:** PDF, DOCX, TXT, Images (JPG, PNG)")
                            file_input_upload = gr.File(
                                label="Select File",
                                file_types=[".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"],
                                type="filepath",
                                file_count="single"
                            )
                            
                            upload_btn_process = gr.Button("🚀 Upload & Process", variant="primary", size="lg")

                        
                        with gr.Group():
                            upload_output_display = gr.Textbox(
                                label="Status",
                                lines=6,
                                interactive=False,
                                show_copy_button=False
                            )
                            
                            doc_id_output_display = gr.Textbox(
                                label="Document ID",
                                interactive=False,
                                visible=False
                            )


            with gr.Tab("🔍 Search Documents"):
                gr.Markdown("""
                ### 🔎 Semantic Search
                Find relevant content across your entire document library using AI-powered semantic search.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            search_query_input = gr.Textbox(
                                label="Search Query",
                                placeholder="What are you looking for?",
                                lines=2,
                                info="Use natural language to describe what you need"
                            )
                            
                            with gr.Accordion("🎛️ Search Options", open=False):
                                search_top_k_slider = gr.Slider(
                                    label="Number of Results",
                                    minimum=1, maximum=20, value=5, step=1,
                                    info="More results = broader search"
                                )
                            
                            search_btn_action = gr.Button("🔍 Search", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            search_output_display = gr.Textbox(
                                label="Results",
                                lines=20,
                                placeholder="Search results will appear here...",
                                show_copy_button=True
                            )

            
            with gr.Tab("🛡️ PII Protection"):
                gr.Markdown("""
                ### 🛡️ Sensitive-Data Protection (India · USA · Global)
                Auto-detect PII (Aadhaar, PAN, IFSC, GSTIN, phone, SSN, ITIN, email, cards), view the
                **masked placeholder** form, and define custom filter keywords. When you ask the agent
                questions, raw values are **never sent to the model** — it only sees placeholders — and
                the final answer restores the real values for you automatically.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            pii_doc_dropdown = gr.Dropdown(
                                label="Scan a Library Document (optional)",
                                choices=get_document_choices(),
                                value=None,
                                interactive=True,
                                info="Filled from your library - refreshes after uploads, queue jobs, or the library Refresh button"
                            )
                            pii_scan_doc_btn = gr.Button("📄 Scan Library Document", variant="secondary")
                            pii_input_text = gr.Textbox(
                                label="Or Paste Text Here",
                                placeholder="e.g. Contact Ravi Sharma at ravi@gmail.com, Aadhaar 2345 1234 5678, SSN 123-45-6789…",
                                lines=8
                            )
                        with gr.Accordion("🎚️ Detection Options", open=True):
                            pii_include_in = gr.Checkbox(
                                label="🇮🇳 India entities (Aadhaar, PAN, GSTIN, IFSC, Passport, Phone)",
                                value=True
                            )
                            pii_include_us = gr.Checkbox(
                                label="🇺🇸 US entities (SSN, ITIN, Phone, ZIP, Routing, DL)",
                                value=True
                            )
                            pii_include_ner = gr.Checkbox(
                                label="🧠 NER person names (requires presidio-analyzer)",
                                value=False,
                                info="Optional model-based detection of names/organizations"
                            )
                            pii_disabled_types = gr.CheckboxGroup(
                                choices=PII_ENTITY_CHOICES,
                                label="Disable specific types"
                            )
                        pii_btn_detect = gr.Button("🔍 Detect & Mask PII", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            pii_masked_text = gr.Textbox(
                                label="Masked View (what the agent receives)",
                                lines=10,
                                placeholder="Masked text with [EMAIL_1], [AADHAAR_1], … placeholders appears here.",
                                show_copy_button=True
                            )
                            pii_findings_df = gr.Dataframe(
                                headers=["Entity", "Type", "Category", "Placeholder", "Original Value"],
                                datatype=["str", "str", "str", "str", "str"],
                                interactive=False,
                                label="Detected Items"
                            )
                            pii_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("""
                ### ➕ Custom Filter Rules
                Add keywords (exact match) or regex patterns that should be treated as sensitive and masked everywhere.
                """)
                with gr.Row():
                    rule_name_input = gr.Textbox(label="Rule Name", placeholder="e.g. Project Codename")
                    rule_text_input = gr.Textbox(label="Keyword / Regex", placeholder="e.g. Falcon-9 or \\bProject\\s+[A-Z][a-z]+\\b")
                    rule_type_input = gr.Dropdown(choices=["keyword", "regex"], value="keyword", label="Rule Type")
                    rule_tag_input = gr.Textbox(label="Placeholder Tag", placeholder="e.g. PROJECT → [PROJECT_1]", value="KEYWORD")
                with gr.Row():
                    rule_add_btn = gr.Button("➕ Add Rule", variant="primary")
                    rule_remove_btn = gr.Button("🗑️ Remove Rule", variant="stop", size="sm")
                    rule_refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                pii_rules_df = gr.Dataframe(
                    headers=["Name", "Type", "Tag", "Rule"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    label="Saved Custom Rules (persisted in the library database)"
                )
                pii_rule_status = gr.Textbox(label="Rule Status", interactive=False)
                
                gr.Markdown(
                    "> ⚠️ **Disclaimer:** detection is best-effort (pattern + checksum based). "
                    "We make no claim of completeness or regulatory compliance — always review masked output "
                    "before relying on it in regulated workflows. Placeholder mappings are kept in memory only "
                    "and are never stored, logged, or written to disk."
                )

            
            with gr.Tab("⏳ Ingestion Queue"):
                gr.Markdown("""
                ### ⏳ Background Ingestion Jobs
                Bulk ingests run as **persistent worker-queue jobs** (SQLite-backed): parsing and embedding
                happen in the background so a bad PDF or heavy OCR can never stall the app, and queued
                work survives restarts.
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        queue_dir_input = gr.Textbox(
                            label="Directory Path",
                            placeholder="C:\\path\\to\\documents\\folder"
                        )
                    with gr.Column(scale=1):
                        queue_concurrency_input = gr.Slider(
                            label="Parallel Files", minimum=1, maximum=16, value=4, step=1
                        )
                        queue_enqueue_btn = gr.Button("📥 Enqueue Directory", variant="primary")
                queue_status = gr.Textbox(label="Enqueue Status", interactive=False)
                
                with gr.Row():
                    jobs_refresh_btn = gr.Button("🔄 Refresh Jobs", variant="secondary")
                    queue_stats_display = gr.Textbox(label="Queue Status", interactive=False)
                jobs_df = gr.Dataframe(
                    headers=["Job ID", "Type", "Target", "Status", "Progress", "Error", "Created"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Recent Jobs (most recent first)"
                )
                
                with gr.Accordion("🔎 Check a single job", open=False):
                    job_id_input = gr.Textbox(label="Job ID", placeholder="Paste a full job ID…")
                    job_lookup_btn = gr.Button("Check Job")
                    job_lookup_output = gr.Textbox(label="Job Detail", interactive=False)

            
            with gr.Tab("📝 Content Studio"):
                gr.Markdown("""
                ### 🎨 Create & Analyze Content
                Transform documents with AI-powered tools: summarize, outline, explain, and more.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Source Selection with Group
                        with gr.Group():
                            gr.Markdown("#### 📄 Content Source")
                            doc_dropdown_content = gr.Dropdown(
                                label="Select Document",
                                choices=get_document_choices(),
                                value=None,
                                interactive=True,
                                info="Choose a document from your library"
                            )
                            
                            gr.Markdown("**OR**")
                            
                            content_text_input = gr.Textbox(
                                label="Enter Text or Topic",
                                placeholder="Paste content or enter a topic...",
                                lines=4,
                                info="For outlines, enter a topic. For other tasks, paste text to analyze."
                            )
                        
                        # Task Configuration with Group
                        with gr.Group():
                            gr.Markdown("#### 🛠️ Task Configuration")
                            task_dropdown = gr.Dropdown(
                                label="Select Task",
                                choices=[
                                    "Summarize", "Generate Outline", "Explain Concept",
                                    "Paraphrase", "Categorize", "Key Insights",
                                    "Generate Questions", "Extract Key Info"
                                ],
                                value="Summarize",
                                interactive=True,
                                info="Choose the type of analysis to perform"
                            )
                        
                        # Dynamic Options with Accordion
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            summary_style_opt = gr.Dropdown(
                                label="Summary Style",
                                choices=["concise", "detailed", "bullet_points", "executive"],
                                value="concise",
                                visible=True,
                                info="How detailed should the summary be?"
                            )
                            
                            outline_sections_opt = gr.Slider(
                                label="Number of Sections",
                                minimum=3, maximum=10, value=5, step=1,
                                visible=False,
                                info="How many main sections?"
                            )
                            outline_detail_opt = gr.Dropdown(
                                label="Detail Level",
                                choices=["brief", "medium", "detailed"],
                                value="medium",
                                visible=False
                            )
                            
                            explain_audience_opt = gr.Dropdown(
                                label="Target Audience",
                                choices=["general", "technical", "beginner", "expert"],
                                value="general",
                                visible=False,
                                info="Who is this explanation for?"
                            )
                            explain_length_opt = gr.Dropdown(
                                label="Length",
                                choices=["brief", "medium", "detailed"],
                                value="medium",
                                visible=False
                            )
                            
                            paraphrase_style_opt = gr.Dropdown(
                                label="Style",
                                choices=["formal", "casual", "academic", "simple", "technical"],
                                value="formal",
                                visible=False,
                                info="Writing style for paraphrasing"
                            )
                            
                            categories_input_opt = gr.Textbox(
                                label="Categories (comma separated)",
                                placeholder="Technology, Business, Science...",
                                visible=False
                            )
                            
                            num_items_opt = gr.Slider(
                                label="Number of Items",
                                minimum=1, maximum=10, value=5, step=1,
                                visible=False
                            )
                            question_type_opt = gr.Dropdown(
                                label="Question Type",
                                choices=["comprehension", "analysis", "application", "creative", "factual"],
                                value="comprehension",
                                visible=False
                            )
                        
                        run_task_btn = gr.Button("🚀 Run Task", variant="primary", size="lg")
                    
                    with gr.Column(scale=3):
                        # Results with copy button and Group
                        with gr.Group():
                            gr.Markdown("#### 📊 Result")
                            content_output_display = gr.Textbox(
                                label="",
                                lines=25,
                                placeholder="Results will appear here...",
                                show_copy_button=True,
                                container=False
                            )

                # Event Handlers
                task_dropdown.change(
                    fn=update_options_visibility,
                    inputs=[task_dropdown],
                    outputs=[
                        summary_style_opt, outline_sections_opt, outline_detail_opt,
                        explain_audience_opt, explain_length_opt, paraphrase_style_opt,
                        categories_input_opt, num_items_opt, question_type_opt
                    ]
                )
                
                run_task_btn.click(
                    fn=execute_content_task,
                    inputs=[
                        task_dropdown, doc_dropdown_content, content_text_input,
                        summary_style_opt, outline_sections_opt, outline_detail_opt,
                        explain_audience_opt, explain_length_opt, paraphrase_style_opt,
                        categories_input_opt, num_items_opt, question_type_opt
                    ],
                    outputs=[content_output_display]
                )

            with gr.Tab("🏷️ Generate Tags"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Document Tags")
                        doc_dropdown_tag_visible = gr.Dropdown(label="Select Document to Tag", choices=get_document_choices(), value=None, interactive=True, allow_custom_value=False)
                        tag_text_input = gr.Textbox(label="Or Paste Text to Generate Tags", placeholder="Paste any text here to generate tags...", lines=8)
                        max_tags_slider = gr.Slider(label="Number of Tags", minimum=3, maximum=15, value=5, step=1)
                        tag_btn_action = gr.Button("🏷️ Generate Tags", variant="primary", size="lg")
                    with gr.Column():
                        tag_output_display = gr.Textbox(label="Generated Tags", lines=10, placeholder="Tags will appear here...")

            with gr.Tab("🎙️ Voice Assistant"):
                gr.Markdown("""
                ### 🗣️ Talk to Your AI Librarian
                
                Have a natural conversation about your documents. Ask questions, request summaries,
                or explore your content library through voice-powered interaction.
                
                **Note:** Requires ElevenLabs API configuration.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Status and Controls
                        with gr.Group():
                            voice_status_display = gr.Textbox(
                                label="Status",
                                value="Ready to start",
                                interactive=False,
                                lines=2
                            )
                            
                            with gr.Row():
                                start_voice_btn = gr.Button("🎤 Start Conversation", variant="primary", size="lg")
                                stop_voice_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", interactive=False)
                        
                        # Message Input
                        with gr.Group():
                            gr.Markdown("#### 💬 Send Message")
                            voice_input_text = gr.Textbox(
                                label="",
                                placeholder="Type your question...",
                                lines=3,
                                container=False,
                                info="Press Enter or click Send"
                            )
                            send_voice_btn = gr.Button("📤 Send", variant="secondary")
                    
                    with gr.Column(scale=3):
                        # Chat Interface with Gradio 6 Chatbot
                        with gr.Group():
                            voice_chatbot = gr.Chatbot(
                                label="Conversation",
                                type="messages",
                                height=500,
                                show_copy_button=True
                            )
                            
                            clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                
                # Voice Assistant event handlers
                start_voice_btn.click(
                    fn=start_voice_conversation,
                    outputs=[voice_status_display, start_voice_btn, stop_voice_btn, voice_chatbot]
                )
                
                stop_voice_btn.click(
                    fn=stop_voice_conversation,
                    outputs=[voice_status_display, start_voice_btn, stop_voice_btn, voice_chatbot]
                )
                
                send_voice_btn.click(
                    fn=send_voice_message_v6,
                    inputs=[voice_input_text, voice_chatbot],
                    outputs=[voice_chatbot, voice_input_text]
                )
                
                voice_input_text.submit(
                    fn=send_voice_message_v6,
                    inputs=[voice_input_text, voice_chatbot],
                    outputs=[voice_chatbot, voice_input_text]
                )
                
                clear_chat_btn.click(
                    fn=lambda: [],
                    outputs=[voice_chatbot]
                )

            with gr.Tab("🎧 Podcast Studio"):
                gr.Markdown("""
                ### 🎙️ AI-Powered Podcast Generation
                
                Transform your documents into engaging audio conversations. Select documents,
                customize the style and voices, and let AI create a professional podcast.
                
                **Powered by:** ElevenLabs AI Voice Technology
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Configuration Panel
                        with gr.Group():
                            gr.Markdown("#### 📚 Select Content")
                            
                            podcast_doc_selector = gr.CheckboxGroup(
                                choices=get_document_choices(),
                                label="Documents to Include",
                                info="Choose 1-5 documents for best results",
                                interactive=True
                            )
                        
                        with gr.Accordion("🎨 Podcast Settings", open=True):
                            with gr.Row():
                                podcast_style = gr.Dropdown(
                                    label="Style",
                                    choices=["conversational", "educational", "technical", "casual"],
                                    value="conversational",
                                    info="Sets the tone and format"
                                )
                                
                                podcast_duration = gr.Slider(
                                    label="Duration (minutes)",
                                    minimum=5,
                                    maximum=30,
                                    value=10,
                                    step=5,
                                    info="Approximate length"
                                )
                            
                            gr.Markdown("#### 🗣️ Voice Selection")
                            with gr.Row():
                                host1_voice_selector = gr.Dropdown(
                                    label="Host 1",
                                    choices=["Rachel", "Adam", "Domi", "Bella", "Antoni", "Elli", "Josh"],
                                    value="Rachel"
                                )
                                host2_voice_selector = gr.Dropdown(
                                    label="Host 2",
                                    choices=["Adam", "Rachel", "Josh", "Sam", "Emily", "Antoni", "Arnold"],
                                    value="Adam"
                                )
                        
                        generate_podcast_btn = gr.Button(
                            "🎙️ Generate Podcast",
                            variant="primary",
                            size="lg"
                        )
                        
                        podcast_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2
                        )
                        
                        podcast_id_display = gr.Textbox(
                            label="Podcast ID",
                            interactive=False,
                            visible=False
                        )
                    
                    with gr.Column(scale=3):
                        # Output Panel
                        with gr.Group():
                            gr.Markdown("#### 🎵 Generated Podcast")
                            
                            podcast_audio_player = gr.Audio(
                                label="",
                                type="filepath",
                                interactive=False,
                                autoplay=True,
                                container=False
                            )
                        
                        with gr.Accordion("📝 Transcript", open=False):
                            podcast_transcript_display = gr.Markdown(
                                value="*Transcript will appear after generation...*"
                            )
                
                # Event handlers
                generate_podcast_btn.click(
                    fn=generate_podcast_ui,
                    inputs=[
                        podcast_doc_selector,
                        podcast_style,
                        podcast_duration,
                        host1_voice_selector,
                        host2_voice_selector
                    ],
                    outputs=[
                        podcast_status,
                        podcast_audio_player,
                        podcast_transcript_display,
                        podcast_id_display
                    ]
                )
                
            with gr.Tab("❓ Ask Questions"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""### Ask Questions About Your Documents
                        The AI will search through all your uploaded documents to find relevant information 
                        and provide comprehensive answers with sources.""")
                        qa_question_input = gr.Textbox(label="Your Question", placeholder="Ask anything about your documents...", lines=3)
                        qa_btn_action = gr.Button("❓ Get Answer", variant="primary", size="lg")
                    with gr.Column():
                        qa_output_display = gr.Textbox(label="AI Answer", lines=20, placeholder="Answer will appear here with sources...")

        all_dropdowns_to_update = [delete_doc_dropdown_visible, doc_dropdown_content, doc_dropdown_tag_visible, pii_doc_dropdown, podcast_doc_selector]

        # New choices flow through a shared State: refresh/upload/delete/queue
        # handlers return plain choice lists, and a single .change handler
        # broadcasts gr.update to every document dropdown above.
        doc_choices_state = gr.State([])

        def _broadcast_document_choices(choices):
            return [gr.update(choices=choices) for _ in all_dropdowns_to_update]

        doc_choices_state.change(
            _broadcast_document_choices,
            inputs=[doc_choices_state],
            outputs=all_dropdowns_to_update,
        )

        refresh_outputs = [document_list_display, doc_choices_state]
        refresh_btn_library.click(fn=refresh_library, outputs=refresh_outputs)

        upload_outputs = [upload_output_display, doc_id_output_display, document_list_display, doc_choices_state]
        upload_btn_process.click(upload_and_process_file, inputs=[file_input_upload], outputs=upload_outputs)

        delete_outputs = [delete_output_display, document_list_display, doc_choices_state]
        delete_btn.click(delete_document_from_library, inputs=[delete_doc_dropdown_visible], outputs=delete_outputs)
        
        search_btn_action.click(perform_search, inputs=[search_query_input, search_top_k_slider], outputs=[search_output_display])
        tag_btn_action.click(generate_tags_for_document, inputs=[doc_dropdown_tag_visible, tag_text_input, max_tags_slider], outputs=[tag_output_display])
        qa_btn_action.click(ask_question, inputs=[qa_question_input], outputs=[qa_output_display])

        # ---- PII Protection tab ----
        pii_btn_detect.click(
            pii_detect_and_mask,
            inputs=[pii_input_text, pii_include_in, pii_include_us, pii_include_ner, pii_disabled_types],
            outputs=[pii_masked_text, pii_findings_df, pii_status]
        )
        pii_scan_doc_btn.click(
            pii_scan_document,
            inputs=[pii_doc_dropdown, pii_include_in, pii_include_us, pii_include_ner, pii_disabled_types],
            outputs=[pii_masked_text, pii_findings_df, pii_status]
        )
        rule_add_btn.click(
            pii_add_custom_rule,
            inputs=[rule_name_input, rule_text_input, rule_type_input, rule_tag_input],
            outputs=[pii_rule_status, pii_rules_df]
        )
        rule_remove_btn.click(
            pii_remove_custom_rule,
            inputs=[rule_name_input],
            outputs=[pii_rule_status, pii_rules_df]
        )
        rule_refresh_btn.click(fn=pii_refresh_rules, outputs=[pii_rules_df])

        # ---- Ingestion Queue tab ----
        queue_enqueue_btn.click(
            enqueue_directory_ui,
            inputs=[queue_dir_input, queue_concurrency_input],
            outputs=[queue_status, jobs_df, queue_stats_display]
        )
        jobs_refresh_btn.click(fn=refresh_jobs_ui, outputs=[jobs_df, queue_stats_display, doc_choices_state])
        job_lookup_btn.click(lookup_job_ui, inputs=[job_id_input], outputs=[job_lookup_output])

        # Load dashboard stats on interface load
        interface.load(
            fn=load_dashboard_stats,
            outputs=[total_docs, total_chunks, storage_size, recent_docs, vector_status, llm_status, voice_status]
        )
        
        interface.load(fn=refresh_library, outputs=[document_list_display, doc_choices_state])
        interface.load(fn=pii_refresh_rules, outputs=[pii_rules_df])
        interface.load(fn=refresh_jobs_ui, outputs=[jobs_df, queue_stats_display, doc_choices_state])
        return interface

if __name__ == "__main__":
    gradio_interface = create_gradio_interface()
    gradio_interface.launch(mcp_server=True, ssr_mode=False)
