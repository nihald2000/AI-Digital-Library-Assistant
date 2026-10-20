import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes", "on")


def _str(s: str) -> str:
    return s.strip()


class Config:
    # API Keys
    NEBIUS_API_KEY: Optional[str] = os.getenv("NEBIUS_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN"))
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # NEBIUS Configuration (OpenAI OSS models)
    NEBIUS_BASE_URL: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
    NEBIUS_MODEL: str = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    # Free-tier LLM provider chain (all OpenAI-compatible REST endpoints).
    # First leg with a usable key serves requests; a leg that fails with a
    # permanent error (quota/auth) is skipped automatically for the session.
    # Default order per user choice: Gemini first, Groq backup, OpenRouter
    # for model comparison, then the paid/legacy legs (Nebius, OpenAI).
    LLM_PROVIDER_CHAIN: str = os.getenv("LLM_PROVIDER_CHAIN", "gemini,groq,openrouter,nebius,openai,mistral")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")          # free tier
    GEMINI_FAST_MODEL: str = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    # Groq free tier after 2026-08-16: Llama chat models are enterprise-only;
    # the free-tier chat models are the GPT-OSS/Qwen open-weight set.
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_FAST_MODEL: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # ":free" suffix routes to OpenRouter's rotating free-model pool.
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    OPENROUTER_FAST_MODEL: str = os.getenv("OPENROUTER_FAST_MODEL", "openai/gpt-oss-20b:free")

    # Model Configuration
    # Using OpenAI managed embeddings for performance/quality
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # Truncate embeddings to this many dimensions (OpenAI Matryoshka support).
    # 512 dims = 3x less RAM/storage for FAISS and SQLite BLOBs with near-identical recall.
    # 0 = use native model dimension.
    EMBEDDING_DIMENSIONS: int = _int("EMBEDDING_DIMENSIONS", 512)
    # Local (free, open-weight) embedding model used when the configured model
    # is not an API model or no API key is available. The 2026 default is
    # microsoft/harrier-oss-v1-270m: MIT license, 94 languages, 640 dims,
    # MTEB Multilingual v2 ~66.5 - chosen for CPU Spaces measured ~4x faster
    # than the 0.6b sibling (~1.7 vs ~0.4 chunks/s, and 640 vs 512-dim storage).
    EMBEDDING_LOCAL_MODEL: str = os.getenv("EMBEDDING_LOCAL_MODEL", "microsoft/harrier-oss-v1-270m")
    # Prompt name applied to query-side embeddings for prompt-trained models.
    # harrier requires "web_search_query" for queries (documents are encoded plain).
    # Qwen3 uses "query". Set empty to disable.
    EMBEDDING_QUERY_PROMPT: str = os.getenv("EMBEDDING_QUERY_PROMPT", "web_search_query")

    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-2407")
    # gpt-5.1-chat-latest was deprecated on OpenAI (observed 2026-10-20);
    # these remain valid served models, override via env if needed.
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    FAST_MODEL: str = os.getenv("FAST_MODEL", "gpt-4.1-nano")

    # Storage Configuration
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    DOCUMENT_STORE_PATH: str = os.getenv("DOCUMENT_STORE_PATH", "./data/documents")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "content_index")
    # Single SQLite database (WAL mode) backing documents, chunks and vectors.
    DB_PATH: str = os.getenv("DB_PATH", "./data/library.db")

    # Processing Configuration
    CHUNK_SIZE: int = _int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 50)
    MAX_CONCURRENT_REQUESTS: int = _int("MAX_CONCURRENT_REQUESTS", 5)

    # ---- Embedding service scaling knobs ----
    # Chunk-level batch size (number of texts per embedding API request).
    EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 256)
    # How many embedding requests may be in flight at once.
    EMBEDDING_CONCURRENCY: int = _int("EMBEDDING_CONCURRENCY", 8)
    # Batched API clients need retry handling for 429 / 5xx.
    EMBEDDING_MAX_RETRIES: int = _int("EMBEDDING_MAX_RETRIES", 5)
    EMBEDDING_TIMEOUT_SECONDS: float = _float("EMBEDDING_TIMEOUT_SECONDS", 120.0)
    # Reused texts skip API calls (normalized sha256 -> embedding via memory-cache).
    EMBEDDING_CACHE_ENABLED: bool = _bool("EMBEDDING_CACHE_ENABLED", True)
    EMBEDDING_CACHE_SIZE: int = _int("EMBEDDING_CACHE_SIZE", 50_000)

    # ---- Vector store scaling knobs ----
    # Vector index mode: "auto" (tiering: flat -> IVF/SQ8 compaction),
    # "flat" (exact brute-force, small corpora), "ivf" (clustered + quantized),
    # "hnsw" (graph, fast + memory hungry).
    VECTOR_INDEX_MODE: str = os.getenv("VECTOR_INDEX_MODE", "auto").lower().strip()
    # Total vectors at which "auto" compacts the flat index into an IVF index.
    IVF_SWITCH_THRESHOLD: int = _int("IVF_SWITCH_THRESHOLD", 250_000)
    # 0 = auto (nlist = clamp(4 * sqrt(N), 1024, 16384) per FAISS wiki guidance).
    IVF_NLIST: int = _int("IVF_NLIST", 0)
    # Number of clusters probed per query. Higher = better recall, slower.
    IVF_NPROBE: int = _int("IVF_NPROBE", 32)
    # Persist the FAISS index to disk after this many new vectors.
    FAISS_PERSIST_BATCH: int = _int("FAISS_PERSIST_BATCH", 25_000)
    # Hard cap on candidate overfetch during filtered ANN search.
    SEARCH_OVERFETCH: int = _int("SEARCH_OVERFETCH", 8)
    SEARCH_MAX_OVERFETCH: int = _int("SEARCH_MAX_OVERFETCH", 4096)
    # Rebuild (compact) the index after this many tombstoned vectors.
    COMPACTION_TOMBSTONE_THRESHOLD: int = _int("COMPACTION_TOMBSTONE_THRESHOLD", 50_000)

    # ---- Hybrid retrieval knobs ----
    BM25_ENABLED: bool = _bool("BM25_ENABLED", True)
    BM25_MAX_CHUNKS: int = _int("BM25_MAX_CHUNKS", 5_000_000)
    # Chunks buffered before the sparse index is rebuilt from SQLite.
    BM25_REBUILD_THRESHOLD: int = _int("BM25_REBUILD_THRESHOLD", 20_000)
    RRF_K: int = _int("RRF_K", 60)
    RERANK_ENABLED: bool = _bool("RERANK_ENABLED", False)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_CANDIDATE_POOL: int = _int("RERANK_CANDIDATE_POOL", 50)
    QUERY_CACHE_SIZE: int = _int("QUERY_CACHE_SIZE", 2048)

    # Search Configuration
    DEFAULT_TOP_K: int = _int("DEFAULT_TOP_K", 5)
    SIMILARITY_THRESHOLD: float = _float("SIMILARITY_THRESHOLD", 0.3)

    # OCR Configuration
    TESSERACT_PATH: Optional[str] = os.getenv("TESSERACT_PATH")
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_AGENT_ID: Optional[str] = os.getenv("ELEVENLABS_AGENT_ID")
    ELEVENLABS_VOICE_MODEL: str = os.getenv("ELEVENLABS_VOICE_MODEL", "Rachel")
    # Cap for RAG lookups inside ElevenLabs conversational tools (was a missing
    # attribute before - the voice RAG tool crashed with AttributeError).
    CONVERSATION_TIMEOUT: float = _float("CONVERSATION_TIMEOUT", 30.0)

    # ---- LlamaIndex (agentic RAG) guardrails ----
    # The agentic service builds a second in-RAM index. It must stay capped
    # (or disabled) at large corpus sizes.
    LLAMAINDEX_ENABLED: bool = _bool("LLAMAINDEX_ENABLED", True)
    LLAMAINDEX_MAX_DOCS: int = _int("LLAMAINDEX_MAX_DOCS", 50_000)

    # Ingestion pipeline knobs
    INGEST_DEFAULT_CONCURRENCY: int = _int("INGEST_DEFAULT_CONCURRENCY", 4)
    # Queue worker: directory/bulk ingests run as persisted SQLite jobs so a
    # parsing/OCR crash can never stall the API or lose work across restarts.
    INGEST_QUEUE_ENABLED: bool = _bool("INGEST_QUEUE_ENABLED", True)
    INGEST_QUEUE_POLL_SECONDS: float = _float("INGEST_QUEUE_POLL_SECONDS", 1.0)
    INGEST_QUEUE_MAX_CONCURRENT_JOBS: int = _int("INGEST_QUEUE_MAX_CONCURRENT_JOBS", 1)

    # ---- Persistence across Space restarts ----
    # HF Space containers are wiped on every restart/upgrade. Survive it in one
    # of two ways: (a) attach Persistent Storage (auto-detected at /data), or
    # (b) free fallback: snapshots of the whole data tree pushed to a private
    # HF dataset repo (restore on boot, debounced flush after ingests).
    # (b) needs two Space secrets: HF_TOKEN (write access) and
    # PERSIST_BACKUP_REPO=<owner>/<repo>.
    PERSIST_BACKUP_ENABLED: bool = _bool("PERSIST_BACKUP_ENABLED", True)
    PERSIST_BACKUP_REPO: Optional[str] = os.getenv("PERSIST_BACKUP_REPO") or None
    # Minimum seconds between snapshot uploads (debounce + repo rate limits).
    PERSIST_BACKUP_INTERVAL_SECONDS: float = _float("PERSIST_BACKUP_INTERVAL_SECONDS", 120.0)
    # Files larger than this (e.g. podcast audio blobs) are skipped in snapshots.
    PERSIST_BACKUP_MAX_FILE_MB: int = _int("PERSIST_BACKUP_MAX_FILE_MB", 200)

    # ---- PII protection knobs ----
    PII_MASKING_ENABLED: bool = _bool("PII_MASKING_ENABLED", True)
    PII_IN_ENABLED: bool = _bool("PII_IN_ENABLED", True)    # India entities
    PII_US_ENABLED: bool = _bool("PII_US_ENABLED", True)    # USA entities
    # Optional NER (presidio-analyzer) for person names / orgs - regex alone
    # cannot reliably detect names. Default off until the dep is installed.
    PII_NER_ENABLED: bool = _bool("PII_NER_ENABLED", False)
    PII_NER_MIN_SCORE: float = _float("PII_NER_MIN_SCORE", 0.5)
    # Reversible placeholder mappings live in memory only (never on disk).
    PII_MAPPING_TTL_SECONDS: int = _int("PII_MAPPING_TTL_SECONDS", 3600)
    PII_MAX_SESSIONS: int = _int("PII_MAX_SESSIONS", 500)
    # Mask obvious PII (emails/phones/Aadhaar-like ids) in log output too.
    PII_LOG_MASKING: bool = _bool("PII_LOG_MASKING", True)

    # App Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _int("PORT", 7860)
    DEBUG: bool = _bool("DEBUG", False)

def _maybe_redirect_to_persistent_disk(cfg: "Config") -> None:
    """HF Spaces Persistent Storage (~$5/mo add-on) mounts a blank volume at /data.

    When it exists we keep every artifact there instead of the evanescent
    container filesystem - the library then survives restarts with zero code
    changes. Explicit env overrides always win.
    """
    if os.getenv("DATA_DIR"):
        return
    try:
        from pathlib import Path

        mount = Path("/data")
        if mount.is_dir() and os.access(mount, os.W_OK):
            cfg.DATA_DIR = "/data"
            os.environ.setdefault("DATA_DIR", "/data")
            if not os.getenv("DOCUMENT_STORE_PATH"):
                cfg.DOCUMENT_STORE_PATH = "/data/documents"
            if not os.getenv("DB_PATH"):
                cfg.DB_PATH = "/data/library.db"
            logger_print = print  # config.py loads before logging is up
            logger_print("Persistent-storage mount /data detected - library state now survives restarts")
    except Exception:
        pass


config = Config()
_maybe_redirect_to_persistent_disk(config)
