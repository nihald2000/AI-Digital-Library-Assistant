# Scaling the RAG Pipeline to 1 Million+ Documents

This document describes the scaled architecture, the research basis for each
choice, memory/capacity math, and the runtime knobs.

The fast path is:

```
ingest -------------------------------------------------------.
  parse (executor) -> chunk -> batch embed -> SQLite WAL rows -> FAISS index
                                              |
                                              `--> BM25 sparse bookkeeping

search ------------------------------------------------------------.
  embed query (LRU cache) -> dense FAISS leg (overfetch + escalation)
                           `-> sparse BM25 leg (bm25s over SQLite chunks)
                                  |            |
                                  `-- RRF fuse +'-> hydrate = one batched SQL read
                                      rerank (opt) `-> enrich docs (batched SQL)
```

## What replaced what (and why)

| Legacy | Scaled | Reason |
|--------|--------|--------|
| `IndexFlatIP` brute force | Auto tiering: exact `IndexFlatIP` ≤ 250k vecs → `IndexIVF_SQ8` beyond | Flat index is ~6 GB RAM per 1M vectors and O(N) per query; SQ8 gives 4:1 compression so 10M vectors ≈ 15.4 GB and search stays fast with `nprobe` tuning. FAISS wiki recommends IVF for the 1M-10M vector band ([faiss wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)). |
| Chunk metadata JSON (one giant file, rewritten per add) | SQLite `chunks` table (WAL) + per-row indexes | SQLite WAL handles millions of rows with concurrent reads; per-add write amplification goes from "rewrite everything" to "append a batch". |
| One JSON/txt file per document, `glob` on every list | SQLite `documents` + `document_content` tables with indexed expression on `created_at` | Survives 1M-file directories; pagination is O(limit) through the index instead of stat-ing every file. |
| Full-text scan for keyword search | `bm25s` sparse index over chunks | Hybrid dense+sparse retrieval via Reciprocal Rank Fusion; +7.4% NDCG over either leg alone on benchmarks ([Turnbull 2025](https://denser.ai/blog/hybrid-search-for-rag)) and exact-token rescue (product codes, error IDs) that pure embeddings miss. |
| Batched cross-encoder skipped | Optional `cross-encoder` rerank on fused top-N | +12-17pp MRR/Recall on production RAG benchmarks ([Strich et al. 2026](https://arxiv.org/html/2604.01733v1)). |
| 32-item embedding batches, sync client, no retries | Async fan-out: `EMBEDDING_BATCH_SIZE` (256) × `EMBEDDING_CONCURRENCY` (8) with exponential backoff | Makes 1M-doc ingestion practical; rate-limit aware. |
| `delete_document`: rebuild whole index vector-by-vector | Tombstone via SQLite; background compaction | Index stays online; rebuilds happen on a scheduled checkpoint, not per delete. |
| LlamaIndex sync re-builds a second index on startup | LlamaIndex only syncs when explicitly triggered, capped by `LLAMAINDEX_MAX_DOCS` | Agentic RAG is a memory hog; cap it or disable for huge corpora. |
| `content.find()` per chunk in chunker (O(N²)) | Monotonic cursor scan (≈ O(N)) | Fixes the quadratic cost on large single documents. |

## Storage layout

Everything lives in one SQLite database (`DB_PATH`, default `./data/library.db`):

| Table | Purpose |
|-------|---------|
| `documents` | Metadata with indexes on `created_at / doc_type / category` |
| `document_content(id, content)` | Full text, loaded lazily on demand |
| `chunks` | One row per chunk (vector metadata + text) with a stable `vector_id`, plus a `deleted` tombstone column |
| `chunk_vectors(vector_id, embedding BLOB)` | Raw embeddings - the *source of truth* for the FAISS index |
| `meta(key, value)` | Schema version + service bookkeeping |

Because embeddings live in SQLite, the FAISS index (`.faiss` file under
`DATA_DIR/faiss/`) is a rebuildable cache - crash recovery is simply "re-run
compaction", no embedding re-spend.

## Free embedding (offline / no API key)

When `EMBEDDING_MODEL` is not an OpenAI `text-embedding-*` model, or no API key
is configured, the vector leg runs a **free open-weight local model**. The 2026
default is `microsoft/harrier-oss-v1-270m`: MIT licensed, 94 languages, 640
dims, ~1.7 chunks/s on desktop CPU (measured; ~4x faster than the 0.6b at
~0.4 chunks/s) with near-class-leading MTEB Multilingual v2 quality. Queries
are automatically encoded with the model's `web_search_query` instruction;
documents are encoded plain.

To switch the whole pipeline to it, set:

```
EMBEDDING_MODEL=microsoft/harrier-oss-v1-270m
```

(Leave `EMBEDDING_DIMENSIONS` unset — truncation applies to OpenAI Matryoshka
models only.) Prefer maximum retrieval quality on faster hosts? Set
`EMBEDDING_LOCAL_MODEL=microsoft/harrier-oss-v1-0.6b` (1024 dims, 32K context,
top MTEB-Multilingual for sub-1B). Other options: `Qwen/Qwen3-Embedding-0.6B`
(Apache-2.0, set `EMBEDDING_QUERY_PROMPT=query`), `google/embeddinggemma-300m`,
`BAAI/bge-m3`.

> **Switching models changes embedding geometry.** Re-embed an existing library
> once after any model change: `python scripts/reembed_library.py` (backs up
> `library.db`, re-encodes all chunks, rebuilds + persists the FAISS index).
> Mixing vectors from different models in one index silently degrades recall.

## Capacity math

Take the default setup: `text-embedding-3-small` with `EMBEDDING_DIMENSIONS=512`.

| Vectors | Flat tier RAM | IVF/SQ8 tier RAM | Sparse leg RAM |
|---|---|---|---|
| 100k | ~0.2 GB | ~50 MB | <0.2 GB |
| 1M | ~2 GB | ~510 MB | ~0.5 GB |
| 10M | ~20 GB | ~5.1 GB | ~5 GB |

Notes:

- **Flat tier** ≈ `d × 4 bytes/vector` (512 × 4 = 2 KB) plus FAISS graph overhead for IVF.
- **SQ8** quantizes each component to one byte → 4x reduction. Uses `nprobe` tuning for recall.
- **1M documents ≈ 20M chunks** at ~50 chunks/doc (reasonable approximation); with SQ8 this fits in ~10 GB.
- If `BM25_MAX_CHUNKS` is exceeded, the sparse leg is automatically disabled and it's dense-only.
- Increase FAISS recall with `IVF_NPROBE` (higher captures more clusters, slower).
- `FaissWriteBatch` (FAISS_PERSIST_BATCH) checks how often a checkpoint flushes to `.faiss` on disk.

## Runtime knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `EMBEDDING_BATCH_SIZE` | 256 | Texts per API request |
| `EMBEDDING_CONCURRENCY` | 8 | Parallel in-flight embedding batches |
| `EMBEDDING_MAX_RETRIES` | 5 | Retry count for 429 / 5xx with exponential backoff (cap 30s) |
| `EMBEDDING_DIMENSIONS` | 512 | Matryoshka truncation (OpenAI 3rd gen only); 0 = model default |
| `EMBEDDING_LOCAL_MODEL` | `microsoft/harrier-oss-v1-0.6b` | Free local model used when no API model runs (see note below) |
| `EMBEDDING_QUERY_PROMPT` | `web_search_query` | Query-side instruction for prompt-trained models (`query` for Qwen3); empty disables |
| `EMBEDDING_CACHE_ENABLED` | true | LRU reuse of identical text embeddings |
| `EMBEDDING_CACHE_SIZE` | 50000 | Max cached entries |
| `VECTOR_INDEX_MODE` | auto | `auto` / `flat` / `ivf` / `hnsw` |
| `IVF_SWITCH_THRESHOLD` | 250000 | Vectors before compaction from flat to IVF |
| `IVF_NLIST` | 0 (=auto) | IVF cluster count; 0 → `clamp(4*sqrt(N), 1024, 16384)` |
| `IVF_NPROBE` | 32 | Clusters probed per query |
| `FAISS_PERSIST_BATCH` | 25000 | Vectors between FAISS disk snapshots |
| `SEARCH_OVERFETCH` | 8 | Multiplier for filtered ANN search (top_k × 8, escalating) |
| `SEARCH_MAX_OVERFETCH` | 4096 | Hard cap on overfetch escalation |
| `COMPACTION_TOMBSTONE_THRESHOLD` | 50000 | Deleted vectors before rebuild |
| `BM25_ENABLED` | true | Master switch for the sparse leg |
| `BM25_MAX_CHUNKS` | 5000000 | Auto-disable the sparse leg past this many chunks |
| `BM25_REBUILD_THRESHOLD` | 20000 | New chunks triggering a sparse rebuild |
| `RRF_K` | 60 | Constant in RRF scoring `1/(k+rank)` |
| `RERANK_ENABLED` | false | Cross-encoder rerank on the fused pool |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sentences/CrossEncoder to load |
| `RERANK_CANDIDATE_POOL` | 50 | Candidates reranked per query |
| `LLAMAINDEX_ENABLED` | true | Agentic RAG service master switch |
| `LLAMAINDEX_MAX_DOCS` | 50000 | Hard cap above which the agentic index aborts |
| `INGEST_DEFAULT_CONCURRENCY` | 4 | Parallelism for directory ingestion |
| `INGEST_QUEUE_ENABLED` | true | Persistent background worker queue for bulk ingests |
| `INGEST_QUEUE_POLL_SECONDS` | 1.0 | Worker poll interval for pending jobs |
| `INGEST_QUEUE_MAX_CONCURRENT_JOBS` | 1 | Jobs processed simultaneously by the worker (bump to 2-3 on larger CPUs; embeddings are CPU-bound so gains come from API-legs, not worker count) |
| `PII_MASKING_ENABLED` | true | Master switch: mask PII before the LLM sees it |
| `PII_IN_ENABLED` | true | India entities (Aadhaar, PAN, GSTIN, IFSC, phone, passport) |
| `PII_US_ENABLED` | true | US entities (SSN, ITIN, phone, ZIP, routing, DL) |
| `PII_NER_ENABLED` | false | Optional NER (presidio-analyzer) for person names/orgs |
| `PII_NER_MIN_SCORE` | 0.5 | Minimum NER confidence to treat a span as PII |
| `PII_MAPPING_TTL_SECONDS` | 3600 | Lifetime of in-memory placeholder→value mappings |
| `PII_MAX_SESSIONS` | 500 | Bound on retained mapping namespaces |
| `PII_LOG_MASKING` | true | Mask PII in emitted log records too |

## Operational playbook

### Ingesting 1M+ documents
1. Point `DB_PATH` at a fast disk (NVMe strongly recommended: every chunk is both a WAL insert and a BLOB write).
2. Bump `EMBEDDING_CONCURRENCY` to your provider's rate limit (OpenAI allows high concurrency on text-embedding endpoints).
3. Bulk-load with the `ingest_directory` tool (`python mcp_server.py`) or `scripts/benchmark_scale.py --docs N --embedder fake` for a dry-run without API spend.
4. Let the maintenance thread rebuild BM25 in the background (`BM25_REBUILD_THRESHOLD`) so query latency isn't impacted.
5. Watch the live counters via the `library_stats` MCP tool (vector counts, tombstones, sparse doc counts).

### Memory footprint at rest
- SQLite + FAISS index: `~(dim/4 + dim) × num_vectors` with SQ8, plus ~1 KB metadata per chunk.
- On 32 GB RAM you can hold ~10M chunks on IVF/SQ8; on 64 GB RAM ~20M-25M chunks.
- For billions, move `chunk_vectors` to a dedicated vector DB (Qdrant, pgvector, Milvus) - the rest of this architecture (hybrid + rerank batches) stays the same.

### Tuning guide
- **Search is slow at 1M+**: increase `SEARCH_OVERFETCH`, reduce `IVF_NPROBE`, disable RERANK, reduce `BM25_ENABLED`.
- **Recall drops after bulk ingest**: kick a `vector_store.compact(force=True)` so IVF/SQ8 re-trains on the new distribution (clusters drift as data evolves).
- **Embedding leg rate-limited**: bump `EMBEDDING_MAX_RETRIES`, add provider-native rate limit headers feeding `EMBEDDING_CONCURRENCY` back.

### Known limits
- IVF/SQ8 recall is slightly lower than exact Flat; if you have RAM to spare for 10M vectors, set `VECTOR_INDEX_MODE=flat` for perfect recall.
- HNSW is supported (`VECTOR_INDEX_MODE=hnsw`) but consumes 1.2-2x raw dataset memory; it's the better option if most of your traffic is query-time.
- BM25 snowball index freezes while the maintenance lock rebuilds it; the dense leg continues serving.

## PII protection & ingestion queue

### Ingestion speed reality (be straight about the numbers)
Typical paper ≈ 30-60 chunks. "1M documents in 1 minute" would mean ~16,700
documents/s ≈ 500k-1M chunk embeddings/s. No single box - Space or GPU server -
embeds at that rate today; packet throughput like that is a distributed-cluster
problem (hundreds of GPUs or precomputed embeddings installed as a snapshot).
What this stack delivers, honestly:

| Path | Throughput (chunks/s) | 1M docs (~30M chunks) |
|---|---|---|
| HF Space cpu-basic, local 270m | ~0.8-1.7 | weeks - fine for incremental libraries |
| Desktop CPU, local 270m | ~1.7 | weeks |
| OpenAI-style API leg (`EMBEDDING_CONCURRENCY=8`, batch 256) | ~200-800+ | ~12-24 h when paid credits exist |
| One A100/H100-worker subprocess running the 0.6b/270m in fp16 | ~50-150 | days; combined with the API leg for mixed workloads |
| Distributed cluster (8+ GPUs / many API workers) | thousands | ~1-3 h |

Levers already wired: multi-file upload (all files enqueue at once), the
persistent queue (crash-safe resume + job progress), `EMBEDDING_CONCURRENCY`
+ `EMBEDDING_BATCH_SIZE` for the API leg, and `INGEST_QUEUE_MAX_CONCURRENT_JOBS`.
The realistic 2026 path to very-large libraries: ingest continuously with the
queue (API leg when credits exist), or attach a GPU runner for the local model.

### PII boundary (agent never sees raw sensitive data)
- Detection runs at every LLM boundary (`GenerativeTool.answer_question`, `summarize`): question + retrieval
  context are rewritten to `[EMAIL_1]`, `[AADHAAR_2]`, … placeholders before the model call, and the reply is
  re-identified back to real values **in memory only** — mappings are never persisted, logged, or written to disk.
- Engine: regex + checksum validators (Aadhaar Verhoeff, credit-card Luhn) with Presidio-style entity ids
  (`core/pii_patterns.py`), covering India (Aadhaar, PAN, GSTIN, IFSC, phones, passport), US (SSN, ITIN, phones,
  ZIP/routing/DL with context gating) and global (email, cards, IBAN, crypto wallets, IPs). Person names via
  optional NER (`presidio-analyzer`, `PII_NER_ENABLED`).
- Custom rules: users add keywords or regexes (persisted in the `meta` table) that get masked everywhere;
  the PII tab can disable any detector type per scan.
- Aadhaar nuance: the Verhoeff checksum decides the entity. Valid numbers are `IN_AADHAAR`; a 12-digit
  number that fails checksums but sits next to an Aadhaar keyword is still masked as `IN_AADHAAR_UNVERIFIED`
  (typo safety net — the raw digits must never reach the agent). A random 12-digit number with no Aadhaar
  context stays undetected to avoid false positives; checksum-validated matches always outrank the fallback.
  Detection remains best effort - no completeness or compliance claim is made.
- **We make no claim of completeness or regulatory compliance** — detection is best-effort; validate before
  regulated use. The voice/LlamaIndex agentic path is not masked today (see `mcp_tools/voice_tool.py`).

### Ingestion queue (worker split)
- Bulk ingests enqueue into `ingest_jobs` (SQLite); a dedicated worker thread drains them, so parsing/OCR
  crashes can't stall the API process and pending work survives restarts.
- Queue-aware tools: `ingest_directory_queue`, `ingest_queued`, `job_status`, `list_ingest_jobs`;
  UI panel on the "Ingestion Queue" tab. The direct `ingest_document`/`ingest_directory` tools stay synchronous.
