# Alex AI — Production-Grade RAG Assistant

A **production-ready Retrieval-Augmented Generation (RAG) system** built to simulate a real-world AI assistant that represents a candidate's professional profile. Designed to answer recruiter-style questions with grounded, low-hallucination responses via real-time streaming.

---

## Related Repositories

| | |
|---|---|
| **Frontend** | [GitHub](https://github.com/PKan06/resume_ai_backend) <!-- replace # with your frontend repo URL --> · [Live Demo](https://tushar-resume-ai.vercel.app/) |
| **Backend** (this repo) | RAG pipeline, Pinecone, Cloudflare embeddings, Gemini streaming |

The frontend is a React + Vite SPA with real-time token streaming, an animated Gemini-style thinking UI, and per-message latency metrics (TOFT / TOLT).

---

## Project Goal

To build a **real-time, streaming AI assistant** that:

- Answers recruiter-style questions from resume data
- Minimises hallucinations via grounded retrieval
- Exposes per-response evaluation metrics
- Runs efficiently on minimal cloud infrastructure (no local model weights)

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │               CLIENT (Browser)               │
                        └──────────────────┬──────────────────────────┘
                                           │  HTTP POST /chat  (streaming)
                        ┌──────────────────▼──────────────────────────┐
                        │           FastAPI  (main.py)                 │
                        │   CORS · SessionManager · Cancel Token       │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │       FastAPIPersonalAssistant               │
                        │            (core/assistant.py)               │
                        └───┬──────────────┬───────────────┬──────────┘
                            │              │               │
               ┌────────────▼───┐   ┌──────▼──────┐  ┌───▼────────────┐
               │  Query Router  │   │  Semantic   │  │  RAG Evaluator │
               │  (LLM classify)│   │  Cache      │  │  (post-resp)   │
               │  + multi-query │   │  (Redis)    │  └────────────────┘
               │  generation    │   └──────┬──────┘
               └────────┬───────┘          │ cache hit → return early
                        │ miss             │
                        │◄─────────────────┘
                        │
          ┌─────────────▼──────────────────────────────┐
          │           VectorStoreManager               │
          │           (rag/vector_store.py)             │
          │                                            │
          │  for each of 3 generated queries:          │
          │    1. embed_query()                        │
          │       │                                    │
          │       ▼                                    │
          │  ┌─────────────────────────┐               │
          │  │  Cloudflare Workers AI  │               │
          │  │  @cf/baai/bge-small-    │               │
          │  │  en-v1.5  (384-dim)     │               │
          │  └──────────┬──────────────┘               │
          │             │ 384-dim vector                │
          │             ▼                              │
          │  ┌─────────────────────────┐               │
          │  │    Pinecone Serverless  │               │
          │  │  index.query(top_k=5)  │               │
          │  │  cosine similarity      │               │
          │  └──────────┬──────────────┘               │
          │             │ 5 matches + scores            │
          │             ▼                              │
          │   Dedup · Max-score fusion                  │
          │   Threshold filter (≥ 0.35)                │
          │   Top-K selection (default 3)              │
          └─────────────┬──────────────────────────────┘
                        │ context chunks
                        ▼
          ┌─────────────────────────────────────────────┐
          │              RagChain  (rag/rag_chain.py)    │
          │   Build prompt  →  LangchainLLM wrapper      │
          └─────────────┬───────────────────────────────┘
                        │
          ┌─────────────▼───────────────────────────────┐
          │         StreamingLLM  (llm/streaming_llm.py) │
          │   google-genai  →  Gemini Flash Lite         │
          │   Token-by-token streaming + retry logic     │
          └─────────────┬───────────────────────────────┘
                        │ token stream
                        ▼
                   Client (TOFT / TOLT metrics)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI (async streaming) |
| **LLM** | Gemini Flash Lite via `google-genai` SDK |
| **Embeddings** | Cloudflare Workers AI — `@cf/baai/bge-small-en-v1.5` (384-dim) |
| **Vector DB** | Pinecone Serverless (cosine similarity) |
| **Semantic Cache** | Redis (optional) |
| **RAG Framework** | LangChain (partial — chains + tracing only) |
| **Observability** | LangSmith (`@traceable`) |
| **PDF Loading** | LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter` |

---

## Core Features

### 1. Multi-Query Retrieval

Generates 3 semantic variations of each query before hitting Pinecone — improves recall for questions that can be phrased multiple ways.

```
"What cloud technologies do you know?"
→ query 1: "cloud infrastructure AWS GCP experience"
→ query 2: "DevOps containerization Kubernetes Docker"
→ query 3: "cloud deployment architecture skills"
```

Each query embeds independently → Pinecone returns top-5 matches → results fused.

### 2. Score-Based Reranking

After multi-query retrieval, results are deduplicated and reranked by cosine score. A chunk appearing in multiple query results keeps its highest score — acting as a natural relevance signal.

### 3. Semantic Cache (Redis)

Cosine similarity lookup against past queries (threshold 0.85). Near-duplicate questions return instantly without hitting Cloudflare or Pinecone.

### 4. Streaming Responses

Token-by-token output via async generators. Two latency metrics emitted per response:
- **TOFT** — Time to First Token
- **TOLT** — Time to Last Token

### 5. Cloudflare-Backed Embeddings (No Local Model)

Embeddings are computed via HTTP calls to Cloudflare Workers AI — no `torch`, no `sentence-transformers`, no local model weights. Deployment footprint is minimal.

### 6. Pinecone Persistent Index

Documents are upserted once on first startup and persist. Subsequent cold starts connect in milliseconds — no re-embedding on every deploy.

### 7. RAG Evaluation Harness

Each response is evaluated inline:
- **Context Score** — are retrieved chunks relevant to the query?
- **Answer Relevance** — does the answer address the question?
- **Faithfulness** — is the answer grounded in the retrieved context?

---

## Retrieval Metrics (Live Eval — 55 Questions)

Evaluated against a 55-question dataset using `@cf/baai/bge-small-en-v1.5` embeddings and Gemini Flash Lite as the generator.

| Metric | Value |
|---|---|
| Retrieval precision | **0.660** |
| Context recall | **0.739** |
| Section hit rate | **0.964** |
| Answer relevance | **0.669** |
| Faithfulness | **0.731** |
| Context score | **0.665** |
| Cache hit rate | 0.182 |
| Avg latency | 25.2 s |
| P95 latency | 43.9 s |

**Config:** Top-K = 10, threshold = 0.40, Redis enabled

### Weakest Areas

| Question | Precision | Recall | Section Hit |
|---|---|---|---|
| `education` | 0.143 | 0.25 | 1.0 |
| `education_v2` | 0.167 | 0.20 | 0.0 |
| `coursework` | 0.375 | 0.60 | 0.0 |
| `current_role` | 0.286 | 0.00 | 1.0 |
| `current_role_v2` | 0.286 | 0.00 | 1.0 |

Education and current-role queries have low recall — the section is found (hit rate 1.0) but the specific answer chunks are not being ranked high enough. Targeted chunking strategy for these sections is the next improvement.

---

## Why Not Just LangChain?

| Feature | LangChain Default | This System |
|---|---|---|
| Retriever | Basic single-query similarity | Multi-query + score fusion |
| Context quality | Moderate | High (0.665 context score) |
| Embeddings | Local model (heavy) | Cloudflare API (zero weight) |
| Vector DB | In-memory FAISS | Pinecone (persistent, cloud) |
| Evaluation | None | Built-in per-response metrics |
| Streaming | Basic | Retry + cancel-token aware |

---

## Project Structure

```
server/
├── main.py                    # FastAPI app, lifespan, endpoints
├── config.py                  # All env vars — single source of truth
├── logger.py                  # Structured logging setup
│
├── core/
│   ├── assistant.py           # Orchestrator — routing, retrieval, streaming
│   └── session_manager.py     # Per-session cancel tokens
│
├── rag/
│   ├── vector_store.py        # Pinecone connect/upsert/query
│   ├── rag_chain.py           # Prompt construction + LLM call
│   ├── document_processor.py  # PDF load, chunk, enrich
│   └── profile_extractor.py   # Name/skills extraction from resume
│
├── services/
│   ├── embedding_service.py   # Cloudflare Workers AI singleton
│   ├── semantic_cache.py      # Redis cosine-similarity cache
│   └── rag_evaluator.py       # Inline context/faithfulness scoring
│
├── llm/
│   ├── streaming_llm.py       # Raw Gemini streaming + retry
│   └── langchain_llm.py       # LangChain Runnable wrapper + @traceable
│
├── schemas/
│   └── chat.py                # Pydantic request/response models
│
└── eval/
    ├── run_eval.py             # Evaluation harness CLI
    └── results/               # Timestamped eval reports (JSON + MD)
```

---

## Running Locally

```bash
pip install -r requirement.txt
uvicorn server.main:app --reload
```

Required `.env` variables:
```
GEMINI_API_KEY=
MODEL_NAME=gemini-3.1-flash-lite
PDF_PATH=./data/resume.pdf
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
PINECONE_API_KEY=
PINECONE_INDEX_NAME=resume-ai
```

On first startup with an empty Pinecone index, all PDF chunks are automatically embedded and upserted. Subsequent startups reconnect instantly.

---

## Deployment Notes

- No local model weights — `torch` and `sentence-transformers` are not required
- Lazy initialisation: VectorStoreManager connects only on first professional query
- Graceful cancel: in-flight HTTP requests (Cloudflare, Pinecone) abort immediately on WebSocket disconnect
- Rate limit retry: exponential backoff on both Gemini and Cloudflare calls

---

## Key Engineering Decisions

**Custom retrieval over LangChain retriever** — LangChain's default retriever does a single ANN lookup. Replacing it with multi-query generation + score fusion lifted context recall from ~0.3 to 0.739.

**Cloudflare embeddings over local model** — eliminates `torch`/`transformers` from the dependency tree entirely, reducing Docker image size by ~2 GB and removing the cold-start penalty of loading model weights.

**Pinecone over FAISS** — FAISS rebuilds the index from scratch on every cold start (expensive embedding calls). Pinecone persists vectors permanently; cold starts are instant after the first deploy.

**Cancel token propagation** — a `threading.Event` is threaded from the WebSocket disconnect handler all the way into the embedding retry loop, so aborted requests release resources immediately rather than waiting for the 30s read timeout.

---

## Future Improvements

- Hybrid search (BM25 + dense vector) for exact-keyword queries (names, tech versions)
- Cross-encoder neural reranking (Cohere Rerank) for higher faithfulness
- Memory-aware multi-turn conversations
- Improved chunking strategy for education and current-role sections (low recall in current eval)
