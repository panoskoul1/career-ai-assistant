# Career AI Assistant

## Quick Setup

**Prerequisites:** Docker 20.10+, Docker Compose v2+, 16 GB RAM recommended, 10 GB free disk. NVIDIA GPU optional — roughly halves inference time.

```bash
cd career-ai-assistant

# Pull the LLM model (first time only — ~4.7 GB, 5–8 min)
docker compose run --rm ollama-init

# Build and start everything
docker compose up --build
```

First build takes 5–10 minutes. After that, `fn-agent` needs ~2 minutes to warm up the LLM and embedding model. Once you see `fn-agent ready: 7 tools loaded` in the logs, everything is live.

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | Upload resume, upload jobs, chat |
| Backend API docs | http://localhost:8000/docs | Swagger UI |
| fn-ingest health | http://localhost:9090/health | Chunk + embed + Qdrant write |
| fn-agent health | http://localhost:9091/health | Intent router + ReActAgent |
| Qdrant dashboard | http://localhost:6333/dashboard | Browse collections and vectors |

```bash
# Tear down (preserve data)
docker compose down

# Tear down + wipe all data
docker compose down -v
```

**No GPU?** Remove the `deploy.resources` block from the `ollama` service in `docker-compose.yml`. Inference will be slower (~15–20 s/response on CPU) but functional.

---

## Architecture Overview

```
Frontend (React/Vite, :3000)
    │  HTTP
    ▼
Backend (FastAPI gateway, :8000) — zero ML dependencies
    │  POST /ingest         │  POST /agent
    ▼                       ▼
fn-ingest (:9090)       fn-agent (:9091)
chunk + embed            intent classifier
+ Qdrant write           + ReActAgent + 7 tools
    │                       │
    └──────────┬────────────┘
               ▼
          Qdrant (:6333) ← Ollama (:11434)
```

The hard boundary is between the backend and the AI compute layer. The FastAPI backend handles file uploads, PDF extraction, document registry, and request routing. It has **no ML dependencies** — `torch`, `sentence-transformers`, and `qdrant-client` are not installed there. All AI compute lives in two isolated Nuclio-style functions:

- **fn-ingest** — receives extracted text, chunks on sentence boundaries, embeds with `BAAI/bge-small-en-v1.5`, upserts to Qdrant. One collection per document (`resume_chunks`, `job_<id>`). Cross-document retrieval contamination is architecturally impossible.

- **fn-agent** — handles every chat request through a 3-path router:
  - *Metadata fast-path* — "how many jobs do I have?" answered directly from Qdrant in ~50 ms, no LLM.
  - *Conversational fast-path* — greetings routed to a direct `llm.chat()` call, bypassing the ReAct loop. Exists because smaller models loop without converging on simple conversational turns.
  - *ReActAgent* — tool calls, analysis, retrieval. LlamaIndex ReActAgent with 7 `FunctionTools`, bounded `ChatMemoryBuffer` at 2048 tokens, and tool hints injected from the intent classifier to reduce iteration count.

Both functions follow the Nuclio contract: `init_context()` loads heavy objects once at startup, `handler()` is stateless per request. Locally they run via a 120-line stdlib `http.server` wrapper — no FastAPI inside functions.

---

## Why Nuclio-Style Functions

**The motivation:** Nuclio's `init_context()` / `handler()` contract was chosen to enforce a clean separation between one-time initialization (model loading, client setup) and per-request logic. Each AI capability — ingestion and agent — is an isolated, stateless function with no shared mutable state between requests. This makes the system modular by design: you can redeploy, scale, or replace one function without touching the other.

The contract also maps directly to production serverless platforms. On real Nuclio (or Kubernetes), these functions get auto-scaling, GPU scheduling, and event triggers for free — without changing `function.py`. The same code runs locally in Docker Compose and in production on Kubernetes. That's the point.

**Why a simulator, not real Nuclio (for now):** Running full Nuclio infrastructure locally means managing controllers, dashboards, and function deployments — overhead that slows down iteration when you're developing on a single machine. Instead, a lightweight simulator (`nuclio_runner.py`) wraps each function in a ~120-line stdlib `http.server`. No extra dependencies. `docker compose up` starts everything. You edit code, restart the container, and you're testing in seconds.

The trade-off is explicit: the simulator doesn't give you auto-scaling, event bindings, or the Nuclio dashboard. But the function code is fully Nuclio-compatible. Switching to real Nuclio later is a deployment and configuration change — the business logic in `function.py` stays unchanged.

---

## Productionization & Scaling

Current state: single machine, Docker Compose, 4–6 concurrent users before the inference queue backs up at ~8 s/response with `llama3.1:8b`.

The scaling story works because the architecture is already separated:

- **Backend** — stateless, scale freely behind any load balancer.
- **fn-ingest** — stateless, same.
- **fn-agent** — stateful (per-session ReActAgent in process). Needs sticky routing by `session_id`. The fix is Redis as a session backend (`RedisSimpleChatStore`), after which it scales freely too.
- **Qdrant** — distributed mode or Qdrant Cloud. Connection string change, not a rewrite.
- **Ollama** — the real bottleneck. In production, swap for a managed API (Bedrock, Vertex AI, Azure OpenAI). The LlamaIndex Ollama client wraps a standard HTTP interface — one-line config change.

| Component | AWS |
|---|---|
| Backend / fn-ingest | ECS Fargate |
| fn-agent | ECS on EC2 (reserved, sticky sessions via ALB) |
| Qdrant | EC2 r6i.xlarge + EBS gp3 |
| LLM | Bedrock (Claude) |
| Document registry | DynamoDB |
| Session memory | ElastiCache (Redis) |
| File storage | S3 (uploaded PDFs) |

fn-agent uses EC2-backed ECS rather than Fargate because it has a ~2 min warm-up (LLM client + embedding model init). Fargate cold starts compound that delay. With `min-instances=1` on Fargate you pay 24/7 anyway for something that behaves like a long-running server — at that point EC2 with a small reserved cluster is cheaper and avoids the cold start entirely. Fargate remains the right choice for backend and fn-ingest, which are lightweight and genuinely bursty.

S3 stores uploaded PDFs durably so documents survive container restarts and can be re-ingested (model upgrade, re-chunking) without asking users to re-upload. Lambda was considered and rejected — fn-agent's stateful session memory, persistent event loop, and 2-min warm-up disqualify it; fn-ingest's ~4 GB image with torch would have brutal cold starts.

**Missing observability:** Prometheus `/metrics` (latency histograms by intent, tool call counters, session gauge). OpenTelemetry spans around intent classification, tool calls, Qdrant queries, LLM completions. The structured log fields (`latency_ms`, `retrieval_scores`, `intent`, `routed_via`) are already emitted — they just need a collector.

**Security gaps:** No prompt injection detection. No rate limiting per session. No PII scrubbing on agent output. Current guardrails are limited to file type validation and `MAX_MESSAGE_LENGTH = 2000`.

---

## RAG / LLM Approach & Decisions

Core philosophy: **the LLM explains, it does not compute.** Every number the system emits — fit scores, matched skill counts, missing skill lists — comes from deterministic set math. The LLM receives pre-computed results and writes narrative around them.

```
resume_skills = extract_skills(resume_text)   ← regex, ~150-skill vocabulary
job_skills    = extract_skills(job_text)
score         = |resume_skills ∩ job_skills| / |job_skills|
matched       = resume_skills ∩ job_skills
missing       = job_skills − resume_skills
```

Why this matters: LLMs hallucinate numbers. A model that confidently says "you're a 78% fit" with nothing grounding that figure is worse than no score at all. The architecture makes it structurally impossible for the LLM to produce a score.

**Chunking:** Sentence-boundary splitting at ~800 estimated tokens (~3200 chars), 150-token overlap. Documents are short (1–3 pages), typically 3–8 chunks per document.

**Retrieval:** Top-k cosine similarity via LlamaIndex `VectorStoreIndex` over per-document Qdrant collections. No MMR — documents are too short for diversity to matter.

**Intent routing:** A constrained JSON prompt classifies each request into one of four intents: `metadata`, `tool`, `retrieval`, `conversational`. Adds ~1–2 s overhead but recovers it on metadata routes (~50 ms total vs. ~10 s through the full agent). The classifier injects a `[USE_TOOL: tool_name]` hint that typically reduces ReAct iterations from 3–5 to 1. On parse failure, falls back to the full agent.

**Memory:** `ChatMemoryBuffer` bounded at 2048 tokens. Tool outputs are verbose (500–800 tokens each) — a larger budget fills quickly and starts evicting early context. 2048 keeps 3–4 conversational turns visible.

---

## Key Technical Decisions

**One Qdrant collection per document.** The alternative — single collection with metadata filters — is simpler but introduces retrieval contamination risk and makes clean deletion harder. With one collection per document, deleting a job is `qdrant.delete_collection(f"job_{job_id}")`.

**Backend as a pure gateway.** The backend's `requirements.txt` is six packages: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `httpx`, `PyPDF2`. No torch. Image is ~200 MB vs. ~4 GB for fn-agent. Fast to build, fast to restart, immune to ML dependency conflicts.

**Persistent asyncio event loop in fn-agent.** LlamaIndex ≥0.14 made `agent.run()` async and schedules `asyncio.Task` objects immediately. Using `asyncio.run()` per request closes the event loop after each call, invalidating tasks in the next request. The fix: a single `asyncio.new_event_loop()` created in `init_context()` and reused across all requests. This took real debugging — not obvious from docs, and AI tools kept suggesting the broken pattern.

**JSON file document registry.** Writes to a Docker volume on every mutation, uses a threading lock and atomic rename. Zero extra dependencies. Sufficient for single-instance deployments. Doesn't work for multi-writer concurrent setups — at that point, swap for DynamoDB or Firestore.

---

## Engineering Standards

**Structure:** Each service is self-contained with a single responsibility. Hard rule: a service must not know about a layer it doesn't own. The backend doesn't know about Qdrant. fn-ingest doesn't know about the agent. fn-agent doesn't know about file uploads.

**Logging:** Structured JSON to stdout on all services. Fields include `timestamp`, `level`, `logger`, `message`, and optional structured fields (`latency_ms`, `retrieval_scores`, `intent`, `routed_via`). Routing path is traceable from logs alone.

**Testing:** None automated. The deterministic layer (`fit_scorer.py`, `skill_extractor.py`, `chunking.py`, `intent_classifier.py`) is all pure functions — easy to test in isolation. `notes.md` has a structured manual test script covering all 7 tools, routing paths, multi-turn memory, and edge cases.

**Containerisation:** Docker Compose with health checks on every service and strict dependency chain. `start_period: 120s` on fn-agent to cover model warm-up. Embedding model downloaded at build time, not runtime.

**Intentionally skipped:** Streaming responses, multi-resume support, automated tests, rate limiting, secrets management, CI/CD.

---

## AI Tool Usage

I used Cursor and Claude for mechanical, high-volume work: scaffolding FastAPI routers, LlamaIndex tool wrapper boilerplate, initial Pydantic models, Dockerfile structure. Getting to a working skeleton quickly was more valuable than writing it by hand.

**Where I overrode AI suggestions:**

- Intent classification routing and fallback behaviour — AI suggestions defaulted to cleaner-but-wrong patterns (e.g., catching exceptions too broadly, not distinguishing classification failure from low-confidence fallback).
- The `[USE_TOOL: tool_name]` hint injection — my addition, not suggested by AI. Meaningfully reduces agent iteration count.
- The persistent asyncio event loop — identified through debugging. AI kept suggesting `asyncio.run()`, which was the bug.
- The scoring pipeline (`fit_scorer.py`, `skill_extractor.py`, vocabulary list) — written by hand. If the deterministic core is wrong, the whole system is wrong.

**What I don't let AI do:** Make architectural decisions, define the boundary between deterministic and LLM-generated output, write system prompt rules, or own any logic with direct correctness consequences for user-facing results.

---

## What I'd Do Differently

- **Evaluation harness.** No eval set exists. I don't know the intent classifier's actual accuracy across a real query distribution. I'd build 50–100 labelled `(query, expected_intent, expected_tool)` pairs and run them on every code change.
- **Reranking.** Straight top-k cosine similarity works, but a cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`) after initial retrieval would improve chunk quality for `analyze_fit` and `interview_preparation_strategy`.
- **Resume-aware chunking.** Current strategy is sentence-boundary splits. Resumes have exploitable structure — work experience blocks, skill lists, education sections. A section-aware parser would let retrieval reason about "experience in X" vs. "skills in X."
- **Semantic skill normalization.** The regex vocabulary extractor has ~150 skills and misses anything outside the list. A spaCy NER model or fine-tuned skill extractor would handle synonyms ("ML" → "machine learning", "k8s" → "kubernetes").
- **Streaming.** The biggest UX problem is the 8–15 second wait with no feedback. Token streaming from Ollama → LlamaIndex → FastAPI SSE → frontend would fix this. LlamaIndex supports it; the frontend just needs an SSE consumer.
- **Persistent sessions.** Redis-backed `ChatMemoryBuffer` so sessions survive restarts and work across replicas.
- **Tests.** Start with the deterministic layer: fit scorer edge cases, skill extractor coverage, chunking boundaries, intent classifier with mocked LLM. Then integration tests for routing paths.

