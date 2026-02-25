# functions/

Two Nuclio serverless functions that form the AI core of the career assistant. Each follows the Nuclio lifecycle: `init_context` loads heavy resources once at cold start; `handler` is stateless per request.

---

## fn-ingest

**Purpose:** Chunk, embed, and persist documents into Qdrant.

Accepts raw extracted text (resume or job description), splits it into overlapping sentence-boundary chunks (~800 token target, 150 token overlap), encodes them with `BAAI/bge-small-en-v1.5` (dim=384, cosine), and upserts into the target Qdrant collection. Creates the collection if it doesn't exist.

This is the **only** place `sentence-transformers` runs — the backend never handles embeddings directly.

**Contract**

```
POST /ingest
{
  "text":            str,
  "collection_name": str,
  "source":          "resume" | "job",
  "job_id":          str | null
}
→ { "status": "ok", "chunks": int, "collection": str }
```

---

## fn-agent

**Purpose:** Conversational career intelligence via a session-aware ReActAgent.

On init, loads an Ollama LLM (`llama3.1:8b`), a `QdrantReader`, and 7 LlamaIndex `FunctionTool`s. Sessions are stored in-process (`context.user_data.sessions`), one `ReActAgent` + `ChatMemoryBuffer` per `session_id`. A single `asyncio` event loop is created at init and reused across all requests to avoid loop teardown issues with llama-index ≥ 0.14.

**Routing** — every query passes through a 4-way intent classifier (LLM call, ~1–2s) before agent invocation:

| Intent | Path |
|---|---|
| `metadata` | Deterministic Qdrant lookup — no LLM, sub-100ms |
| `tool` | Tool name injected as a hint; ReActAgent converges in 1 iteration |
| `retrieval` | Full ReActAgent with semantic search |
| `conversational` | Direct LLM call, bypasses ReActAgent entirely |

**Tools (7)**

| Tool | Type |
|---|---|
| `list_jobs` | Metadata scan |
| `compute_fit_score` | Deterministic scoring |
| `list_skill_gaps` | Deterministic gap analysis |
| `analyze_fit` | Scoring + LLM narrative |
| `compare_jobs` (job_ranking_based_on_fit) | Multi-job scoring + LLM summary |
| `interview_plan` | Gap analysis + LLM questions |
| `resume_summary` | Skill extraction + LLM narrative |

**Contract**

```
POST /agent
{ "query": str, "session_id": str, "job_id": str | null }
→ { "answer": str, "session_id": str, "intent": str, "routed_via": str }

DELETE /session/{session_id}
→ { "status": "ok", "session_id": str }
```
