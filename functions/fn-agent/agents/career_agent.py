"""Career Intelligence Agent — factory and session management.

All initialization is deferred to build_components(), which is called
ONCE from init_context(). No module-level model loading — no side effects
on import.

Public API:
  build_components(ollama_base_url, qdrant_host, qdrant_port) -> dict
    Returns a dict of all initialized AI components.

  get_or_create_agent(sessions, tools, llm, session_id) -> ReActAgent
    Returns the existing per-session agent or creates a new one.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.1:8b"
LLM_TIMEOUT = 180.0
MEMORY_TOKEN_LIMIT = 2048
AGENT_MAX_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are a Career Intelligence Assistant. Your job is to help candidates
understand how their skills match job requirements and prepare for interviews.

You are a conversational assistant. Engage naturally with the user in any discussion.

Available tools:
- list_jobs(): list and count all uploaded job descriptions
- resume_summary(): structured overview of the uploaded resume
- fit_score(job_id): deterministic skill-coverage score (0.0–1.0)
- skill_gap_analysis(job_id): missing, matching, and bonus skills vs a job
- analyze_fit(job_id): deep fit analysis with grounded narrative
- job_ranking_based_on_fit(): rank ALL uploaded jobs by fit score
- interview_preparation_strategy(job_id): technical + behavioral + storytelling prep

CRITICAL RULES:
1. For greetings ("hello", "hi", "thanks") — respond conversationally without tools.
2. For questions about job ranking, fit scores, skill gaps, or interview prep —
   you MUST use the appropriate tool. Do NOT answer these from memory or general knowledge.
3. When you see [USE_TOOL: tool_name] at the start of the query, you MUST call that
   exact tool immediately in your first action. Do not reason about alternatives.
4. For "what is the best job", "rank jobs", "compare jobs", "which job fits best" —
   you MUST call job_ranking_based_on_fit(). Do not answer from memory.
5. For "what's my fit score", "how well do I fit" — you MUST call fit_score(job_id) or
   analyze_fit(job_id). Do not estimate or guess.
6. For "what skills am I missing", "show gaps" — you MUST call skill_gap_analysis(job_id).
7. Never fabricate skills, experience, or qualifications. If you need factual
   information about the user's background, use resume_summary() first.
8. If no resume or jobs have been uploaded and the user asks about them, say so clearly.
9. Format responses clearly with sections and bullet points when appropriate.
10. Keep responses concise and relevant. Avoid repeating yourself.\
"""


# ---------------------------------------------------------------------------
# Component factory — called once from init_context()
# ---------------------------------------------------------------------------

def build_components(
    ollama_base_url: str,
    qdrant_host: str,
    qdrant_port: int,
) -> dict:
    """Build and return all AI components.

    Heavy imports happen here — not at module load time.
    Call this exactly once from init_context().
    """
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from qdrant_client import QdrantClient

    from indexes.index_store import IndexStore
    from services.qdrant_reader import QdrantReader
    from tools import build_all_tools

    logger.info("Loading LLM: %s via %s", LLM_MODEL, ollama_base_url)
    llm = Ollama(
        model=LLM_MODEL,
        base_url=ollama_base_url,
        request_timeout=LLM_TIMEOUT,
        # llama3.1:8b defaults to a 128K context window which requires ~20 GB.
        # 4096 tokens is sufficient for this use case and fits in 16 GB RAM.
        context_window=4096,
        additional_kwargs={"options": {"num_ctx": 4096}},
    )

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, normalize=True)

    # Apply global LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    index_store = IndexStore(qdrant_client=qdrant_client, embed_model=embed_model)
    qdrant_reader = QdrantReader(client=qdrant_client)

    tools = build_all_tools(
        index_store=index_store,
        qdrant_reader=qdrant_reader,
        llm=llm,
    )

    logger.info(
        "Components built: LLM=%s embed=%s tools=%d",
        LLM_MODEL, EMBEDDING_MODEL, len(tools),
    )
    return {
        "llm": llm,
        "embed_model": embed_model,
        "index_store": index_store,
        "qdrant_reader": qdrant_reader,
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Session management — sessions dict lives in context.user_data.sessions
# ---------------------------------------------------------------------------

def get_or_create_agent(sessions: dict, tools: list, llm, session_id: str):
    """Get existing ReActAgent + memory for a session, or create a new one.

    Each session gets its own ChatMemoryBuffer so conversation history
    is isolated per user. Tools and the agent instance are shared within a
    session; memory is passed per-call so it accumulates across turns.

    Returns a (agent, memory) tuple.

    Note: llama-index >=0.14 replaced ReActAgent.from_tools() with a direct
    constructor and made agent invocation async (agent.run() → WorkflowHandler).
    """
    if session_id not in sessions:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.memory import ChatMemoryBuffer

        memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=True,
            streaming=False,
            system_prompt=SYSTEM_PROMPT,
            timeout=LLM_TIMEOUT,
        )
        sessions[session_id] = (agent, memory)
        logger.info("Created new agent session: %s", session_id)
    return sessions[session_id]  # (agent, memory)
