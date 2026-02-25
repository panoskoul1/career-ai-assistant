"""fn-agent: Career Intelligence AI agent.

Nuclio function contract:
  init_context(context)  — load LLM, embeddings, tools; init session store ONCE
  handler(context, event) — classify intent, route, return answer

Routes handled:
  POST /agent                  { query, session_id, job_id? }  → agent chat
  DELETE /session/{session_id}                                  → clear session memory

Session memory (ReActAgent per session) is stored in context.user_data.sessions.
This is intentional in-process state — Nuclio functions support it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


# ---------------------------------------------------------------------------
# Response type — duck-typed by nuclio_runner._send()
# ---------------------------------------------------------------------------

@dataclass
class Response:
    body: str = ""
    status_code: int = 200
    content_type: str = "application/json"


# ---------------------------------------------------------------------------
# Nuclio lifecycle
# ---------------------------------------------------------------------------

def init_context(context) -> None:
    """Load all AI components ONCE at startup.

    Building LLM + embedding model + 7 tools takes ~15-30s.
    After init_context() returns, every request is fast.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    context.logger.info("Initialising fn-agent ...")

    from agents.career_agent import build_components

    components = build_components(
        ollama_base_url=OLLAMA_BASE_URL,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
    )

    context.user_data.llm = components["llm"]
    context.user_data.qdrant_reader = components["qdrant_reader"]
    context.user_data.tools = components["tools"]
    context.user_data.sessions = {}  # session_id → (ReActAgent, ChatMemoryBuffer)

    # Persistent event loop reused across ALL requests.
    # asyncio.run() closes the loop after each call; the workflow-based ReActAgent
    # (llama-index >=0.14) schedules asyncio.Tasks that become invalid once the
    # loop is closed. Reusing one loop keeps those tasks valid across requests.
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    context.user_data.loop = loop

    context.logger.info(f"fn-agent ready: {len(components['tools'])} tools loaded")


def handler(context, event) -> Response:
    """Route the request to session reset or agent chat."""

    # --- Session reset: DELETE /session/{id} ---
    if event.method == "DELETE" and "/session/" in event.path:
        session_id = event.path.split("/session/", 1)[1].strip("/")
        context.user_data.sessions.pop(session_id, None)
        logger.info("Session cleared: %s", session_id)
        return Response(body=json.dumps({"status": "ok", "session_id": session_id}))

    # --- Agent chat: POST /agent ---
    try:
        data = event.get_json()
        query: str = data["query"]
        session_id: str = data.get("session_id", "default")
        job_id: str | None = data.get("job_id")
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return Response(
            body=json.dumps({"error": f"Bad request: {exc}"}),
            status_code=400,
        )

    logger.info("Agent request: session=%s query=%.60s", session_id, query)

    try:
        answer, intent, routed_via = _classify_and_route(
            context, query, session_id, job_id
        )
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        return Response(
            body=json.dumps({"error": str(exc)}),
            status_code=502,
        )

    return Response(body=json.dumps({
        "answer": answer,
        "session_id": session_id,
        "intent": intent,
        "routed_via": routed_via,
    }))


# ---------------------------------------------------------------------------
# Internal: intent routing + agent invocation
# ---------------------------------------------------------------------------

def _classify_and_route(
    context,
    query: str,
    session_id: str,
    job_id: str | None,
) -> tuple[str, str, str]:
    """Classify intent then route to metadata fast-path or full ReActAgent.

    Returns (answer, intent, routed_via).
    """
    from router.intent_classifier import classify_intent, handle_metadata_query
    from agents.career_agent import get_or_create_agent, AGENT_MAX_ITERATIONS

    llm = context.user_data.llm
    qdrant_reader = context.user_data.qdrant_reader
    tools = context.user_data.tools
    sessions = context.user_data.sessions

    # --- Step 1: Intent classification (~1-2s LLM call) ---
    try:
        classification = classify_intent(query, llm)
    except Exception as exc:
        logger.warning("Classification failed: %s — routing to agent", exc)
        classification = None

    # --- Step 2: Metadata fast-path (deterministic, no ReAct loop) ---
    if classification and classification.requires_metadata:
        try:
            answer = handle_metadata_query(query, qdrant_reader)
            return answer, classification.intent, "metadata"
        except Exception as exc:
            logger.warning("Metadata handler failed: %s — falling back to agent", exc)

    # --- Step 3: Conversational fast-path (direct LLM, no ReAct loop) ---
    # mistral:7b struggles with the ReAct format for simple conversation — it
    # loops without converging. For greetings / general chat, a direct LLM call
    # is faster and more reliable.
    if classification and classification.intent == "conversational" and not classification.requires_tool:
        from agents.career_agent import SYSTEM_PROMPT
        from llama_index.core.llms import ChatMessage, MessageRole
        try:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=query),
            ]
            response = llm.chat(messages)
            answer = response.message.content
            logger.info("Conversational response via direct LLM (bypassed ReActAgent)")
            return answer, "conversational", "direct_llm"
        except Exception as exc:
            logger.warning("Direct LLM call failed: %s — falling back to agent", exc)

    # --- Step 4: Build effective query with routing hints ---
    effective_query = query
    if job_id:
        effective_query = f"[Selected job_id: {job_id}] {effective_query}"
    if classification and classification.requires_tool and classification.tool_name:
        effective_query = f"[USE_TOOL: {classification.tool_name}] {effective_query}"
        logger.info("Tool hint injected: %s", classification.tool_name)

    # --- Step 5: ReActAgent (async bridged via persistent event loop) ---
    # llama-index >=0.14: agent.run() schedules asyncio.Tasks immediately.
    # We use context.user_data.loop (never closed) instead of asyncio.run()
    # (which closes the loop after each call, invalidating subsequent Tasks).
    loop = context.user_data.loop

    agent, memory = get_or_create_agent(sessions, tools, llm, session_id)

    async def _invoke():
        handler = agent.run(
            effective_query,
            memory=memory,
            max_iterations=AGENT_MAX_ITERATIONS,
            early_stopping_method="generate",
        )
        return await handler

    result = loop.run_until_complete(_invoke())
    # result is AgentOutput; result.response is a ChatMessage
    answer = result.response.content

    intent = classification.intent if classification else "unknown"
    return answer, intent, "agent"
