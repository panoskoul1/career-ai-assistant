"""Chat endpoint — forwards requests to the LlamaIndex Nuclio agent.

The backend no longer does vector retrieval. The nuclio-agent owns the full
RAG + reasoning pipeline via LlamaIndex. The backend only:
1. Validates the request.
2. Forwards it to the agent with session_id and optional job_id.
3. Returns the agent's response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from shared.models import AgentRequest, AgentResponse, ChatRequest, ChatResponse
from app.core.config import settings
from app.services.agent_client import call_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle a user chat message.

    Forwards the query, session_id, and optional job_id to the
    LlamaIndex ReActAgent running in the Nuclio container.
    """
    if len(request.message) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Message too long")

    agent_request = AgentRequest(
        query=request.message,
        session_id=request.session_id,
        job_id=request.job_id,
    )

    try:
        agent_response: AgentResponse = await call_agent(agent_request)
    except Exception as exc:
        logger.error("Agent call failed: %s", str(exc))
        raise HTTPException(status_code=502, detail="AI agent unavailable")

    return ChatResponse(
        answer=agent_response.answer,
        sources=[],
        agent_reasoning=None,
    )
