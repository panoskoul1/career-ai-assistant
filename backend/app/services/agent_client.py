"""HTTP client for communicating with the Nuclio agent microservice."""

from __future__ import annotations

import logging
import time

import httpx

from shared.models import AgentRequest, AgentResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

TIMEOUT = 180.0  # LLM reasoning can take time


async def call_agent(request: AgentRequest) -> AgentResponse:
    """Send a request to the Nuclio LlamaIndex agent and return the response."""
    url = f"{settings.NUCLIO_URL}/agent"
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=request.model_dump())
        resp.raise_for_status()

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Agent call session=%s latency=%.1f ms",
        request.session_id, latency_ms,
        extra={"latency_ms": latency_ms},
    )

    data = resp.json()
    return AgentResponse(**data)
