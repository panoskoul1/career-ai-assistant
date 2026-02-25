"""FastAPI application entrypoint — API gateway only.

Responsibilities:
  - Route and validate HTTP requests
  - Orchestrate calls to Nuclio functions (fn-ingest, fn-agent)
  - Manage the document registry (JSON file, persistent metadata)
  - Proxy session management to fn-agent

NOT responsible for:
  - Loading ML models (zero ML deps in this service)
  - Accessing Qdrant directly (delegated to fn-ingest / fn-agent)
  - Embedding or chunking (delegated to fn-ingest)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logging_config import setup_logging
from app.routers import upload, chat, jobs

setup_logging()

app = FastAPI(
    title="Career Intelligence Assistant — Gateway",
    version="3.0.0",
    description="API gateway: routes requests, orchestrates Nuclio function calls",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, tags=["chat"])
app.include_router(jobs.router, tags=["jobs", "metadata"])


@app.delete("/session/{session_id}")
async def proxy_reset_session(session_id: str) -> dict:
    """Proxy session reset to fn-agent."""
    import httpx
    from app.core.config import settings
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(f"{settings.NUCLIO_URL}/session/{session_id}")
    except Exception:
        pass
    return {"status": "ok", "session_id": session_id}


@app.on_event("startup")
async def startup_event() -> None:
    import logging
    logger = logging.getLogger(__name__)

    from app.services.document_registry import get_registry
    registry = get_registry()
    logger.info(
        "Document registry ready: %d resumes, %d jobs",
        registry.count_resumes(), registry.count_jobs(),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
