"""HTTP client for Nuclio AI functions.

The backend delegates ALL AI compute to Nuclio functions:
  - fn-ingest: document chunking, embedding, and Qdrant storage
  - fn-agent:  intent classification, ReActAgent reasoning

The backend never loads ML models or touches embeddings directly.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_INGEST_TIMEOUT = 60.0   # embed + Qdrant upsert
_AGENT_TIMEOUT = 180.0   # LLM reasoning can be slow


async def call_ingest(
    text: str,
    collection_name: str,
    source: str,
    job_id: str | None = None,
) -> dict:
    """Delegate document ingestion to fn-ingest.

    fn-ingest chunks the text, embeds with BAAI/bge-small-en-v1.5,
    and upserts to the specified Qdrant collection.

    Returns:
        {"status": "ok", "chunks": int, "collection": str}
    """
    async with httpx.AsyncClient(timeout=_INGEST_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.FN_INGEST_URL}/",
            json={
                "text": text,
                "collection_name": collection_name,
                "source": source,
                "job_id": job_id,
            },
        )
        resp.raise_for_status()

    result = resp.json()
    logger.info(
        "fn-ingest: %d chunks → %s",
        result.get("chunks", 0), collection_name,
    )
    return result
