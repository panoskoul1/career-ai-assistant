"""fn-ingest: Document chunking, embedding, and Qdrant storage.

Nuclio function contract:
  init_context(context)  — load SentenceTransformer + QdrantClient ONCE
  handler(context, event) — chunk text, embed, upsert to Qdrant

Expected POST body:
  {
    "text":            str,        # extracted document text
    "collection_name": str,        # Qdrant collection to write to
    "source":          str,        # "resume" | "job"
    "job_id":          str | null  # present for job documents
  }

Success response:
  { "status": "ok", "chunks": int, "collection": str }

This function is the ONLY place where sentence-transformers is loaded.
The backend never touches embeddings.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from uuid import uuid4

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 800     # estimated tokens (4 chars ≈ 1 token)
CHUNK_OVERLAP = 150  # estimated tokens


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
    """Load embedding model and Qdrant client ONCE at startup.

    Called by nuclio_runner.py before serving any requests.
    Equivalent to a Nuclio init_context() in production.
    """
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    context.logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    context.user_data.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    context.logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    context.user_data.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    context.logger.info("fn-ingest ready")


def handler(context, event) -> Response:
    """Chunk text, embed, and upsert to Qdrant.

    Stateless per request. All heavy objects live in context.user_data.
    """
    # --- Parse ---
    try:
        data = event.get_json()
        text: str = data["text"]
        collection_name: str = data["collection_name"]
        source: str = data.get("source", "unknown")
        job_id: str | None = data.get("job_id")
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return Response(
            body=json.dumps({"error": f"Bad request: {exc}"}),
            status_code=400,
        )

    # --- Chunk ---
    chunks = _chunk_text(text)
    if not chunks:
        return Response(
            body=json.dumps({"error": "No text chunks produced from input"}),
            status_code=400,
        )

    # --- Embed ---
    vectors = context.user_data.embed_model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    # --- Store ---
    _ensure_collection(context.user_data.qdrant, collection_name)
    _upsert(context.user_data.qdrant, collection_name, chunks, vectors, source, job_id)

    context.logger.info(f"Ingested {len(chunks)} chunks into '{collection_name}'")
    return Response(body=json.dumps({
        "status": "ok",
        "chunks": len(chunks),
        "collection": collection_name,
    }))


# ---------------------------------------------------------------------------
# Internal helpers — no ML imports, only stdlib + qdrant-client
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries.

    Uses the same algorithm as the original shared/chunking.py so that
    chunk size and overlap are consistent with what was used before.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = len(sentence) // 4
        if current_tokens + s_tokens > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            # Carry over tail overlap
            overlap: list[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                t = len(s) // 4
                if overlap_tokens + t > CHUNK_OVERLAP:
                    break
                overlap.insert(0, s)
                overlap_tokens += t
            current, current_tokens = overlap, overlap_tokens
        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _ensure_collection(qdrant, collection_name: str) -> None:
    """Create Qdrant collection if it does not exist."""
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in qdrant.get_collections().collections}
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _upsert(qdrant, collection_name: str, chunks, vectors, source, job_id) -> None:
    """Upsert chunk vectors into Qdrant."""
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "source": source,
                "job_id": job_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
