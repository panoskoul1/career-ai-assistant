"""Local embedding service using sentence-transformers with BGE-small."""

from __future__ import annotations

import time
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning vectors."""
    model = get_model()
    start = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Embedded %d texts in %.1f ms (%.1f ms/text)",
        len(texts), elapsed_ms, elapsed_ms / max(len(texts), 1),
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    return embed_texts([text])[0]
