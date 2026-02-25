"""LlamaIndex VectorStoreIndex store.

Lazily loads one VectorStoreIndex per Qdrant collection (one for the resume,
one per uploaded job). Re-uses the same Qdrant collections that fn-ingest
already populates — no double ingestion.

The embedding model MUST match the one used by fn-ingest
(BAAI/bge-small-en-v1.5, 384-dim, cosine, normalised).
"""

from __future__ import annotations

import logging
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

RESUME_COLLECTION = "resume_chunks"


class IndexStore:
    """Cache of lazily-loaded LlamaIndex VectorStoreIndex objects."""

    def __init__(self, qdrant_client: QdrantClient, embed_model: BaseEmbedding) -> None:
        self._client = qdrant_client
        self._embed_model = embed_model
        self._cache: dict[str, VectorStoreIndex] = {}

    # ------------------------------------------------------------------
    # Private loader
    # ------------------------------------------------------------------

    def _load(self, collection_name: str) -> Optional[VectorStoreIndex]:
        """Return a VectorStoreIndex for an existing Qdrant collection.

        Returns None if the collection does not exist yet.
        """
        if collection_name in self._cache:
            return self._cache[collection_name]

        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            logger.warning("Collection %s not found in Qdrant", collection_name)
            return None

        try:
            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=collection_name,
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embed_model,
            )
        except Exception as exc:
            # qdrant-client >= 1.9 calls /collections/{name}/exists which does
            # not exist on Qdrant server < 1.9 — treat it as a non-fatal miss.
            logger.warning("Could not build VectorStoreIndex for %s: %s", collection_name, exc)
            return None
        self._cache[collection_name] = index
        logger.info("Loaded index for collection %s", collection_name)
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self) -> Optional[VectorStoreIndex]:
        return self._load(RESUME_COLLECTION)

    def job(self, job_id: str) -> Optional[VectorStoreIndex]:
        return self._load(f"job_{job_id}")

    def invalidate(self, collection_name: str) -> None:
        """Evict a cached index (e.g. after re-upload)."""
        self._cache.pop(collection_name, None)
