"""Qdrant full-text reader.

Scrolls all chunks from a Qdrant collection and returns them sorted
by chunk_index so the reconstructed text is in the original document order.
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_SCROLL_BATCH = 100


class QdrantReader:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def collection_exists(self, collection_name: str) -> bool:
        names = {c.name for c in self._client.get_collections().collections}
        return collection_name in names

    def list_job_ids(self) -> list[str]:
        """Return all job_ids by scanning collections named job_<id>."""
        names = {c.name for c in self._client.get_collections().collections}
        return [n[4:] for n in names if n.startswith("job_")]

    def get_full_text(self, collection_name: str) -> Optional[str]:
        """Return ordered, concatenated text from all chunks in a collection.

        Returns None if the collection does not exist or is empty.
        """
        if not self.collection_exists(collection_name):
            logger.warning("Collection %s does not exist", collection_name)
            return None

        chunks: list[tuple[int, str]] = []
        offset = None

        while True:
            result, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=_SCROLL_BATCH,
                with_payload=True,
                offset=offset,
            )
            for point in result:
                payload = point.payload or {}
                text = payload.get("text", "")
                idx = payload.get("chunk_index", 0)
                if text:
                    chunks.append((idx, text))

            if next_offset is None:
                break
            offset = next_offset

        if not chunks:
            return None

        chunks.sort(key=lambda x: x[0])
        full_text = "\n\n".join(t for _, t in chunks)
        logger.info("Read %d chunks from %s (%d chars)", len(chunks), collection_name, len(full_text))
        return full_text

    def get_first_line(self, collection_name: str) -> str:
        """Return the first non-empty line of the chunk with chunk_index=0.

        Scroll may return points in any order, so we fetch up to 20 and pick
        the one with the lowest chunk_index rather than trusting the first hit.
        """
        if not self.collection_exists(collection_name):
            return ""
        result, _ = self._client.scroll(
            collection_name=collection_name,
            limit=20,
            with_payload=True,
        )
        if not result:
            return ""
        # Sort by chunk_index; fall back to 0 if the field is absent
        points = sorted(result, key=lambda p: (p.payload or {}).get("chunk_index", 0))
        text = (points[0].payload or {}).get("text", "")
        # Collapse PDF whitespace artefacts before scanning lines
        text = " ".join(text.split())
        for line in text.split("."):
            line = line.strip()
            if line and len(line) > 5:
                return line[:120]
        return ""
