"""Text chunking utilities with configurable size and overlap."""

from __future__ import annotations

import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token for English)."""
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks based on estimated token count.

    Uses sentence boundaries when possible to avoid splitting mid-sentence.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build overlap from the end of current chunk
            overlap_chunk: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_tokens += s_tokens

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
