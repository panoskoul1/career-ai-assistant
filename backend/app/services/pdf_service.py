"""PDF text extraction service."""

from __future__ import annotations

import io
import logging

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

MAX_PAGES = 100
MAX_TEXT_LENGTH = 500_000


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes.

    Raises ValueError if the PDF is too large or extraction fails.
    """
    reader = PdfReader(io.BytesIO(content))

    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDF has {len(reader.pages)} pages, maximum is {MAX_PAGES}")

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n\n".join(pages).strip()

    if not full_text:
        raise ValueError("Could not extract any text from PDF")

    if len(full_text) > MAX_TEXT_LENGTH:
        logger.warning("Truncating extracted text from %d to %d chars", len(full_text), MAX_TEXT_LENGTH)
        full_text = full_text[:MAX_TEXT_LENGTH]

    logger.info("Extracted %d characters from %d page PDF", len(full_text), len(reader.pages))
    return full_text
