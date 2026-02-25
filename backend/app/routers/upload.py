"""Upload endpoints for resume and job descriptions.

The backend handles:
  - file validation and text extraction (PDF/plain text)
  - document registry updates (JSON file, no DB)
  - delegates chunking + embedding → fn-ingest via HTTP

The backend never loads ML models or writes to Qdrant directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, HTTPException

from shared.models import DocumentRecord, DocumentType, JobInfo
from app.core.config import settings
from app.services.pdf_service import extract_text_from_pdf
from app.services.fn_client import call_ingest
from app.services.job_store import add_job, job_collection_name
from app.services.document_registry import get_registry

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _validate_file(file: UploadFile) -> None:
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or plain text.",
        )


async def _read_content(file: UploadFile) -> str:
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )
    if file.content_type == "application/pdf":
        try:
            return extract_text_from_pdf(content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return content.decode("utf-8", errors="replace")


@router.post("/resume")
async def upload_resume(file: UploadFile = File(...)) -> dict:
    """Upload a resume PDF or text file.

    Extracts text, then delegates chunking + embedding to fn-ingest.
    Updates the document registry to mark this resume as active.
    """
    _validate_file(file)
    text = await _read_content(file)

    collection = settings.RESUME_COLLECTION

    try:
        result = await call_ingest(
            text=text,
            collection_name=collection,
            source=DocumentType.RESUME,
        )
    except Exception as exc:
        logger.error("fn-ingest unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="Ingest service unavailable")

    chunk_count = result.get("chunks", 0)

    # Registry: deactivate previous resumes, register the new one
    registry = get_registry()
    registry.deactivate_resumes()

    document_id = str(uuid4())
    registry.register(DocumentRecord(
        document_id=document_id,
        document_type="resume",
        filename=file.filename or "resume",
        upload_timestamp=datetime.now(timezone.utc),
        collection_name=collection,
        is_active=True,
        title=None,
    ))

    logger.info(
        "Resume uploaded: %s (doc_id=%s, %d chunks)", file.filename, document_id, chunk_count
    )
    return {
        "status": "ok",
        "document_id": document_id,
        "filename": file.filename,
        "chunks": chunk_count,
        "characters": len(text),
    }


@router.post("/job")
async def upload_job(file: UploadFile = File(...)) -> dict:
    """Upload a job description PDF or text file.

    Extracts text, then delegates chunking + embedding to fn-ingest.
    Each job gets its own Qdrant collection (job_{id}).
    Registers the job in the persistent document registry.
    """
    _validate_file(file)
    text = await _read_content(file)

    job_id = str(uuid4())[:8]
    collection = job_collection_name(job_id)

    try:
        result = await call_ingest(
            text=text,
            collection_name=collection,
            source=DocumentType.JOB,
            job_id=job_id,
        )
    except Exception as exc:
        logger.error("fn-ingest unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="Ingest service unavailable")

    chunk_count = result.get("chunks", 0)

    title = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
    title = (title or file.filename or "Untitled")[:100]

    add_job(JobInfo(job_id=job_id, title=title, filename=file.filename or "unknown"))

    document_id = str(uuid4())
    get_registry().register(DocumentRecord(
        document_id=document_id,
        document_type="job_description",
        filename=file.filename or "unknown",
        upload_timestamp=datetime.now(timezone.utc),
        collection_name=collection,
        is_active=True,
        title=title,
    ))

    logger.info(
        "Job uploaded: %s (job_id=%s, doc_id=%s, %d chunks)",
        file.filename, job_id, document_id, chunk_count,
    )
    return {
        "status": "ok",
        "job_id": job_id,
        "document_id": document_id,
        "title": title,
        "filename": file.filename,
        "chunks": chunk_count,
    }
