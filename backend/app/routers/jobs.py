"""Job listing and document metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from shared.models import DocumentRecord, JobInfo
from app.services.job_store import list_jobs
from app.services.document_registry import get_registry

router = APIRouter()


@router.get("/jobs", response_model=list[JobInfo])
async def get_jobs() -> list[JobInfo]:
    """Return all uploaded job descriptions.

    Used by the frontend JobSelector. Returns the legacy JobInfo format
    for backward compatibility.
    """
    return list_jobs()


@router.get("/metadata/documents", response_model=list[DocumentRecord])
async def get_all_documents() -> list[DocumentRecord]:
    """Return the full document registry — all uploaded resumes and jobs.

    Includes document_type, filename, upload_timestamp, is_active, and
    collection_name for every document ever uploaded in this session.
    This endpoint answers metadata queries without LLM involvement:
      - How many job descriptions are uploaded?
      - Which resume is active?
      - When was a document uploaded?
    """
    registry = get_registry()
    jobs = registry.list_jobs()
    resumes = registry.list_resumes()
    return resumes + jobs


@router.get("/metadata/resume/active", response_model=DocumentRecord | None)
async def get_active_resume() -> DocumentRecord | None:
    """Return the currently active resume record, or null if none uploaded."""
    return get_registry().get_active_resume()


@router.get("/metadata/stats")
async def get_metadata_stats() -> dict:
    """Return a summary of uploaded document counts.

    Suitable for a quick metadata panel in the UI.
    """
    registry = get_registry()
    active_resume = registry.get_active_resume()
    return {
        "total_resumes": registry.count_resumes(),
        "total_jobs": registry.count_jobs(),
        "active_resume": active_resume.filename if active_resume else None,
        "active_resume_uploaded_at": (
            active_resume.upload_timestamp.isoformat() if active_resume else None
        ),
        "jobs": [
            {
                "document_id": r.document_id,
                "title": r.title or r.filename,
                "filename": r.filename,
                "uploaded_at": r.upload_timestamp.isoformat(),
            }
            for r in registry.list_jobs()
        ],
    }
