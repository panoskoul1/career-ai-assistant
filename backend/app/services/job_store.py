"""Job metadata store — backed by the document registry.

After the architecture refactoring, the backend no longer has direct
Qdrant access. Job metadata is sourced from the document registry
(JSON file at /app/uploads/document_registry.json).

This module preserves the same public interface so the routers
(jobs.py, upload.py) do not need changes.
"""

from __future__ import annotations

import logging

from shared.models import JobInfo

logger = logging.getLogger(__name__)


# def add_job(job: JobInfo) -> None:
#     """No-op — jobs are already registered via get_registry().register() in upload.py."""
#     pass


def get_job(job_id: str) -> JobInfo | None:
    from app.services.document_registry import get_registry
    for record in get_registry().list_jobs():
        if record.collection_name == f"job_{job_id}":
            return JobInfo(
                job_id=job_id,
                title=record.title or record.filename,
                filename=record.filename,
            )
    return None


def list_jobs() -> list[JobInfo]:
    from app.services.document_registry import get_registry
    result = []
    for record in get_registry().list_jobs():
        if record.collection_name.startswith("job_"):
            job_id = record.collection_name[4:]  # strip "job_" prefix
            result.append(JobInfo(
                job_id=job_id,
                title=record.title or record.filename,
                filename=record.filename,
            ))
    return result


def job_collection_name(job_id: str) -> str:
    """Return the Qdrant collection name for a specific job."""
    return f"job_{job_id}"
