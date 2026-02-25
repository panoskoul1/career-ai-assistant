"""Persistent document registry backed by a JSON file.

Tracks all uploaded documents with full provenance:
  - document_id, document_type, filename, upload_timestamp
  - collection_name (Qdrant), is_active (for resumes)

This is the authoritative source of truth for metadata queries.
Metadata answers (counts, lists, active resume) are served directly from
this registry — no LLM reasoning required.

Storage: /app/uploads/document_registry.json (mounted Docker volume).
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from shared.models import DocumentRecord

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("/app/uploads/document_registry.json")


class DocumentRegistry:
    """Thread-safe JSON-file-backed document registry."""

    def __init__(self, storage_path: Path = _DEFAULT_PATH) -> None:
        self._path = storage_path
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, record: DocumentRecord) -> None:
        """Add or update a document record, then persist to disk."""
        with self._lock:
            self._records[record.document_id] = record
            self._save()
        logger.info(
            "Registered document: id=%s type=%s file=%s",
            record.document_id, record.document_type, record.filename,
        )

    def deactivate_resumes(self) -> None:
        """Mark all existing resumes as inactive before uploading a new one."""
        with self._lock:
            changed = False
            for doc_id, record in self._records.items():
                if record.document_type == "resume" and record.is_active:
                    self._records[doc_id] = record.model_copy(update={"is_active": False})
                    changed = True
            if changed:
                self._save()

    # ------------------------------------------------------------------
    # Read operations (no lock needed — dict reads are thread-safe in CPython)
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        return self._records.get(document_id)

    def list_jobs(self) -> list[DocumentRecord]:
        """Return all job_description records sorted by upload time (oldest first)."""
        return sorted(
            (r for r in self._records.values() if r.document_type == "job_description"),
            key=lambda r: r.upload_timestamp,
        )

    def list_resumes(self) -> list[DocumentRecord]:
        """Return all resume records sorted by upload time."""
        return sorted(
            (r for r in self._records.values() if r.document_type == "resume"),
            key=lambda r: r.upload_timestamp,
        )

    def get_active_resume(self) -> Optional[DocumentRecord]:
        """Return the most recently activated resume, or None."""
        active = [r for r in self._records.values()
                  if r.document_type == "resume" and r.is_active]
        if active:
            return max(active, key=lambda r: r.upload_timestamp)
        # Fall back to most recent resume even if not marked active
        resumes = self.list_resumes()
        return resumes[-1] if resumes else None

    def count_jobs(self) -> int:
        return sum(1 for r in self._records.values() if r.document_type == "job_description")

    def count_resumes(self) -> int:
        return sum(1 for r in self._records.values() if r.document_type == "resume")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("No registry file found at %s — starting empty", self._path)
            return
        try:
            data = json.loads(self._path.read_text())
            self._records = {k: DocumentRecord(**v) for k, v in data.items()}
            logger.info(
                "Registry loaded: %d total records (%d jobs, %d resumes)",
                len(self._records), self.count_jobs(), self.count_resumes(),
            )
        except Exception as exc:
            logger.error("Failed to load registry from %s: %s", self._path, exc)
            self._records = {}

    def _save(self) -> None:
        """Atomically write registry to disk. Caller must hold self._lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        data = {k: json.loads(v.model_dump_json()) for k, v in self._records.items()}
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)  # atomic rename


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[DocumentRegistry] = None


def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry
