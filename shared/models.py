"""Shared Pydantic models used across backend and nuclio-agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    RESUME = "resume"
    JOB = "job"


class ChunkMetadata(BaseModel):
    """Metadata stored alongside each vector chunk."""
    source: DocumentType
    job_id: Optional[str] = None
    section_type: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0


class DocumentRecord(BaseModel):
    """Persistent document metadata entry for the registry.

    Tracks every uploaded document with full provenance.
    This is the source of truth for metadata queries — no LLM required.
    """
    document_id: str
    document_type: Literal["resume", "job_description"]
    filename: str
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    collection_name: str
    is_active: bool = True
    title: Optional[str] = None


class ResumeAnalysis(BaseModel):
    """Structured output from the Resume Analyzer Agent."""
    skills: list[str] = Field(default_factory=list)
    years_of_experience: Optional[int] = None
    industries: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    summary: str = ""


class JobAnalysis(BaseModel):
    """Structured output from the Job Analyzer Agent."""
    job_id: str
    title: str = ""
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    seniority_level: str = ""
    domain: str = ""
    summary: str = ""


class GapAnalysis(BaseModel):
    """Structured output from the Gap Analysis Agent."""
    job_id: str
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    fit_score: float = 0.0
    interview_suggestions: list[str] = Field(default_factory=list)
    summary: str = ""


class ChatRequest(BaseModel):
    """Incoming chat request from the user."""
    message: str = Field(..., max_length=2000)
    job_id: Optional[str] = None
    session_id: str = Field(default="default", description="Unique session ID for chat memory")


class ChatResponse(BaseModel):
    """Chat response returned to the user."""
    answer: str
    sources: list[str] = Field(default_factory=list)
    agent_reasoning: Optional[dict] = None


class JobInfo(BaseModel):
    """Summary of an uploaded job — kept for frontend backward compatibility."""
    job_id: str
    title: str
    filename: str


class AgentRequest(BaseModel):
    """Request sent from backend to the Nuclio agent (LlamaIndex era)."""
    query: str
    session_id: str = "default"
    job_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from the Nuclio agent."""
    answer: str
    session_id: str
