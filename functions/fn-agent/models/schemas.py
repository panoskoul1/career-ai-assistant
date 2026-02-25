"""Pydantic output schemas for all agent tools."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class FitScore(BaseModel):
    job_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of job skills covered by resume")
    matched_skills: list[str] = Field(default_factory=list)
    total_job_skills: int = 0
    matched_count: int = 0


class SkillGapReport(BaseModel):
    job_id: str
    missing_skills: list[str] = Field(default_factory=list, description="Job skills absent from resume")
    matching_skills: list[str] = Field(default_factory=list, description="Skills present in both")
    bonus_skills: list[str] = Field(default_factory=list, description="Resume skills not required by job")


class FitAnalysis(BaseModel):
    job_id: str
    fit_score: float = Field(..., ge=0.0, le=1.0)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    resume_highlights: str = ""
    job_requirements_summary: str = ""
    narrative: str = ""


class RankedJob(BaseModel):
    job_id: str
    title: str = ""
    fit_score: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class JobComparison(BaseModel):
    ranked_jobs: list[RankedJob] = Field(default_factory=list)
    best_fit_job_id: Optional[str] = None
    summary: str = ""


class InterviewQuestion(BaseModel):
    category: str  # "technical" | "behavioral" | "storytelling"
    question: str
    context: str = ""


class InterviewPlan(BaseModel):
    job_id: str
    focus_areas: list[str] = Field(default_factory=list)
    technical_questions: list[str] = Field(default_factory=list)
    behavioral_questions: list[str] = Field(default_factory=list)
    storytelling_suggestions: list[str] = Field(default_factory=list)
    prep_tips: str = ""


class ResumeSummary(BaseModel):
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    experience_highlights: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    narrative: str = ""
