"""Tool: fit_score(job_id)

Fully deterministic — no LLM. Extracts skills from both the resume and the
job description text then computes skill-coverage score (Jaccard variant).

Tool name: fit_score (target spec compliant)
"""

from __future__ import annotations

import json
import logging

from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import FitScore
from services.fit_scorer import coverage_score, skill_gap
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


def make_compute_fit_score_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
) -> FunctionTool:

    def fit_score(job_id: str) -> str:
        """Compute a deterministic fit score (0.0–1.0) between the uploaded resume
        and a specific job description.

        The score represents the fraction of the job's required skills that are
        present in the resume. No LLM is used — the result is reproducible.

        Args:
            job_id: The unique identifier of the job to score against.

        Returns:
            JSON string with FitScore fields: score, matched_skills,
            total_job_skills, matched_count.
        """
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet."})

        job_text = qdrant_reader.get_full_text(f"job_{job_id}")
        if not job_text:
            return json.dumps({"error": f"Job {job_id} not found or not uploaded yet."})

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)

        score = coverage_score(resume_skills, job_skills)
        matched, _, _ = skill_gap(resume_skills, job_skills)

        result = FitScore(
            job_id=job_id,
            score=score,
            matched_skills=matched,
            total_job_skills=len(job_skills),
            matched_count=len(matched),
        )
        logger.info("Fit score for job %s: %.3f (%d/%d skills)", job_id, score, len(matched), len(job_skills))
        return result.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=fit_score,
        name="fit_score",
        description=(
            "Compute a deterministic skill-coverage fit score (0.0–1.0) between "
            "the candidate's resume and a specific job. Higher means better fit. "
            "Use this when the user asks for a score, rating, or percentage match."
        ),
    )
