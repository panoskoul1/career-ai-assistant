"""Tool: interview_preparation_strategy(job_id)

Builds a structured interview preparation plan.
Skill gaps are computed deterministically; question generation uses the LLM.

Tool name: interview_preparation_strategy (target spec compliant)
"""

from __future__ import annotations

import json
import logging

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import InterviewPlan
from services.fit_scorer import skill_gap
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)

_TECHNICAL_PROMPT = """\
You are a senior technical interviewer. Based on the job description context and the
candidate's skill gaps listed below, generate exactly 5 likely technical interview
questions. Focus on the gaps — areas the candidate may be weak in.

Job context: {job_ctx}
Skill gaps: {gaps}

Return ONLY a JSON array of 5 question strings. Example:
["Question 1?", "Question 2?", ...]"""

_BEHAVIORAL_PROMPT = """\
You are a senior HR interviewer. Based on the job context below, generate exactly
5 likely behavioral interview questions (STAR format) relevant to this role.

Job context: {job_ctx}

Return ONLY a JSON array of 5 question strings."""

_STORYTELLING_PROMPT = """\
You are a career coach. Based on the candidate's resume highlights below,
suggest exactly 3 storytelling angles the candidate should prepare —
specific experiences they should be ready to narrate for this job.

Resume highlights: {resume_ctx}
Job context: {job_ctx}

Return ONLY a JSON array of 3 short suggestion strings."""


def _safe_parse_list(text: str, fallback_count: int = 5) -> list[str]:
    """Parse a JSON array from LLM output, with fallback."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
    # Fallback: split lines
    lines = [l.strip(" -•*\"'") for l in text.splitlines() if l.strip()]
    return lines[:fallback_count] if lines else [text[:200]]


def make_interview_plan_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
    llm: LLM,
) -> FunctionTool:

    def interview_preparation_strategy(job_id: str) -> str:
        """Generate a structured interview preparation strategy for a specific job.

        The plan includes:
        - Focus areas (deterministic, based on skill gaps)
        - 5 likely technical questions (LLM, grounded in job context)
        - 5 likely behavioral questions (LLM, grounded in job context)
        - 3 storytelling suggestions (LLM, grounded in resume highlights)
        - Prep tips

        Args:
            job_id: The unique identifier of the target job.

        Returns:
            JSON string with InterviewPlan fields.
        """
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet."})

        job_text = qdrant_reader.get_full_text(f"job_{job_id}")
        if not job_text:
            return json.dumps({"error": f"Job {job_id} not found."})

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        _, missing, _ = skill_gap(resume_skills, job_skills)

        # Grounded retrieval from indexes
        job_ctx = job_text[:1500]
        resume_ctx = resume_text[:1200]

        job_idx = index_store.job(job_id)
        resume_idx = index_store.resume()
        if job_idx:
            try:
                qe = job_idx.as_query_engine(llm=llm, similarity_top_k=4)
                job_ctx = str(qe.query("What are the key technical requirements and responsibilities?"))
            except Exception as exc:
                logger.warning("Job index query failed: %s", exc)

        if resume_idx:
            try:
                qe = resume_idx.as_query_engine(llm=llm, similarity_top_k=3)
                resume_ctx = str(qe.query("What are the candidate's most notable technical achievements?"))
            except Exception as exc:
                logger.warning("Resume index query failed: %s", exc)

        gaps_str = ", ".join(missing[:12]) if missing else "none identified"

        tech_questions: list[str] = []
        behavioral_questions: list[str] = []
        storytelling: list[str] = []

        try:
            raw = llm.complete(_TECHNICAL_PROMPT.format(job_ctx=job_ctx[:800], gaps=gaps_str)).text
            tech_questions = _safe_parse_list(raw, 5)
        except Exception as exc:
            logger.warning("Technical questions failed: %s", exc)

        try:
            raw = llm.complete(_BEHAVIORAL_PROMPT.format(job_ctx=job_ctx[:800])).text
            behavioral_questions = _safe_parse_list(raw, 5)
        except Exception as exc:
            logger.warning("Behavioral questions failed: %s", exc)

        try:
            raw = llm.complete(_STORYTELLING_PROMPT.format(
                resume_ctx=resume_ctx[:800], job_ctx=job_ctx[:600]
            )).text
            storytelling = _safe_parse_list(raw, 3)
        except Exception as exc:
            logger.warning("Storytelling suggestions failed: %s", exc)

        plan = InterviewPlan(
            job_id=job_id,
            focus_areas=missing[:8],
            technical_questions=tech_questions,
            behavioral_questions=behavioral_questions,
            storytelling_suggestions=storytelling,
            prep_tips=(
                "Review the missing skills listed in focus_areas. "
                "Prepare concrete STAR stories for behavioral questions. "
                "Brush up on any technical gaps before the interview."
            ),
        )
        return plan.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=interview_preparation_strategy,
        name="interview_preparation_strategy",
        description=(
            "Generate a structured interview preparation strategy for a specific job. "
            "Includes focus areas (skill gaps), technical interview questions, "
            "behavioral questions, and storytelling suggestions. "
            "Use this when the user asks about interview prep for a specific job."
        ),
    )
