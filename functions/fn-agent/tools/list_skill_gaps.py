"""Tool: skill_gap_analysis(job_id)

Deterministic skill gap analysis — no LLM. Returns three sorted lists:
missing (gaps), matching (strengths), and bonus (extra) skills.

Tool name: skill_gap_analysis (target spec compliant)
"""

from __future__ import annotations

import json
import logging

from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import SkillGapReport
from services.fit_scorer import skill_gap
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


def make_list_skill_gaps_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
) -> FunctionTool:

    def skill_gap_analysis(job_id: str) -> str:
        """List skill gaps between the resume and a specific job description.

        Returns three lists:
        - missing_skills: skills required by the job that the resume lacks (gaps to address)
        - matching_skills: skills present in both (strengths)
        - bonus_skills: resume skills not required by this job (transferable extras)

        Fully deterministic — no LLM used.

        Args:
            job_id: The unique identifier of the job to compare against.

        Returns:
            JSON string with SkillGapReport fields.
        """
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet."})

        job_text = qdrant_reader.get_full_text(f"job_{job_id}")
        if not job_text:
            return json.dumps({"error": f"Job {job_id} not found."})

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)

        matched, missing, bonus = skill_gap(resume_skills, job_skills)

        report = SkillGapReport(
            job_id=job_id,
            missing_skills=missing,
            matching_skills=matched,
            bonus_skills=bonus,
        )
        logger.info(
            "Skill gap for job %s: %d missing, %d matching, %d bonus",
            job_id, len(missing), len(matched), len(bonus),
        )
        return report.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=skill_gap_analysis,
        name="skill_gap_analysis",
        description=(
            "List the skill gaps between the candidate's resume and a specific job. "
            "Returns missing skills (gaps to address), matching skills (strengths), "
            "and bonus skills (extras not required). Fully deterministic — no LLM. "
            "Use this when the user asks what skills they are missing for a job."
        ),
    )
