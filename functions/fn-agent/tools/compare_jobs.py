"""Tool: job_ranking_based_on_fit()

Ranks all uploaded job descriptions against the resume by deterministic
fit score. No LLM used for scoring — LLM writes only the summary sentence.

Tool name: job_ranking_based_on_fit (target spec compliant)
"""

from __future__ import annotations

import json
import logging

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import JobComparison, RankedJob
from services.fit_scorer import coverage_score, skill_gap
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


def make_compare_jobs_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
    llm: LLM,
) -> FunctionTool:

    def job_ranking_based_on_fit() -> str:
        """Compare all uploaded job descriptions against the resume and rank them
        by deterministic fit score (highest first).

        For each job returns: job_id, title snippet, fit_score, matched skills,
        and missing skills. Scoring is fully deterministic (no LLM).
        A one-line LLM summary is appended at the end.

        Returns:
            JSON string with JobComparison: ranked_jobs list and best_fit_job_id.
        """
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet."})

        job_ids = qdrant_reader.list_job_ids()
        if not job_ids:
            return json.dumps({"error": "No job descriptions uploaded yet."})

        resume_skills = extract_skills(resume_text)
        ranked: list[RankedJob] = []

        for job_id in job_ids:
            job_text = qdrant_reader.get_full_text(f"job_{job_id}")
            if not job_text:
                continue

            job_skills = extract_skills(job_text)
            score = coverage_score(resume_skills, job_skills)
            matched, missing, _ = skill_gap(resume_skills, job_skills)

            title = qdrant_reader.get_first_line(f"job_{job_id}")

            ranked.append(RankedJob(
                job_id=job_id,
                title=title or f"Job {job_id}",
                fit_score=score,
                matched_skills=matched[:10],
                missing_skills=missing[:10],
            ))

        ranked.sort(key=lambda j: j.fit_score, reverse=True)
        best_id = ranked[0].job_id if ranked else None

        # One-sentence LLM summary (scoring already done deterministically)
        summary = ""
        if ranked:
            try:
                snippet = "\n".join(
                    f"  {i+1}. {j.title[:60]} — score {j.fit_score:.0%}"
                    for i, j in enumerate(ranked[:5])
                )
                prompt = (
                    f"Based on these fit scores (higher is better), write one sentence "
                    f"recommending which job the candidate should prioritise and why:\n{snippet}"
                )
                summary = llm.complete(prompt).text.strip()
            except Exception as exc:
                logger.warning("Summary generation failed: %s", exc)

        comparison = JobComparison(
            ranked_jobs=ranked,
            best_fit_job_id=best_id,
            summary=summary,
        )
        logger.info(
            "Ranked %d jobs, best fit: %s (%.3f)",
            len(ranked), best_id, ranked[0].fit_score if ranked else 0,
        )
        return comparison.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=job_ranking_based_on_fit,
        name="job_ranking_based_on_fit",
        description=(
            "Compare ALL uploaded job descriptions against the resume and rank them "
            "by fit score (0.0–1.0, higher is better). Use this when the user asks "
            "which job fits best, wants a ranking, or asks to compare all jobs."
        ),
    )
