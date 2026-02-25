"""Tool: analyze_fit(job_id)

Combines deterministic skill scoring with LlamaIndex retrieval to produce
a structured, grounded FitAnalysis with LLM-written narrative.

Performance note: query engines are cached per job_id in a closure-level
dict — they are built once and reused across calls, avoiding repeated
LLMSingleSelector initialization overhead.
"""

from __future__ import annotations

import json
import logging

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import FitAnalysis
from services.fit_scorer import coverage_score, skill_gap
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)

_NARRATIVE_PROMPT = """\
You are a career analyst. Based only on the context below, write a concise 3-4 sentence
narrative explaining how well the candidate fits this job.
Be specific — cite actual skills and experience. Do not invent anything.

Resume context:
{resume_ctx}

Job context:
{job_ctx}

Matched skills: {matched}
Missing skills: {missing}
Fit score: {score:.0%}

Write the narrative now:"""


def make_analyze_fit_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
    llm: LLM,
) -> FunctionTool:
    # Cache: job_id → query engine (built once, reused across calls)
    _job_qe_cache: dict = {}
    _resume_qe_cache: dict = {"qe": None}

    def _get_resume_qe():
        """Lazily build and cache the resume query engine."""
        if _resume_qe_cache["qe"] is None:
            resume_idx = index_store.resume()
            if resume_idx:
                _resume_qe_cache["qe"] = resume_idx.as_query_engine(
                    llm=llm, similarity_top_k=4
                )
        return _resume_qe_cache["qe"]

    def _get_job_qe(job_id: str):
        """Lazily build and cache the job query engine for a given job_id."""
        if job_id not in _job_qe_cache:
            job_idx = index_store.job(job_id)
            if job_idx:
                _job_qe_cache[job_id] = job_idx.as_query_engine(
                    llm=llm, similarity_top_k=4
                )
            else:
                _job_qe_cache[job_id] = None
        return _job_qe_cache[job_id]

    def analyze_fit(job_id: str) -> str:
        """Perform a deep fit analysis between the candidate's resume and a specific job.

        Steps:
        1. Computes deterministic skill-coverage score (no LLM).
        2. Retrieves relevant context from the resume index (cached query engine).
        3. Retrieves relevant context from the job index (cached query engine).
        4. Asks the LLM to write a grounded narrative (explanation only, score pre-computed).

        Args:
            job_id: The unique identifier of the job to analyse.

        Returns:
            JSON string with FitAnalysis fields including score, skills, and narrative.
        """
        # --- 1. Deterministic scoring (no LLM) ---
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet."})

        job_text = qdrant_reader.get_full_text(f"job_{job_id}")
        if not job_text:
            return json.dumps({"error": f"Job {job_id} not found."})

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        matched, missing, _ = skill_gap(resume_skills, job_skills)
        score = coverage_score(resume_skills, job_skills)

        # --- 2. Cached retrieval (no RouterQueryEngine rebuild per call) ---
        resume_ctx = resume_text[:1500]
        job_ctx = job_text[:1500]

        resume_qe = _get_resume_qe()
        if resume_qe:
            try:
                resume_ctx = str(resume_qe.query(
                    "What are the candidate's main technical skills and work experience?"
                ))
            except Exception as exc:
                logger.warning("Resume QE query failed: %s — using raw text fallback", exc)

        job_qe = _get_job_qe(job_id)
        if job_qe:
            try:
                job_ctx = str(job_qe.query(
                    "What are the key required skills and responsibilities for this job?"
                ))
            except Exception as exc:
                logger.warning("Job QE query failed for %s: %s — using raw text fallback", job_id, exc)

        # --- 3. LLM narrative (explanation only — score already computed) ---
        narrative = ""
        try:
            prompt = _NARRATIVE_PROMPT.format(
                resume_ctx=resume_ctx[:1200],
                job_ctx=job_ctx[:1200],
                matched=", ".join(matched[:15]) if matched else "none detected",
                missing=", ".join(missing[:15]) if missing else "none detected",
                score=score,
            )
            narrative = llm.complete(prompt).text.strip()
        except Exception as exc:
            logger.warning("Narrative generation failed: %s", exc)
            narrative = "Narrative unavailable."

        result = FitAnalysis(
            job_id=job_id,
            fit_score=score,
            matched_skills=matched,
            missing_skills=missing,
            resume_highlights=resume_ctx[:400],
            job_requirements_summary=job_ctx[:400],
            narrative=narrative,
        )
        return result.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=analyze_fit,
        name="analyze_fit",
        description=(
            "Perform a comprehensive fit analysis between the resume and a specific job. "
            "Returns matched skills, missing skills, fit score, and a grounded narrative. "
            "Use this when the user asks how well they fit a job or wants a full analysis."
        ),
    )
