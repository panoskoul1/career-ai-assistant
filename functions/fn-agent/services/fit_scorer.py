"""Deterministic fit scorer.

Computes skill coverage: what fraction of the job's required skills
the candidate's resume covers. No LLM involved.
"""

from __future__ import annotations


def coverage_score(resume_skills: set[str], job_skills: set[str]) -> float:
    """Return |resume ∩ job| / |job|.

    A score of 1.0 means the resume covers every skill the job asks for.
    Returns 0.0 if job_skills is empty.
    """
    if not job_skills:
        return 0.0
    matched = resume_skills & job_skills
    return round(len(matched) / len(job_skills), 4)


def skill_gap(resume_skills: set[str], job_skills: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (matched, missing, bonus) skill lists.

    - matched: skills in both resume and job
    - missing: job skills absent from resume (gaps)
    - bonus:   resume skills not required by the job (extras)
    """
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    bonus = sorted(resume_skills - job_skills)
    return matched, missing, bonus
