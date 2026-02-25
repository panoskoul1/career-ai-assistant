"""Tool: list_jobs()

Lists all uploaded job descriptions with their IDs and titles.
Simple utility for counting and listing jobs without requiring a resume.
"""

from __future__ import annotations

import json
import logging

from llama_index.core.tools import FunctionTool

from services.qdrant_reader import QdrantReader

logger = logging.getLogger(__name__)


def make_list_jobs_tool(
    qdrant_reader: QdrantReader,
) -> FunctionTool:

    def list_jobs() -> str:
        """List all uploaded job descriptions with their IDs and titles.

        Returns a simple count and list of all jobs. Does not require a resume.
        Use this when the user asks how many jobs are uploaded, wants to see
        the list of jobs, or asks "what jobs do I have?".

        Returns:
            JSON string with count and list of jobs: {count: int, jobs: list[{job_id, title}]}
        """
        job_ids = qdrant_reader.list_job_ids()

        if not job_ids:
            return json.dumps({
                "count": 0,
                "jobs": [],
                "message": "No job descriptions uploaded yet."
            })

        jobs = []
        for job_id in job_ids:
            title = qdrant_reader.get_first_line(f"job_{job_id}")
            jobs.append({
                "job_id": job_id,
                "title": title or f"Job {job_id}",
            })

        result = {
            "count": len(jobs),
            "jobs": jobs,
        }

        logger.info("Listed %d jobs", len(jobs))
        return json.dumps(result, indent=2)

    return FunctionTool.from_defaults(
        fn=list_jobs,
        name="list_jobs",
        description=(
            "List all uploaded job descriptions with their IDs and titles. "
            "Use this when the user asks how many jobs are uploaded, wants to see "
            "the list of jobs, asks 'what jobs do I have?', or needs to count jobs. "
            "This tool does not require a resume to be uploaded."
        ),
    )
