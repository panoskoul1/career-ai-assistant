"""Tool: resume_summary()

Queries the resume VectorStoreIndex via LlamaIndex to produce a structured
summary of the candidate's skills, technologies, experience, and education.
"""

from __future__ import annotations

import json
import logging

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from models.schemas import ResumeSummary
from services.qdrant_reader import QdrantReader
from services.skill_extractor import extract_skills

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are a career analyst. Summarise this candidate's resume in structured form.

Resume context:
{resume_ctx}

Detected technologies/skills: {skills}

Write a 3-4 sentence professional narrative summarising their background,
strongest technical areas, and career trajectory. Be specific and factual.
Do not invent anything not present in the context."""


def make_resume_summary_tool(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
    llm: LLM,
) -> FunctionTool:

    def resume_summary() -> str:
        """Provide a structured summary of the uploaded resume.

        Extracts detected skills and technologies (deterministic), then
        uses the LlamaIndex resume index to retrieve experience highlights
        and education. The LLM writes a grounded narrative summary.

        Returns:
            JSON string with ResumeSummary fields: skills, technologies,
            experience_highlights, education, narrative.
        """
        resume_text = qdrant_reader.get_full_text("resume_chunks")
        if not resume_text:
            return json.dumps({"error": "Resume not uploaded yet. Please upload a resume first."})

        detected_skills = sorted(extract_skills(resume_text))

        resume_ctx = resume_text[:2000]
        experience_highlights: list[str] = []
        education: list[str] = []

        resume_idx = index_store.resume()
        if resume_idx:
            try:
                qe = resume_idx.as_query_engine(llm=llm, similarity_top_k=5)

                exp_response = qe.query(
                    "List the candidate's work experience, job titles, companies, and key achievements."
                )
                experience_highlights = [
                    line.strip(" -•*")
                    for line in str(exp_response).splitlines()
                    if line.strip()
                ][:8]

                edu_response = qe.query(
                    "What degrees, certifications, or educational qualifications does the candidate have?"
                )
                education = [
                    line.strip(" -•*")
                    for line in str(edu_response).splitlines()
                    if line.strip()
                ][:5]

                resume_ctx = str(exp_response)[:1500]
            except Exception as exc:
                logger.warning("Resume index query failed: %s", exc)

        # LLM narrative
        narrative = ""
        try:
            prompt = _SUMMARY_PROMPT.format(
                resume_ctx=resume_ctx[:1200],
                skills=", ".join(detected_skills[:25]),
            )
            narrative = llm.complete(prompt).text.strip()
        except Exception as exc:
            logger.warning("Narrative generation failed: %s", exc)

        # Split skills into general skills vs technologies
        tech_keywords = {
            "pytorch", "tensorflow", "keras", "opencv", "yolo", "docker", "kubernetes",
            "qdrant", "pinecone", "mlflow", "onnx", "openvino", "fastapi", "aws", "gcp",
            "azure", "python", "c++", "java", "typescript", "spark", "kafka", "huggingface",
        }
        technologies = sorted(s for s in detected_skills if s in tech_keywords)
        skills = sorted(s for s in detected_skills if s not in tech_keywords)

        summary = ResumeSummary(
            skills=skills,
            technologies=technologies,
            experience_highlights=experience_highlights,
            education=education,
            narrative=narrative,
        )
        return summary.model_dump_json(indent=2)

    return FunctionTool.from_defaults(
        fn=resume_summary,
        name="resume_summary",
        description=(
            "Provide a structured summary of the uploaded resume including skills, "
            "technologies, experience highlights, education, and a narrative overview. "
            "Use this when the user asks about their resume, background, or profile."
        ),
    )
