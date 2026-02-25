"""Intent-based router for the Career Intelligence Agent.

Classifies each incoming user query into one of four intents using
an LLM with a constrained JSON prompt. The classification result
drives routing in function.py — no hard-coded keyword matching.

Intent types:
  metadata      → answered directly from the document registry / Qdrant.
                  No LLM reasoning, no ReAct loop. Sub-100ms responses.
  tool          → a specific named tool is applicable. The tool name is
                  injected as a hint so the ReActAgent converges in 1 iteration.
  retrieval     → needs semantic search over resume/job content.
                  Routes to the full ReActAgent.
  conversational → general career advice or open discussion.
                  Routes to the full ReActAgent with memory.

Classification uses the already-loaded LLM (llama3.1:8b via Ollama).
The prompt is short and forces JSON output — adds ~1-2 seconds overhead,
which is recovered on metadata/tool routes that skip the ReAct loop.

Metadata answers (after routing) are produced by handle_metadata_query()
using Qdrant directly — completely deterministic, no LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

VALID_TOOL_NAMES = {
    "fit_score",
    "skill_gap_analysis",
    "analyze_fit",
    "job_ranking_based_on_fit",
    "interview_preparation_strategy",
    "resume_summary",
    "list_jobs",
}


class IntentClassification(BaseModel):
    """Structured output of the intent classifier.

    Returned as JSON by the LLM and parsed into this model.
    Used by function.py to decide the routing path.
    """
    intent: Literal["metadata", "tool", "retrieval", "conversational"]
    requires_retrieval: bool
    requires_metadata: bool
    requires_tool: bool
    tool_name: Optional[str] = None  # one of VALID_TOOL_NAMES, or null


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are a routing classifier for a career intelligence assistant.
Classify the user query into EXACTLY ONE of the four intents below.

INTENT DEFINITIONS:
- "metadata": user only asks what documents are uploaded — no analysis needed.
  Examples: "how many jobs uploaded?", "list the jobs", "is my resume uploaded?"
- "tool": user asks for analysis that maps directly to one specific tool.
  Examples: "describe my resume", "summarise my CV", "tell me about my background",
  "what is my fit score?", "show skill gaps", "rank all jobs", "which job fits me best?",
  "compare jobs", "best job", "prepare me for interview", "how well do I fit?"
- "retrieval": user asks a question requiring search over resume/job content but no specific tool.
  Examples: "what does this job require?", "what experience do I have in NLP?"
- "conversational": greetings, thanks, vague open questions, follow-up chat.
  Examples: "hello", "thanks", "what should I do?", "can you help me?"

TOOL NAMES — use ONLY these exact strings, or null:
- resume_summary: ANY question about resume content — "describe my CV", "summarise my resume",
  "tell me about my background", "what skills do I have?", "walk me through my CV",
  "can you see what I have done?", "talk me through my resume"
- job_ranking_based_on_fit: ranking or comparing ALL jobs — "which job fits me most?",
  "which job is best?", "rank all jobs", "compare jobs", "best job for me",
  "which job do I fit more?", "which job am I most suitable for?"
- fit_score: fit score for ONE specific job — "what is my score for job X?"
- analyze_fit: deep fit analysis for ONE specific job — "analyse my fit for job X"
- skill_gap_analysis: skill gaps for ONE specific job — "what am I missing for job X?"
- interview_preparation_strategy: interview prep for a job
- list_jobs: list all uploaded jobs (metadata path)

CRITICAL RULES:
1. "describe/summarise/explain/walk me through my CV/resume" → intent="tool", tool_name="resume_summary"
2. "which job fits me most/best/more", "best job", "rank jobs", "compare jobs" → intent="tool", tool_name="job_ranking_based_on_fit"
3. NEVER return a composite like "metadata|retrieval". Return exactly ONE of: metadata, tool, retrieval, conversational.
4. If unsure between retrieval and tool, prefer "tool".

User query: "{query}"

Respond with ONLY valid JSON matching this exact schema, no explanation, no markdown:
{{
  "intent": "metadata|tool|retrieval|conversational",
  "requires_retrieval": true_or_false,
  "requires_metadata": true_or_false,
  "requires_tool": true_or_false,
  "tool_name": "one_of_the_tool_names_above_or_null"
}}"""

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_FALLBACK = IntentClassification(
    intent="conversational",
    requires_retrieval=True,
    requires_metadata=False,
    requires_tool=False,
    tool_name=None,
)


def classify_intent(query: str, llm) -> IntentClassification:
    """Classify the user query using the loaded LLM.

    Returns a safe fallback (conversational) on any parse error so the
    system always degrades gracefully to the full ReActAgent.

    Args:
        query: The raw user query string.
        llm:   The already-loaded LlamaIndex LLM (Ollama llama3.1:8b).

    Returns:
        IntentClassification with intent and routing flags.
    """
    prompt = _CLASSIFICATION_PROMPT.format(query=query.replace('"', "'"))
    try:
        response = llm.complete(prompt)
        text = response.text.strip()

        # Extract first JSON object from the response (model may add preamble)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("Intent classifier returned no JSON — falling back. Raw: %s", text[:200])
            return _FALLBACK

        data = json.loads(text[start:end])

        # Validate tool_name is one of the known tools
        tool_name = data.get("tool_name")
        if tool_name and tool_name not in VALID_TOOL_NAMES:
            logger.warning("Unknown tool_name '%s' from classifier — clearing it", tool_name)
            data["tool_name"] = None

        classification = IntentClassification(**data)
        logger.info(
            "Intent classified: intent=%s tool=%s requires_retrieval=%s requires_metadata=%s",
            classification.intent,
            classification.tool_name,
            classification.requires_retrieval,
            classification.requires_metadata,
        )
        return classification

    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        logger.warning("Intent classification failed (%s) — falling back to conversational", exc)
        return _FALLBACK


# ---------------------------------------------------------------------------
# Metadata answer generator (deterministic — no LLM)
# ---------------------------------------------------------------------------

def handle_metadata_query(query: str, qdrant_reader) -> str:
    """Answer metadata queries directly from Qdrant — zero LLM involvement.

    Called only when the intent classifier routes to "metadata".
    Returns a human-readable markdown string.

    Args:
        query:         The original user query.
        qdrant_reader: QdrantReader instance for collection introspection.

    Returns:
        A formatted string response ready to return to the user.
    """
    job_ids = qdrant_reader.list_job_ids()
    resume_exists = qdrant_reader.collection_exists("resume_chunks")
    query_lower = query.lower()

    # --- Job listing / count queries ---
    if any(kw in query_lower for kw in ("list job", "show job", "what job", "which job", "uploaded job")):
        if not job_ids:
            return "No job descriptions have been uploaded yet."
        lines = []
        for jid in job_ids:
            title = qdrant_reader.get_first_line(f"job_{jid}") or f"Job {jid}"
            lines.append(f"• **{title}** (id: `{jid}`)")
        header = f"**{len(job_ids)} job description(s) uploaded:**"
        return header + "\n" + "\n".join(lines)

    # --- Count queries ---
    if any(kw in query_lower for kw in ("how many", "count", "number of")):
        parts = []
        parts.append(f"**{len(job_ids)}** job description(s) uploaded")
        if resume_exists:
            parts.append("**1** resume uploaded (active)")
        else:
            parts.append("**No** resume uploaded yet")
        return "\n".join(f"• {p}" for p in parts)

    # --- Resume status queries ---
    if "resume" in query_lower and any(kw in query_lower for kw in ("which", "active", "current", "uploaded")):
        if resume_exists:
            return "A resume is currently uploaded and active."
        return "No resume has been uploaded yet. Please upload a resume to begin analysis."

    # --- Generic document summary ---
    lines = ["**Uploaded documents:**"]
    lines.append(f"• Resume: {'uploaded (active)' if resume_exists else 'not uploaded'}")
    lines.append(f"• Job descriptions: **{len(job_ids)}**")
    if job_ids:
        for jid in job_ids:
            title = qdrant_reader.get_first_line(f"job_{jid}") or f"Job {jid}"
            lines.append(f"  — {title} (id: `{jid}`)")
    return "\n".join(lines)
