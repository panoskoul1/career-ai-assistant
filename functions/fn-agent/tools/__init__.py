"""Tool registry — assembles all FunctionTool instances.

Tool names are aligned with the target specification:
  - fit_score                      (deterministic)
  - skill_gap_analysis             (deterministic)
  - job_ranking_based_on_fit       (deterministic scoring + LLM summary)
  - interview_preparation_strategy (deterministic gaps + LLM questions)
  - analyze_fit                    (deterministic scoring + LLM narrative)
  - resume_summary                 (deterministic skills + LLM narrative)
  - list_jobs                      (metadata — Qdrant collection scan)
"""

from __future__ import annotations

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from indexes.index_store import IndexStore
from services.qdrant_reader import QdrantReader

from tools.analyze_fit import make_analyze_fit_tool
from tools.compare_jobs import make_compare_jobs_tool
from tools.compute_fit_score import make_compute_fit_score_tool
from tools.interview_plan import make_interview_plan_tool
from tools.list_jobs import make_list_jobs_tool
from tools.list_skill_gaps import make_list_skill_gaps_tool
from tools.resume_summary import make_resume_summary_tool


def build_all_tools(
    index_store: IndexStore,
    qdrant_reader: QdrantReader,
    llm: LLM,
) -> list[FunctionTool]:
    """Return the full tool list for the career agent.

    Order matters: deterministic tools first so the ReActAgent considers
    them before heavier LLM-augmented tools.
    """
    return [
        make_list_jobs_tool(qdrant_reader),
        make_compute_fit_score_tool(index_store, qdrant_reader),
        make_list_skill_gaps_tool(index_store, qdrant_reader),
        make_analyze_fit_tool(index_store, qdrant_reader, llm),
        make_compare_jobs_tool(index_store, qdrant_reader, llm),
        make_interview_plan_tool(index_store, qdrant_reader, llm),
        make_resume_summary_tool(index_store, qdrant_reader, llm),
    ]
