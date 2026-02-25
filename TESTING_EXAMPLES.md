# Testing Examples

Manual test queries organized by tool and routing path. Upload a resume and a few job descriptions before starting.

---

## Metadata (fast-path, no LLM)

- "Is my resume uploaded?"
- "List all the jobs you can see"

These should resolve in ~50 ms — answered directly from Qdrant, no LLM involved. If they take seconds, something is routing through the agent unnecessarily.

---

## Resume Reading (`resume_summary`)

- "Describe my CV"
- "What are my strongest technical skills?"
- "What is my educational background?"
- "Summarise my work experience"

---

## Fit Scoring — Single Job (`fit_score`)

Pick any `job_id` from the list (e.g. `9097f068`):

- "What is my fit score for job 9097f068?"
- "How well do I match the AgriSight role?" — tests whether the agent resolves a name to an id
- "Rate my match for the CivicDocs job"

---

## Skill Gap Analysis (`skill_gap_analysis`)

- "What skills am I missing for the HelioGrid job?"
- "Show me the gaps for job 7a10f189"
- "What does the Orbit Logistics role require that I don't have?"

---

## Job Ranking (`job_ranking_based_on_fit`)

- "Which job fits me the most?"
- "Rank all jobs by how well I match them"
- "Which job should I apply to first?"
- "Compare all the jobs against my profile"

---

## Deep Fit Analysis (`analyze_fit`)

- "Give me a full fit analysis for the HelioGrid job"
- "Analyse how well I fit the CivicDocs AI Engineer role"

---

## Interview Prep (`interview_preparation_strategy`)

- "Prepare me for the Orbit Logistics interview"
- "What technical questions should I expect for job 9097f068?"
- "Give me an interview strategy for the AgriSight role"

---

## Conversational / Multi-Turn Memory

- "hello" → then "so what can you help me with?"
- Ask a fit score → then "why is that score low?" — tests memory of previous answer
- "What job should I apply to first and why?"

These test the conversational fast-path and that `ChatMemoryBuffer` retains prior context across turns.

---

## Stress / Edge Cases

- "Which job am I least suitable for?" — reverse ranking
- "Do I have any skills none of the jobs ask for?" — bonus skills
- "What should I learn to improve my chances?" — crosses tools + reasoning
- "What is the capital of France?" — tests graceful deflection of off-topic queries
