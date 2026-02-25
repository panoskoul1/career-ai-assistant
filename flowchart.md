# System Flowcharts

## 1. System Architecture — Service Interaction

```mermaid
flowchart TD
    User([User]) --> Frontend

    subgraph Frontend["Frontend — React/Vite :3000"]
        ChatUI[Chat Interface]
        UploadUI[Resume & Job Upload]
    end

    UploadUI -->|"file upload (PDF/TXT)"| UploadRouter
    ChatUI -->|"chat message + session_id"| ChatRouter

    subgraph Backend["Backend — FastAPI :8000 (zero ML deps)"]
        UploadRouter["Upload Router<br/>PDF extraction to plain text"]
        ChatRouter["Chat Router<br/>proxy to fn-agent"]
        DocRegistry[("Document Registry<br/>JSON / upload_data vol")]
        UploadRouter --> DocRegistry
    end

    UploadRouter -->|"POST /ingest extracted text"| FnIngest
    ChatRouter -->|"POST /agent {query, session_id, job_id?}"| FnAgent

    subgraph FnIngest["fn-ingest :9090"]
        Chunk["Sentence-boundary chunking<br/>~800 tokens, 150 overlap"]
        Embed["Embed chunks<br/>BAAI/bge-small-en-v1.5"]
        Upsert["Upsert — one collection per document"]
        Chunk --> Embed --> Upsert
    end

    subgraph FnAgent["fn-agent :9091"]
        AgentEntry["Intent classifier<br/>3-path router + ReActAgent"]
    end

    Upsert --> Qdrant
    AgentEntry <-->|"vector queries"| Qdrant
    AgentEntry <-->|"LLM inference"| Ollama
    FnAgent -->|"response + intent + routed_via"| ChatRouter
    ChatRouter --> ChatUI

    Qdrant[("Qdrant :6333<br/>per-document collections")]
    Ollama[("Ollama :11434<br/>llama3.1:8b")]

    style Frontend fill:#e8f4fd,stroke:#1a73e8
    style Backend fill:#fef3e0,stroke:#e8a317
    style FnIngest fill:#e8f5e9,stroke:#2e7d32
    style FnAgent fill:#fce4ec,stroke:#c62828
    style Qdrant fill:#f3e5f5,stroke:#7b1fa2
    style Ollama fill:#f3e5f5,stroke:#7b1fa2
    style DocRegistry fill:#fff9c4,stroke:#f9a825
```

---

## 2. AI Core — fn-agent Internal Flow

```mermaid
flowchart TD
    Request["Incoming request<br/>{query, session_id, job_id?}"]

    Request --> Classify

    subgraph Classification["Intent Classification (~1-2s)"]
        Classify["LLM classifies query<br/>constrained JSON prompt"]
        Classify --> IntentResult{{"intent:<br/>metadata | tool | retrieval | conversational<br/>+ tool_name hint (nullable)"}}
    end

    IntentResult -->|"classification failed"| Fallback["Fallback: treat as<br/>conversational → full agent"]
    Fallback --> HintBuild

    IntentResult -->|"requires_metadata = true"| MetaPath

    subgraph MetaFastPath["Metadata Fast-Path (~50ms)"]
        MetaPath["handle_metadata_query()<br/>direct Qdrant scan<br/>no LLM, no ReAct"]
    end

    MetaPath -->|"success"| ResponseOut
    MetaPath -->|"handler failed"| ConvCheck

    IntentResult -->|"intent = conversational<br/>requires_tool = false"| ConvCheck

    subgraph ConvFastPath["Conversational Fast-Path"]
        ConvCheck["Direct llm.chat()<br/>system prompt + user query<br/>bypasses ReAct loop"]
    end

    ConvCheck -->|"success"| ResponseOut
    ConvCheck -->|"LLM call failed"| HintBuild

    IntentResult -->|"intent = tool / retrieval"| HintBuild

    subgraph QueryBuild["Build Effective Query"]
        HintBuild["Prepend routing hints"]
        HintBuild --> JobHint["[Selected job_id: X]<br/>if job_id provided"]
        HintBuild --> ToolHint["[USE_TOOL: tool_name]<br/>if classifier identified a tool"]
        JobHint --> EffectiveQuery["Effective query string"]
        ToolHint --> EffectiveQuery
    end

    EffectiveQuery --> AgentLoop

    subgraph ReactLoop["ReActAgent Loop"]
        AgentLoop["Get or create per-session agent<br/>ChatMemoryBuffer (2048 tokens)"]
        AgentLoop --> Reason["Reason: read query +<br/>memory + tool descriptions"]
        Reason --> ToolDecision{"Select tool<br/>(hint reduces iterations<br/>from 3-5 → 1)"}
        ToolDecision --> ToolExec["Execute tool"]
        ToolExec --> ToolOutput["Tool returns structured result"]
        ToolOutput --> Sufficient{Sufficient to answer?}
        Sufficient -->|"no — needs more info"| Reason
        Sufficient -->|"yes"| Synthesize["LLM synthesizes<br/>narrative from tool output"]
    end

    Synthesize --> ResponseOut
    ResponseOut(["Response to user<br/>{answer, intent, routed_via}"])

    style Classification fill:#fff3e0,stroke:#e65100
    style MetaFastPath fill:#e8f5e9,stroke:#2e7d32
    style ConvFastPath fill:#e3f2fd,stroke:#1565c0
    style QueryBuild fill:#fce4ec,stroke:#c62828
    style ReactLoop fill:#f3e5f5,stroke:#6a1b9a
    style ResponseOut fill:#c8e6c9,stroke:#2e7d32
```

---

## 3. ReActAgent — Tool Execution Detail

```mermaid
flowchart LR
    Agent["ReActAgent"] --> ToolSelect

    subgraph Tools["7 FunctionTools"]
        T1["list_jobs<br/>collection scan"]
        T2["resume_summary<br/>vector + LLM narrative"]
        T3["fit_score<br/>deterministic"]
        T4["skill_gap_analysis<br/>deterministic"]
        T5["job_ranking_based_on_fit<br/>deterministic + LLM summary"]
        T6["analyze_fit<br/>deterministic + LLM narrative"]
        T7["interview_preparation_strategy<br/>deterministic + LLM questions"]
    end

    ToolSelect{"Tool selected<br/>by agent"} --> Tools

    subgraph Deterministic["Deterministic Scoring Layer"]
        SkillEx["skill_extractor<br/>regex, ~150-skill vocabulary"]
        FitScore["fit_scorer<br/>score = |resume ∩ job| / |job|"]
        Matched["matched = resume ∩ job"]
        Missing["missing = job - resume"]
        Bonus["bonus = resume - job"]
        SkillEx --> FitScore
        SkillEx --> Matched
        SkillEx --> Missing
        SkillEx --> Bonus
    end

    T3 --> Deterministic
    T4 --> Deterministic
    T5 --> Deterministic
    T6 --> Deterministic
    T7 --> Deterministic

    T1 -->|"collection scan"| Qdrant
    T2 -->|"vector retrieval"| Qdrant
    T5 -->|"vector retrieval"| Qdrant
    T6 -->|"vector retrieval"| Qdrant
    T7 -->|"vector retrieval"| Qdrant

    T2 -->|"LLM narrative"| Ollama
    T5 -->|"LLM narrative"| Ollama
    T6 -->|"LLM narrative"| Ollama
    T7 -->|"LLM narrative"| Ollama

    Deterministic -->|"pre-computed numbers fed to LLM"| Ollama

    Qdrant[("Qdrant")]
    Ollama[("Ollama")]

    style Tools fill:#e8f5e9,stroke:#2e7d32
    style Deterministic fill:#fff3e0,stroke:#e65100
    style Qdrant fill:#f3e5f5,stroke:#7b1fa2
    style Ollama fill:#f3e5f5,stroke:#7b1fa2
```

**Key:** The LLM never computes scores or skill matches. Every number flows through the deterministic layer first. The LLM receives pre-computed results and writes narrative around them.
