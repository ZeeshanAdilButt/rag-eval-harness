# Architecture at a glance

## The flow

```mermaid
flowchart TD
    subgraph IN["Inputs (data/)"]
        KB[kb.json]
        Q[queries.json]
        CA[candidate_answers.json]
    end

    subgraph CFG["Configuration"]
        TOML["config/eval.toml<br/>defaults, profiles, env, --set"]
        TAX[config/failure_taxonomy.json]
        PROMPT[prompts/llm_review.toml]
    end

    RUN(["python run.py"]) --> LOAD["load + validate inputs<br/>schema.py"]
    IN --> LOAD
    TOML --> ORCH

    LOAD --> ORCH["orchestrator<br/>pipeline.py<br/>one claimed task per stage"]

    ORCH --> A1["1. bm25_retriever<br/>retrieval.py"]
    A1 -->|top 2 per query| A2["2. rule_checker<br/>checks.py"]
    A2 -->|scores + flags| A3["3. llm_reviewer<br/>llm.py"]
    A3 -->|reviews| A4["4. safety_first_aggregator<br/>aggregate.py"]
    A4 -->|decision| A5["5. markdown_reporter<br/>report.py"]

    PROMPT --> A3
    TAX --> A2

    A3 --> CACHE{"same prompt hash<br/>in llm_calls.jsonl?"}
    CACHE -->|yes| REPLAY[replay, no API call]
    CACHE -->|no, key set| LIVE["ONE live call<br/>schema enforced + validated<br/>one retry if invalid"]
    CACHE -->|no key| STUB["labelled stub<br/>zero weight"]

    A1 --> R1[outputs/retrieval.json]
    A2 --> R2[outputs/automated_scores.json]
    A3 --> R3[outputs/llm_review.json]
    A3 --> R4[outputs/llm_calls.jsonl]
    A5 --> R5[outputs/recommendation.md]
    A5 --> R6[outputs/review_report.md]

    ORCH -.every stage writes rows.-> DB[("run database<br/>Postgres if DATABASE_URL<br/>else SQLite")]

    VAL(["python validate.py<br/>validate.py"]) -.rebuilds and compares.-> R1 & R2 & R3 & R5 & R6
```

### How one answer gets judged

```mermaid
flowchart LR
    ANS[answer] --> INC{"must_include<br/>phrase present?"}
    ANS --> BAN{"must_not_claim<br/>claimed, not negated?"}
    ANS --> GR{"grounded in<br/>evidence?"}
    ANS --> TONE{"certainty marker<br/>not negated?"}
    INC -->|no| F1[missed_key_fact]
    BAN -->|yes| F2[policy_violation]
    GR -->|no| F3[unsupported_claim]
    TONE -->|yes| F4[overconfident_tone]
    F2 & F3 & F4 --> VETO{"veto for this<br/>risk level?"}
    VETO -->|yes| OUT[removed from the query]
    VETO -->|no| TIER[tier 2: risk flag]
    F1 --> TIER1[tier 1: incomplete]
    TIER & TIER1 --> TB["weights break ties<br/>only inside a tier"]
```

## Read the code in this order

About 15 minutes end to end.

| # | File | What it does |
|---|---|---|
| 1 | `README.md` | what the harness is for, the quick start, every rule and its threshold |
| 2 | `config/eval.toml` | every knob in one place: thresholds, weights, vetoes, model, storage, the stage list |
| 3 | `data/*.json` | the inputs: KB docs, queries with their must and must-not phrases, the answers for each variant |
| 4 | `rag_eval/pipeline.py` | the entry point: load inputs, open storage, create and claim one task per stage, run the agents |
| 5 | `rag_eval/agents.py` | the five stage agents: what each reads, writes and stores |
| 6 | `rag_eval/text.py` | the one normalisation everything shares: quotes, contractions, stemming, clauses |
| 7 | `rag_eval/retrieval.py` | BM25, top k, ties broken by `doc_id` |
| 8 | `rag_eval/checks.py` | the rule checks: include, banned claims with negation, grounding, numbers, tone |
| 9 | `rag_eval/llm.py` | the single review call: prompt, blind labels, schema, providers, replay, retry, stub |
| 10 | `prompts/llm_review.toml` | the judge rubric (versioned) |
| 11 | `rag_eval/aggregate.py` | the decision: vetoes, tiers, tie-break weights, ship blockers |
| 12 | `rag_eval/report.py` | renders `recommendation.md` and `review_report.md` |
| 13 | `rag_eval/validate.py` | rebuilds every deterministic output and compares it with what is on disk |
| 14 | `rag_eval/storage.py` | the run database: 12 tables, one schema for Postgres and SQLite, the task claim |
| 15 | `rag_eval/config.py`, `rag_eval/schema.py` | how config layers are merged and validated, and how input errors are reported |
| 16 | `tests/` | start with `test_pipeline.py` (end to end, third variant, tampering), then the unit tests |
| 17 | `outputs/` | what a run produced on the sample data |
| 18 | `docs/requirements.md`, `docs/design.md`, `docs/phases.md` | the brief as IDs, the design with tradeoffs, and how it was built |
