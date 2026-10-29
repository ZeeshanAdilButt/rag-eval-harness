# Build phases

The harness was built in seven gated phases. Each phase ends in a commit and a note here. A phase only closes when its gate passes: a critic pass for documents, a real run for code.

## Phase 1: requirements

- Read the brief and split it into explicit requirements (IN, M, S, X, C) and implicit expectations (I), each with an ID, in `docs/requirements.md`.
- Wrote the sample fixtures in `data/`. Each check has at least one passing and one failing case: Q1 and Q3 contain negated banned phrases that must not trip the detector, Q2 `prompt_b` overpromises on a high-risk question, Q4 `prompt_a` is safe but misses the key fact.
- Gate: every line of the brief maps to an ID.

## Phase 2: design

- Wrote `docs/design.md`: architecture, normalisation, every scoring rule with its threshold and reason, the LLM contract, aggregation, validation and tradeoffs.
- Mid-build the scope grew: each stage runs as a pluggable agent that claims its task, every run is stored in Postgres (SQLite when no `DATABASE_URL`), and configuration is layered like ampwise (defaults, TOML, profile, env, `--set`). Added sections for agents, storage and configuration.
- Borrowed from the AI engineering course: a versioned prompt file, blind labels in shuffled order for the judge, "length is not a criterion" in the rubric, and per-attempt request logging with tokens, latency and finish reason.
- Gate: a blind critic reviewed the requirements, design and config without the code. Findings and fixes are under phase 7.

## Phase 3: deterministic stages

- `text.py` (normalisation, stemming, clauses), `schema.py` (input models and clear errors), `config.py` (validated layers), `retrieval.py` (BM25 with doc_id tie-break) and `checks.py` (include, negation-aware banned claims, grounding with refusal handling and number checks, certainty markers, relevance).
- Gate: tests for banned claims and negation, paraphrase limits, grounding, refusals, retrieval ties and every input error case.

## Phase 4: LLM stage

- `llm.py`: one request for every query, schema built from the data, forced tool use (Anthropic) or strict `json_schema` (OpenAI-compatible), pydantic plus semantic validation, one retry with the errors, then a labelled `stub_fallback`.
- Replay cache keyed by the SHA-256 of the full request; `--offline` replays only. Transport retries only on timeouts, 429 and 5xx. Keys come from the environment and never reach the log.
- Gate: tests with a fake provider for retry, fallback, no retry on auth errors, replay without a call, and the no-key stub.

## Phase 5: aggregation and reports

- `aggregate.py`: veto by risk level, then failure tier, then the weighted tie-break; overall ranking by high-risk vetoes first. A stub review gets zero weight. A "ready to ship" line only when no configured blocker fires.
- `report.py`: `recommendation.md` (selection, summary table, top reasons, per-query reasoning, clarity versus safety, limitations) and `review_report.md` per query.
- Gate: tests that a clearer answer with a high-risk violation loses, that unsupported claims veto only on high risk, all-vetoed queries, zero stub weight and retrieval misses.

## Phase 6a: agents and storage

- `agents.py`: a registry of stage agents with declared inputs and outputs; the stage list lives in config.
- `pipeline.py`: one claimed task per stage, a conditional `UPDATE` claim that can be taken over when stale, and a terminal state for every run.
- `storage.py`: one schema rendered for Postgres and SQLite, 12 tables. `docker compose up -d` starts Postgres on port 5446.
- Gate: a real run against Postgres 17 filled every table; tests cover claim exclusivity and stale takeover.
