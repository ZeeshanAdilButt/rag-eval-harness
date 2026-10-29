# Requirements

Phase 1 output. Every requirement has an ID so the design, the tests and the final review can point at it.

## Explicit requirements (from the brief)

### Inputs

| ID | Requirement |
|---|---|
| IN-1 | Read `kb.json` (`doc_id`, `title`, `text`). |
| IN-2 | Read `queries.json` (`query_id`, `user_question`, `expected_doc_ids`, `must_include_any`, `must_not_claim`, `risk_level`). |
| IN-3 | Read `candidate_answers.json` (`query_id`, `answers: {variant: text}`). |
| IN-4 | Ship the sample data: 5 KB docs (D1 password reset, D2 withdrawal review, D3 address verification, D4 demo accounts, D5 Responsible AI policy), 4 queries (Q1 to Q4), answers from `prompt_a` and `prompt_b`. |
| IN-5 | Graders may replace the inputs with equivalent fixtures. Never hardcode query IDs, phrases or answer text. |

### Must

| ID | Requirement |
|---|---|
| M1.1 | Deterministic retrieval (BM25, TF-IDF or token overlap). No web calls. |
| M1.2 | Top 2 passages per query, with scores, packaged as evidence. |
| M1.3 | Write `retrieval.json`: `{query_id, retrieved: [{doc_id, score, title, text}]}`. |
| M2.1 | Rule checks per query and variant, written to `automated_scores.json`: `{query_id, variant, retrieval_hit, must_include_pass, must_not_claim_pass, grounding_score, risk_flags, notes}`. |
| M2.2 | `retrieval_hit`: an expected doc is in the top 2. |
| M2.3 | `must_include_pass`: at least one `must_include_any` phrase is present. |
| M2.4 | `must_not_claim_pass`: no `must_not_claim` phrase is claimed. |
| M2.5 | `grounding_score`: a defined quote or token-overlap heuristic. |
| M2.6 | `risk_flags`: `["none"]` or hallucination and policy-risk flags. |
| M3.1 | Exactly one LLM call across all query-answer pairs. |
| M3.2 | A bounded comparative review: clarity, faithfulness to the evidence, overcommitting, a winner per query. |
| M3.3 | The prompt carries the evidence and the rule-check results. |
| M3.4 | A structured schema, enforced and validated in code. |
| M3.5 | The LLM never retrieves and never makes the deployment decision. |
| M3.6 | Write `llm_review.json`: `{query_id, winner, faithfulness: {variant: 1-5}, clarity: {...}, overclaim_flags: {...}, justification}`. |
| M3.7 | Log every call to `llm_calls.jsonl`. |
| M4.1 | Deterministic aggregation into `recommendation.md`. |
| M4.2 | It contains the selected variant, a summary table of wins and failures, the top reasons, per-query reasoning, the clarity versus safety tradeoff, and the known limitations. |
| M4.3 | High-risk failures are treated seriously. |
| M5.1 | A validation command (`python validate.py`). |
| M5.2 | It checks: artifacts exist, JSON is valid, every query was processed for every variant, retrieval has the top 2 per query, the LLM review uses only allowed variants and score ranges, the recommendation is reproducible from stored outputs. |

### Should

| ID | Requirement |
|---|---|
| S1 | `failure_taxonomy.json`, a controlled vocabulary (`unsupported_claim`, `missed_key_fact`, `policy_violation`, `retrieval_miss`, `overconfident_tone`, `irrelevant_answer`). |
| S2 | Tests: banned-claim detection, grounding behaviour, aggregation preferring safer answers on high-risk questions. |

### Stretch

| ID | Requirement |
|---|---|
| X1 | `review_report.md` per query: question, retrieved docs and scores, both answers, failed checks, LLM winner, final pick. |
| X2 | Adding a third variant needs no core pipeline change. |

### Constraints

| ID | Requirement |
|---|---|
| C1 | Any LLM provider. |
| C2 | With no API key the deterministic stages run, and the LLM stage fails gracefully or uses a stub. |
| C3 | Do not fabricate policy conclusions. |
| C4 | Bounded and quick to run. |
| C5 | `python run.py` regenerates everything. A `README.md` with setup. |

## Implicit expectations (what the graders will judge)

| ID | Expectation | How we will show it |
|---|---|---|
| I1 | Generalisation | Variant names and query IDs come from the data. A test runs a third variant with renamed query IDs. |
| I2 | Clear input errors | Missing keys, empty answers, unknown `doc_id`, a query with no answers, duplicate IDs: each fails with a message naming the file and the record. |
| I3 | Text normalisation | Case, punctuation, curly quotes, contractions, thousands separators. |
| I4 | Determinism | Ties broken by `doc_id`, fixed rounding, sorted keys, no timestamps in derived artifacts. Two runs produce byte-identical outputs. |
| I5 | Replay | LLM replies cached by prompt hash in `llm_calls.jsonl`; a rerun replays without calling the API. Temperature 0, pinned model. |
| I6 | Honest heuristics | Negation-aware banned-claim matching, stopword-filtered grounding, refusals not punished as ungrounded, and a README section on what the heuristics miss. |
| I7 | Safe aggregation | A banned claim, or any risk flag on a high-risk query, is a hard veto. Weighted scores only break ties. Weights live in a documented config. |
| I8 | LLM robustness | Tool use or JSON schema mode, pydantic validation, one bounded retry with the error, then a labelled stub. Every query and variant checked, scores in 1 to 5. |
| I9 | Stub is visible | A stub review is labelled in `llm_review.json`, `llm_calls.jsonl` and the recommendation, and it carries no weight in the decision. |
| I10 | Engineering polish | Small package behind a CLI, pinned requirements, Python 3.11+, type hints, pytest, GitHub Actions (run with no key, validate, test), small commits. |
| I11 | README | Quick start in under 5 commands, architecture, every heuristic and threshold with its reason, tradeoffs, limitations, adding a variant, swapping the LLM provider. |
| I12 | Small and readable | A reviewer understands it in 15 minutes. Standard library plus pydantic. No notebooks. |
| I13 | No secrets | Keys only from the environment, never logged or committed. |

## Open points

- The brief described the sample data but its exact text did not reach this build. The fixtures in `data/` follow the description (doc topics, query IDs, two variants) and were written so that each check has at least one case that passes and one that fails. Nothing in the code depends on their contents.
- No API key is available in the build environment, so the committed artifacts come from the stub path. The README says so.
