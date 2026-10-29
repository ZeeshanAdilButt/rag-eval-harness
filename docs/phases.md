# Build phases

The harness was built in gated phases. Each phase ended in a commit, and a phase only closed when its gate passed: a review for documents, a real run for code.

## 1. Requirements

The brief was split into explicit requirements and implicit expectations, each with an ID, in `docs/requirements.md`. Gate: every line of the brief maps to an ID.

## 2. Design

`docs/design.md` covers the architecture, every scoring rule with its threshold and reason, the LLM contract, the aggregation, validation and tradeoffs. Later sections cover the stage agents, the run database and the layered configuration. Gate: an independent design review of the requirements, design and config, without the code. It found 12 defects, 3 of them severe, all fixed in phase 7.

## 3. Deterministic stages

Text normalisation, input validation, layered config, BM25 retrieval and the rule checks, with tests for negation, paraphrase limits, grounding, refusals, retrieval ties and every input error. Gate: tests pass.

## 4. LLM stage

One request for every query, a schema built from the data, enforcement at the provider and again in code, a replay cache keyed by the request hash, and a labelled stub with no key. Gate: tests with a fake provider for fallback, replay without a call, and the no-key stub.

## 5. Aggregation and reports

Vetoes by risk level, then failure tiers, then weighted tie-breaks. The two Markdown reports. Gate: tests that a clearer answer with a high-risk violation loses, and that a stub carries no weight.

## 6. Agents, storage, validation, CI and README

Each stage runs as an agent that claims its task. Runs are stored in Postgres or SQLite. `validate.py` rebuilds every deterministic output. CI runs the offline path on Python 3.11 and 3.12, plus a Postgres job. Gate: `python run.py && python validate.py && pytest` pass from a fresh clone.

## 7. Review and fixes

The design review's findings were fixed:
- risk flags rank below incomplete answers;
- negation is scoped to the words near the claim;
- grounding ignores passages with zero retrieval score;
- includes match on token boundaries;
- the prompt hash is checked for staleness.

A final review against the brief, from five angles (compliance, swapped fixtures, engineering, readability, LLM safety), led to:
- the brief's exact data in `data/`;
- the brief's record shapes;
- `failure_taxonomy.json` in `outputs/`;
- exactly one LLM call by default;
- no stored HTTP error bodies;
- a stub never reads as ready to ship;
- strict types in the reply model;
- escaped questions;
- a validator that reports instead of crashing.

Gate: tests pass, outputs are byte-identical across runs, the call log holds exactly one live call, and CI is green.
