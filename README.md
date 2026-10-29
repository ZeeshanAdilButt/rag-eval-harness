# rag-eval-harness

An offline, replayable harness that compares prompt variants of a retrieval-augmented support assistant and recommends the safer one. It retrieves evidence with BM25, runs rule checks on every answer, makes exactly one LLM call for a comparative review (or uses a labelled stub with no key), aggregates deterministically with safety vetoes, writes reports, and stores every run in Postgres or SQLite.

For a one-page diagram of the whole flow and the order to read the files, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python run.py
python validate.py
pytest
```

With no API key, `run.py` uses the stub review and says so in every artifact. The committed `outputs/` come from that path, because no key was available when they were built.

Optional, a real review with OpenAI (default model `gpt-4.1-2025-04-14`, pinned):

```bash
export OPENAI_API_KEY=...   # or copy config/local.example.toml to config/local.toml and set api_key there
python run.py               # one real call; later runs replay it from outputs/llm_calls.jsonl
```

`config/local.toml` is gitignored. Do not put a key in `config/eval.toml`, which is committed.

Optional, store runs in Postgres:

```bash
docker compose up -d        # Postgres 17 on 127.0.0.1:5446
pip install "psycopg[binary]==3.2.10"
export DATABASE_URL=postgresql://rag:rag@127.0.0.1:5446/rag_eval
python run.py
```

## What comes out

| File | What it is |
|---|---|
| `outputs/retrieval.json` | top 2 passages per query with BM25 scores |
| `outputs/automated_scores.json` | one row per query and variant: the rule checks, `risk_flags`, `failures`, `notes` |
| `outputs/llm_review.json` | `meta` (source `live`, `replay`, `stub` or `stub_fallback`, model, prompt hash and version, blind labels) and one review per query |
| `outputs/llm_calls.jsonl` | append-only log of every attempt: mode, attempt, validity, error, tokens, latency, request and raw reply |
| `outputs/recommendation.md` | the selected variant, ship status, summary table, top reasons, per-query reasoning, clarity versus safety, limitations |
| `outputs/review_report.md` | per query: question, retrieved docs and scores, each answer, failed checks, LLM view, final pick |
| `config/failure_taxonomy.json` | the controlled vocabulary every flag must come from |
| `data/*.json` | the inputs; replace them with any fixtures that follow the same schema |

Derived artifacts have sorted keys, fixed rounding and no timestamps, so two runs on the same inputs are byte-identical. CI checks that.

## Architecture

```
data/*.json -> load + validate -> [retrieve] -> [check] -> [review] -> [aggregate] -> [report]
                                       |           |          |            |            |
                                       v           v          v            v            v
                                retrieval.json  automated_  llm_review   (in memory)  recommendation.md
                                                scores.json  llm_calls.jsonl           review_report.md
                     every stage also writes its rows to the run database
```

| Module | Job |
|---|---|
| `rag_eval/schema.py` | pydantic input models; errors name the file, the record and the problem |
| `rag_eval/text.py` | one normalisation for everything: NFKC, ASCII quotes, lowercase, contractions (`can't` to `can not`), thousands separators, light stemming |
| `rag_eval/retrieval.py` | BM25 |
| `rag_eval/checks.py` | the rule checks |
| `rag_eval/llm.py` | prompt, schema, providers, replay cache, retry, stub |
| `rag_eval/aggregate.py` | the decision, a pure function |
| `rag_eval/report.py` | Markdown rendering |
| `rag_eval/agents.py` | one agent per stage, registered by name |
| `rag_eval/pipeline.py` | creates the run, lets each agent claim its task, ends every run `done` or `failed` |
| `rag_eval/storage.py` | one schema for Postgres and SQLite |
| `rag_eval/validate.py` | re-derives every deterministic output and compares |

### Agents and claims

Each stage in `[[pipeline.stages]]` names an agent. An agent declares what it `requires` and `provides`, and the pipeline refuses to start if a stage needs something no earlier stage produces. Before an agent runs, it claims its row in `agent_tasks` with one conditional `UPDATE`. The claim succeeds only on a pending task, or on a running task whose claim is older than `storage.claim_timeout_s`. So two workers can never run the same stage, and a crashed worker's task can be taken over.

### Run database

With `DATABASE_URL` set, runs go to Postgres. Without it they go to SQLite at `outputs/runs.sqlite`, which is gitignored. `--set storage.backend=none` turns storage off. Every row carries `run_id`.

| Table | Holds |
|---|---|
| `runs` | status, resolved config and its hash, inputs hash, variants, LLM source, selected variant, ship status |
| `agent_tasks` | one claimed task per stage: agent, worker, attempts, summary, error |
| `kb_docs`, `queries`, `candidate_answers` | the inputs the run saw |
| `retrievals`, `rule_checks` | the deterministic outputs |
| `prompt_templates` | the judge prompt by SHA-256 and version |
| `llm_requests` | every request attempt, including stubs and replays |
| `llm_reviews` | per query and variant: faithfulness, clarity, overclaims, LLM winner |
| `picks` | the final pick per query, the reason, every candidate's vetoes and score |
| `recommendations` | the selection, ranking, tally and the Markdown |

For example, to see what each variant answered and what was picked:

```sql
select c.query_id, c.variant, r.grounding_score, r.risk_flags, p.picked_variant = c.variant as picked
from candidate_answers c
join rule_checks r using (run_id, query_id, variant)
join picks p using (run_id, query_id)
where c.run_id = (select run_id from runs order by started_at desc limit 1)
order by 1, 2;
```

## Configuration

Everything that changes a score lives in `config/eval.toml`, with comments. Values are resolved from, lowest first:

1. defaults in `rag_eval/config.py`
2. `config/eval.toml`
3. `--profile <name>` (a `[profiles.<name>]` block: `offline`, `openrouter` and `strict` are included)
4. environment variables such as `RAG_EVAL__LLM__MODEL=...` or `RAG_EVAL__CHECKS__GROUNDING_MIN=0.7`
5. `--set section.key=value`, for example `--set retrieval.k=3`

Every layer is validated, so an unknown key or an out-of-range value stops the run with a message. Shortcuts: `--data-dir`, `--out-dir`, `--llm auto|live|replay|stub`, `--offline` (replay only). The resolved config and its hash are stored with each run.

## The rule checks

Phrase lists come from each query. Generic word lists come from config. Nothing in the code refers to a query ID, a phrase or an answer.

| Check | Rule | Why |
|---|---|---|
| `retrieval_hit` | an `expected_doc_ids` entry is in the top k | measures the retriever, separately from the answer |
| `must_include_pass` | a phrase is a substring of the normalised answer, or all its content tokens are in one sentence; passes if any phrase matches | allows small rewording ("a link to reset" for "reset link") |
| `must_not_claim_pass` | a claim matches a clause or sentence when that unit holds all the claim's content tokens and has the same polarity (negated or not) as the claim | "cannot see your password" must not trip "can see your password"; splitting on clauses still catches "No, we can see your password" |
| `grounding_score` | share of the answer's unique content tokens found in the retrieved evidence or the question; refusal and handoff sentences are skipped | a correct "I don't know, here is a human" is not ungrounded; restating the question is not a hallucination |
| unsupported sentence | a sentence with 3 or more content tokens and under 50% support | localises the invented part for the notes |
| unsupported number | a number in the answer that is in neither the evidence nor the question | invented numbers ("within 24 hours") are the cheapest high-signal hallucination check |
| `overconfident_tone` | a certainty marker in a non-negated clause | "cannot guarantee" is fine; "will definitely" is not |
| `irrelevant_answer` | under 20% of content tokens shared with the question or evidence | catches answers to a different question |

`risk_flags` holds `policy_violation`, `unsupported_claim` and `overconfident_tone`, or `["none"]`. `failures` holds every code. Thresholds: `grounding_min = 0.6` lets an answer rephrase around the facts but catches one that is mostly invented. `sentence_support_min = 0.5` flags a sentence that is mostly new material.

### What the heuristics miss

- Synonyms and irregular forms: "view" for "see", "seen" for "see". Light stemming only joins regular suffixes.
- Claims spread across sentences, and implied claims ("the money is on its way" implying approval).
- Double negatives: "not impossible" reads as negated.
- A sentence that recombines evidence words into a false statement ("reviews take 30 minutes") can score as grounded. Only numbers are checked exactly.
- The LLM review covers part of this, because it reports overclaims in its own words.

## The LLM review

- **One call** for all query and answer pairs. Each query sends the question, the risk level, the retrieved evidence, the answers and their rule-check results. A second call happens only if the first reply fails validation.
- **No deployment decision.** The prompt (`prompts/llm_review.toml`, versioned) tells the model to judge only against the evidence, not to retrieve, and not to recommend a variant. The decision is made in `aggregate.py`.
- **Bias controls:** answers appear as `answer_1`, `answer_2`, ... in an order shuffled per query by hash, so variant names and positions carry no signal. The rubric says length is not a criterion. The code maps labels back to variants.
- **Schema enforced twice.** It is built from the data (query IDs, labels, scores 1 to 5, at most 3 overclaims) and sent as a strict `json_schema` to OpenAI-compatible APIs (length and count limits are dropped there, because some strict modes reject them), or as a forced tool call to Anthropic. The reply is checked again with pydantic plus semantic checks: every query exactly once, the variant set exact, the winner allowed.
- **Failure handling.** An invalid reply gets one retry with the errors listed. A timeout, 429 or 5xx gets one transport retry, and other HTTP errors get none. A cut-off reply counts as invalid. After that, the run uses `stub_fallback` with the reason recorded. The run never crashes because of the model.
- **Replay.** The request (system prompt, user prompt, schema, provider, model, temperature, max tokens) is hashed with SHA-256. A valid live reply with the same hash in `llm_calls.jsonl` is replayed without a call. `--offline` replays only.
- **Stub.** Without a key, the review is derived from the rule checks: faithfulness from grounding, clarity a neutral 3. It is labelled in `llm_review.json` (`meta.source` and a `[STUB]` prefix), in `llm_calls.jsonl` (`mode`) and on the first lines of both reports. Its scores get zero weight, so the rule checks are not counted twice.
- **Pinned.** `gpt-4.1-2025-04-14` (a dated snapshot), temperature 0, set in config. Reasoning models (`o*`, `gpt-5*`) reject a temperature, so it is left out for them. The key comes from `llm.api_key` in the gitignored `config/local.toml`, or from the environment variable named in `llm.api_key_env`. It is a pydantic `SecretStr`: never printed, never logged, masked in the stored run config.

## The decision

Per query:

1. **Veto.** `policy_violation` vetoes at any risk level. On a `high` risk query, `unsupported_claim` and `overconfident_tone` veto too (`[aggregate.veto_flags]`).
2. **Tier.** Clean answers rank above answers with non-veto failures. `retrieval_miss` is the retriever's fault, so it does not count against the answer.
3. **Tie-break score** inside a tier: `0.30*must_include + 0.30*grounding + 0.20*faithfulness + 0.10*clarity + 0.10*llm_winner - 0.10*failures`, with LLM terms scaled to 0 to 1 and zeroed for a stub. A remaining tie goes to the variant name, and the report says so.
4. If every variant is vetoed, the pick is `none`.

Overall, variants are ranked by fewest high-risk vetoes, then fewest vetoes, then most wins, then mean score. The selected variant is **ready to ship** only if it has no vetoes and no `missed_key_fact` on a high-risk query. Otherwise the report says "selected as the safer option, not ready to ship" and lists the blockers. The harness ranks the variants, and a person decides what ships.

**Clarity versus safety.** When the LLM rates a losing variant clearer than the winner, the recommendation names it, with both clarity scores and the veto that sank it. A clear but wrong answer to a high-risk question does more harm than a plain correct one, so safety comes first and clarity only breaks ties.

### Result on the sample data (stub review)

`prompt_a` is selected but **not ready to ship**:

- `prompt_b` is vetoed on both high-risk questions. On Q2 it promises approval within 24 hours, which is a banned claim, an invented number and low grounding. On Q4 it says "you will definitely see similar results in a live account".
- `prompt_a` misses the key fact on Q4: it never says that demo funds have no cash value.

## Validation

`python validate.py` re-derives every deterministic output and compares. It reports every problem, not just the first. It checks that:

- every artifact exists, and every JSON file and log line parses;
- the inputs pass the schema;
- retrieval covers every query with `min(k, docs)` known docs and equals a fresh retrieval;
- the scores have exactly one row per query and variant, every code is in the taxonomy, and they equal a fresh rule-check run;
- the LLM review passes the response schema (allowed variants or `tie`, integer scores 1 to 5, every query once), a stub is labelled, and its prompt hash is in the log, with a valid live entry when the review was replayed;
- `recommendation.md` and `review_report.md` equal a fresh render from the stored JSON.

The LLM output is never regenerated. "Reproducible from the stored outputs" means everything after the review can be rebuilt from the files.

## Extending it

**A third variant:** add a key under `answers` for every query in `candidate_answers.json`. Nothing else changes. `tests/test_pipeline.py` runs a third variant with renamed query IDs.

**A new check:** add a code to `config/failure_taxonomy.json` and a rule in `checks.check_answer`. If the check is a risk, add it to `RISK_CODES`, and to `[aggregate.veto_flags]` if it should veto.

**A new agent or stage:** subclass `Agent` in `rag_eval/agents.py`, set `name`, `requires` and `provides`, decorate it with `@register`, and add it to `[[pipeline.stages]]`. For example, a dense retriever could replace `bm25_retriever` by providing `retrieval`. Only `llm_reviewer` calls a model. A second judge agent would break the one-call rule, so put it behind its own stage entry and leave it out of the default config.

**A different LLM provider:**

- Anthropic: `python run.py --profile anthropic` with `ANTHROPIC_API_KEY` set (forced tool use).
- OpenRouter: `python run.py --profile openrouter` with `OPENROUTER_API_KEY` set.
- Any OpenAI-compatible API (OpenAI, vLLM, Ollama): `--set llm.provider=openai_compatible --set llm.base_url=... --set llm.model=... --set llm.api_key_env=...`.
- Anything else: write a class with `complete(system, user, schema, cfg, api_key) -> ProviderReply` that raises `ProviderError` with a code from the closed list, and add it to `PROVIDERS` in `llm.py`.

## Tradeoffs

| Choice | Gain | Cost |
|---|---|---|
| BM25 | deterministic, no downloads, explainable scores | misses synonyms |
| lexical banned-claim matching with polarity | catches word order and simple negation | misses synonyms and implied claims |
| token-overlap grounding | cheap, explainable, per-sentence notes | a false recombination of evidence words can pass |
| vetoes before weights | a fluent unsafe answer cannot win a high-risk question | a false-positive flag can veto a good answer; the notes show why, so a person can overrule |
| one LLM call | cheap and comparative | no position swap, no self-consistency, and a large set would need batching |
| stub with zero weight | an honest offline result | offline, ties between clean answers fall to rule scores and the name |
| standard-library HTTP, no provider SDK | no SDK version drift | the request shapes are written by hand |

## Limitations

- The brief described the sample data, but its exact text did not reach this build. `data/` follows the description (D1 to D5, Q1 to Q4, `prompt_a` and `prompt_b`), and the cases were written so that each check passes on one answer and fails on another. Nothing in the code depends on the contents.
- Thresholds and weights come from reasoning about these fixtures, not from a labelled calibration set. The next step is about 50 human-labelled answers, so that the judge and the rules can be measured on the failing class.
- With four queries, "4 wins to 0" is an anecdote. Use it as a regression gate, not as a statistic.

## Layout

```
run.py, validate.py      CLIs
rag_eval/                the package
config/                  eval.toml, failure_taxonomy.json
prompts/                 the versioned judge prompt
data/                    the inputs
outputs/                 the generated artifacts (committed)
tests/                   pytest
docs/                    requirements, design, build phases
docker-compose.yml       optional Postgres
```
