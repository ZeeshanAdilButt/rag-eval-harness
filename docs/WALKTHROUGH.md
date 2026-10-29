# What happens, step by step

A support bot can answer customer questions in different styles, called prompt variants (`prompt_a`, `prompt_b`, ...). This project answers one question: **which variant is safer to ship, and why?** It does that without the internet, and it gives the same result every time.

Below is one run, from typing the command to reading the result. The file that does each step is in brackets.

---

## 0. You type one command

`python run.py` [run.py, rag_eval/cli.py]

The command reads the settings first [config/eval.toml]. Thresholds, weights, the model name and the order of steps all live in that file, so changing behaviour means changing config, not code. Your API key lives in a separate file that is never committed [config/local.toml].

## 1. Read and check the inputs

[rag_eval/schema.py]

Three files go in [data/]:

- **The knowledge base:** 5 help articles, such as "password reset" and "withdrawal review" [data/kb.json].
- **The questions:** each one says which article should answer it, which phrases a good answer must include, which claims it must never make, and how risky it is [data/queries.json].
- **The answers** each variant gave to each question [data/candidate_answers.json].

If anything is wrong, such as a missing field, an empty answer or an article ID that doesn't exist, the run stops and names the file and the record. The variant names are read from the data, so a third variant works without touching code.

## 2. Start a run and hand out the jobs

[rag_eval/pipeline.py, rag_eval/agents.py, rag_eval/storage.py]

The run gets an ID. Each step below is done by its own small **agent**. Before it starts, an agent has to **claim** its job in the database. Because of the claim, the same step can never run twice, and if a worker crashes its job can be picked up again. Everything the run sees and produces is saved to the database: Postgres if `DATABASE_URL` is set, a local SQLite file otherwise [outputs/runs.sqlite].

## 3. Find the evidence

Agent: `bm25_retriever` [rag_eval/retrieval.py, rag_eval/text.py]

For each question it scores every article with BM25, a classic keyword-matching formula, and keeps the **top 2**. Ties are broken by article ID, so the order never changes. Text is cleaned the same way everywhere: lowercase, straight quotes, "can't" becomes "can not", and "2,000" becomes "2000".

Output: [outputs/retrieval.json]

## 4. Check every answer with plain rules

Agent: `rule_checker` [rag_eval/checks.py]

Each answer from each variant goes through the same checklist:

- **Did we find the right article?** (`retrieval_hit`)
- **Does it say the key fact?** At least one "must include" phrase. (`must_include_pass`)
- **Does it make a banned claim?** Negation is handled, so "we **cannot** see your password" is fine and "we can see your password" is not. (`must_not_claim_pass`)
- **How much of it comes from the evidence?** The share of its meaningful words found in the retrieved articles. A polite "I don't know, let me connect you to a human" is not punished. (`grounding_score`)
- **Does it invent numbers** like "within 24 hours" that appear nowhere in the evidence?
- **Does it sound overconfident**, with words like "definitely" or "guaranteed"?

Every problem gets a code from a fixed list [config/failure_taxonomy.json], such as `policy_violation`, `unsupported_claim`, `missed_key_fact` or `overconfident_tone`.

Output: [outputs/automated_scores.json]

## 5. Ask an LLM for a second opinion, once

Agent: `llm_reviewer` [rag_eval/llm.py, prompts/llm_review.toml]

All questions go into **one** request. Each one carries the question, its evidence, every answer and the rule-check results. The answers are shown as "answer_1" and "answer_2" in a shuffled order, so the model can't favour a name or a position. The model scores **faithfulness** and **clarity** from 1 to 5, quotes any overclaims, and names a winner per question. It is told never to decide what ships.

The reply has to fit a strict schema, and the code checks it again. If it doesn't fit, the model gets one retry with the errors listed. If it still fails, or there is no API key, a clearly labelled **stub** stands in, and it gets zero weight in the decision.

Every attempt is logged [outputs/llm_calls.jsonl]. On the next run with the same inputs, the stored reply is **replayed** and no API call is made.

Output: [outputs/llm_review.json]

## 6. Make the decision in code

Agent: `safety_first_aggregator` [rag_eval/aggregate.py]

For each question:

1. **Veto.** A banned claim removes the answer. On a high-risk question, so does any risk flag.
2. **Tier.** Clean answers beat incomplete ones, and incomplete ones beat risky ones.
3. **Tie-break.** Only inside the same tier, a weighted score decides. The weights are set in [config/eval.toml].

Across all questions, the variant with the fewest high-risk vetoes wins. It is only called **ready to ship** if nothing blocks it, for example a missed key fact on a high-risk question or the LLM flagging an overclaim there. If a variant is clearer but less safe, the report says so plainly.

## 7. Write the reports

Agent: `markdown_reporter` [rag_eval/report.py]

- [outputs/recommendation.md]: the pick, a summary table, top reasons, reasoning for each question, the clarity versus safety tradeoff, and known limitations.
- [outputs/review_report.md]: one section per question with the evidence, each answer, the failed checks, the LLM's view and the final pick.

The run is marked `done` in the database.

## 8. Prove it

`python validate.py` [validate.py, rag_eval/validate.py]

It rebuilds everything that can be rebuilt from the saved files and compares it with what is on disk: retrieval, rule checks, the LLM prompt hash and both reports. It also checks that every question was scored for every variant, that scores are 1 to 5, and that the stub is labelled. Any mismatch is listed.

`pytest` [tests/] covers the tricky parts: negation, grounding, the safety vetoes, retry and replay, a third variant with renamed IDs, tampered files and the database claims.

CI runs all of this on every push with no API key [.github/workflows/ci.yml].

---

## What came out on the sample data

**`prompt_a` is selected but not ready to ship.**

- `prompt_b` reads friendlier, but it promised a withdrawal "will be approved within 24 hours" and told a demo user they "will definitely see similar results". Both were vetoed as high-risk.
- `prompt_a` was safe everywhere, but on the demo-account question it never said that demo money has no cash value. That gap blocks shipping until it's fixed.

## Where to look next

- The diagram and reading order: [ARCHITECTURE.md]
- Every rule, threshold and tradeoff: [README.md]
- The brief as numbered requirements: [docs/requirements.md]
- The design and its tradeoffs: [docs/design.md]
- How it was built, phase by phase: [docs/phases.md]
