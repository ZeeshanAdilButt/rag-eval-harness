# Build phases

The harness was built in seven gated phases. Each phase ends in a commit and a note here. A phase only closes when its gate passes: a critic pass for documents, a real run for code.

## Phase 1: requirements

- Read the brief and split it into explicit requirements (IN, M, S, X, C) and implicit expectations (I), each with an ID, in `docs/requirements.md`.
- Wrote the sample fixtures in `data/`. Each check has at least one passing and one failing case: Q1 and Q3 contain negated banned phrases that must not trip the detector, Q2 `prompt_b` overpromises on a high-risk question, Q4 `prompt_a` is safe but misses the key fact.
- Gate: every line of the brief maps to an ID.
