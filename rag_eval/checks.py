"""Rule checks per query and variant. Phrases come from the query, word lists from config."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ChecksConfig
from .schema import Query
from .text import clauses, content_tokens, is_negated, normalize, numbers, sentences, tokens

RISK_CODES = ("overconfident_tone", "policy_violation", "unsupported_claim")


@dataclass(frozen=True)
class Evidence:
    text: str
    token_set: frozenset[str]
    numbers: frozenset[str]

    @classmethod
    def build(cls, retrieved: list[dict], question: str) -> "Evidence":
        joined = " ".join(f"{r['title']}. {r['text']}" for r in retrieved)
        return cls(
            text=joined,
            token_set=frozenset(tokens(joined) + tokens(question)),
            numbers=frozenset(numbers(joined) | numbers(question)),
        )


def _covers(unit: str, phrase_tokens: list[str]) -> bool:
    unit_tokens = set(content_tokens(unit))
    return bool(phrase_tokens) and all(t in unit_tokens for t in phrase_tokens)


def phrase_present(answer: str, phrase: str) -> bool:
    """Substring match after normalisation, or every content token in one sentence."""
    if normalize(phrase).strip() in normalize(answer):
        return True
    wanted = content_tokens(phrase)
    return any(_covers(s, wanted) for s in sentences(answer))


def claim_made(answer: str, claim: str) -> str | None:
    """Return the clause or sentence that makes the claim, honouring negation."""
    wanted = content_tokens(claim)
    if not wanted:
        return None
    claim_negated = is_negated(claim)
    for sentence in sentences(answer):
        for unit in [*clauses(sentence), sentence]:
            if _covers(unit, wanted) and is_negated(unit) == claim_negated:
                return unit
    return None


def _marker_hit(unit: str, marker: str) -> bool:
    unit_tokens = tokens(unit, keep_stopwords=True)
    marker_tokens = tokens(marker, keep_stopwords=True)
    n = len(marker_tokens)
    return n > 0 and any(unit_tokens[i : i + n] == marker_tokens for i in range(len(unit_tokens) - n + 1))


def _is_refusal(sentence: str, cfg: ChecksConfig) -> bool:
    norm = normalize(sentence)
    return any(normalize(m) in norm for m in cfg.refusal_markers)


def grounding(answer: str, ev: Evidence, cfg: ChecksConfig) -> tuple[float, list[str], list[str], bool]:
    """Return (score, unsupported sentences, unsupported numbers, all_refusal)."""
    factual = [s for s in sentences(answer) if not _is_refusal(s, cfg)]
    if not factual:
        return 1.0, [], [], True
    answer_tokens = {t for s in factual for t in content_tokens(s)}
    if not answer_tokens:
        return 1.0, [], [], False
    supported = sum(1 for t in answer_tokens if t in ev.token_set)
    score = round(supported / len(answer_tokens), 3)

    weak = []
    for s in factual:
        toks = set(content_tokens(s))
        if len(toks) >= cfg.sentence_min_tokens:
            share = sum(1 for t in toks if t in ev.token_set) / len(toks)
            if share < cfg.sentence_support_min:
                weak.append(s)
    bad_numbers = sorted(n for s in factual for n in numbers(s) if n not in ev.numbers)
    return score, weak, sorted(set(bad_numbers)), False


def check_answer(
    query: Query, variant: str, answer: str, retrieved: list[dict], cfg: ChecksConfig
) -> dict:
    ev = Evidence.build(retrieved, query.user_question)
    notes: list[str] = []
    failures: set[str] = set()

    retrieved_ids = [r["doc_id"] for r in retrieved]
    retrieval_hit = any(d in retrieved_ids for d in query.expected_doc_ids)
    if not retrieval_hit:
        failures.add("retrieval_miss")
        notes.append(f"no expected doc {query.expected_doc_ids} in top {len(retrieved)} {retrieved_ids}")

    matched = [p for p in query.must_include_any if phrase_present(answer, p)]
    must_include_pass = not query.must_include_any or bool(matched)
    if matched:
        notes.append(f"includes '{matched[0]}'")
    elif query.must_include_any:
        failures.add("missed_key_fact")
        notes.append(f"none of {query.must_include_any} present")

    violations = [(c, u) for c in query.must_not_claim if (u := claim_made(answer, c))]
    must_not_claim_pass = not violations
    for claim, unit in violations:
        failures.add("policy_violation")
        notes.append(f"claims '{claim}' in \"{unit}\"")

    score, weak, bad_numbers, all_refusal = grounding(answer, ev, cfg)
    if all_refusal:
        notes.append("refusal or handoff only; nothing to ground")
    if score < cfg.grounding_min:
        failures.add("unsupported_claim")
        notes.append(f"grounding {score} below {cfg.grounding_min}")
    for s in weak:
        failures.add("unsupported_claim")
        notes.append(f"weakly supported sentence \"{s}\"")
    if bad_numbers:
        failures.add("unsupported_claim")
        notes.append(f"numbers not in evidence or question: {bad_numbers}")

    for sentence in sentences(answer):
        for unit in clauses(sentence):
            hits = [m for m in cfg.certainty_markers if _marker_hit(unit, m)]
            if hits and not is_negated(unit):
                failures.add("overconfident_tone")
                notes.append(f"certainty marker '{hits[0]}' in \"{unit}\"")

    a_tokens = set(content_tokens(answer))
    if a_tokens and not all_refusal:
        relevance = sum(1 for t in a_tokens if t in ev.token_set) / len(a_tokens)
        if relevance < cfg.relevance_min:
            failures.add("irrelevant_answer")
            notes.append(f"relevance {relevance:.2f} below {cfg.relevance_min}")

    risk = sorted(failures & set(RISK_CODES))
    return {
        "query_id": query.query_id,
        "variant": variant,
        "retrieval_hit": retrieval_hit,
        "must_include_pass": must_include_pass,
        "must_not_claim_pass": must_not_claim_pass,
        "grounding_score": score,
        "risk_flags": risk or ["none"],
        "failures": sorted(failures),
        "notes": "; ".join(dict.fromkeys(notes)) or "all checks passed",
    }


def run_checks(inputs, retrieval: list[dict], cfg: ChecksConfig) -> list[dict]:
    by_query = {r["query_id"]: r["retrieved"] for r in retrieval}
    rows = []
    for q in inputs.queries:
        for variant in inputs.variants:
            rows.append(check_answer(q, variant, inputs.answers[q.query_id][variant], by_query[q.query_id], cfg))
    return rows
