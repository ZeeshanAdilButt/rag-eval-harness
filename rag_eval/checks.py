"""Rule checks per query and variant. Phrases come from the query, word lists from config."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ChecksConfig
from .retrieval import is_hit
from .schema import Doc, Query
from .text import NEGATORS, clauses, content_tokens, is_negated, negated_near, normalize, numbers, sentences, tokens

RISK_CODES = ("overconfident_tone", "policy_violation", "unsupported_claim")


@dataclass(frozen=True)
class Evidence:
    text: str
    token_set: frozenset[str]
    numbers: frozenset[str]

    @classmethod
    def build(cls, retrieved: list[dict], question: str) -> "Evidence":
        joined = " ".join(f"{r['title']}. {r['text']}" for r in retrieved if r.get("score", 1.0) > 0)
        return cls(
            text=joined,
            token_set=frozenset(tokens(joined) + tokens(question)),
            numbers=frozenset(numbers(joined) | numbers(question)),
        )


def _covers(unit: str, phrase_tokens: list[str]) -> bool:
    unit_tokens = set(content_tokens(unit))
    return bool(phrase_tokens) and all(t in unit_tokens for t in phrase_tokens)


def _units(answer: str) -> list[str]:
    """Clauses first, then the whole sentence, for every sentence."""
    return [u for sentence in sentences(answer) for u in [*clauses(sentence), sentence]]


def phrase_present(answer: str, phrase: str) -> bool:
    """The phrase as a contiguous token run, or all its content tokens, in one unit.

    A unit that negates the phrase's words does not count unless the phrase itself is negated.
    """
    wanted = content_tokens(phrase)
    phrase_negated = is_negated(phrase)
    for unit in _units(answer):
        if (_marker_hit(unit, phrase) or _covers(unit, wanted)) and negated_near(unit, wanted) == phrase_negated:
            return True
    return False


def claim_made(answer: str, claim: str) -> str | None:
    """Return the clause or sentence that makes the claim, honouring negation near the claim's words."""
    wanted = content_tokens(claim)
    if not wanted:
        return None
    claim_negated = is_negated(claim)
    for unit in _units(answer):
        if _covers(unit, wanted) and negated_near(unit, wanted) == claim_negated:
            return unit
    return None


def _marker_hit(unit: str, marker: str) -> bool:
    unit_tokens = tokens(unit, keep_stopwords=True)
    marker_tokens = tokens(marker, keep_stopwords=True)
    n = len(marker_tokens)
    return n > 0 and any(unit_tokens[i : i + n] == marker_tokens for i in range(len(unit_tokens) - n + 1))


def _is_refusal(clause: str, cfg: ChecksConfig) -> bool:
    norm = normalize(clause)
    return any(normalize(m) in norm for m in cfg.refusal_markers)


def factual_units(answer: str, cfg: ChecksConfig) -> list[str]:
    """Sentences with their refusal clauses dropped; a sentence that is all refusal vanishes."""
    out = []
    for sentence in sentences(answer):
        parts = clauses(sentence)
        kept = [c for c in parts if not _is_refusal(c, cfg)]
        if kept:
            out.append(sentence if len(kept) == len(parts) else ", ".join(kept))
    return out


def grounding(answer: str, ev: Evidence, cfg: ChecksConfig) -> tuple[float, list[str], list[str], bool]:
    """Return (score, unsupported sentences, unsupported numbers, all_refusal)."""
    factual = factual_units(answer, cfg)
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
    query: Query,
    variant: str,
    answer: str,
    retrieved: list[dict],
    cfg: ChecksConfig,
    kb: dict[str, Doc] | None = None,
) -> dict:
    notes: list[str] = []
    failures: set[str] = set()

    retrieved_ids = [r["doc_id"] for r in retrieved]
    retrieval_hit = is_hit(retrieved, query.expected_doc_ids) if query.expected_doc_ids else True
    evidence_docs = retrieved
    if not query.expected_doc_ids:
        notes.append("no expected docs")
    elif not retrieval_hit:
        failures.add("retrieval_miss")
        notes.append(f"no expected doc {query.expected_doc_ids} with score > 0 in top {len(retrieved)} {retrieved_ids}")
        # A retrieval miss must not make a correct answer look unsupported.
        extra = [
            {"doc_id": d.doc_id, "title": d.title, "text": d.text}
            for i in query.expected_doc_ids
            if kb and (d := kb.get(i)) and i not in retrieved_ids
        ]
        if extra:
            evidence_docs = [*retrieved, *extra]
            notes.append("graded against expected docs because retrieval missed")
    ev = Evidence.build(evidence_docs, query.user_question)

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

    # A marker the evidence itself uses ("100 percent secure") is not an overclaim.
    markers = [m for m in cfg.certainty_markers if not _marker_hit(ev.text, m)]
    for sentence in sentences(answer):
        for unit in clauses(sentence):
            for m in markers:
                if _marker_hit(unit, m) and not negated_near(unit, [t for t in tokens(m, keep_stopwords=True) if t not in NEGATORS]):
                    failures.add("overconfident_tone")
                    notes.append(f"certainty marker '{m}' in \"{unit}\"")
                    break

    a_tokens = {t for s in sentences(answer) for t in content_tokens(s)}
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
            rows.append(
                check_answer(q, variant, inputs.answers[q.query_id][variant], by_query[q.query_id], cfg, inputs.docs_by_id)
            )
    return rows
