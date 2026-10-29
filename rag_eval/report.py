"""Markdown rendering. No timestamps, so validate.py can rebuild and compare bytes."""

from __future__ import annotations

TAXONOMY_ORDER = ["policy_violation", "unsupported_claim", "overconfident_tone", "missed_key_fact", "irrelevant_answer", "retrieval_miss"]

LIMITATIONS = [
    "Banned-claim detection is lexical. It handles word order and simple negation, but misses synonyms (\"view\" for \"see\"), claims split across sentences, and implied claims (\"the money is on its way\" implying approval).",
    "Grounding is token overlap with the retrieved evidence. A sentence that recombines evidence words into a false statement can still score well; only numbers are checked exactly.",
    "Retrieval is BM25 over a handful of documents. A paraphrased question with no shared words can retrieve the wrong document; retrieval_miss marks this and is not charged to the answer.",
    "The LLM review is one call with one model and no human calibration set, so its scores are a second opinion, not ground truth. Blind labels and shuffled order reduce name and position bias; they do not remove model bias.",
    "Thresholds and weights in config/eval.toml were set by reasoning about these fixtures, not fitted on labelled data.",
    "The harness ranks variants. Shipping is a human decision; a \"ready to ship\" line only means no configured blocker fired.",
]


def _table(header: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _source_line(meta: dict) -> str:
    src = meta["source"]
    if src.startswith("stub"):
        return f"**LLM review: STUB ({src}).** Reason: {meta.get('reason')}. The stub is derived from the rule checks, so its scores were given zero weight."
    return f"LLM review: {src}, model `{meta['model']}`, prompt `{meta['prompt_name']}` v{meta['prompt_version']} (sha256 {meta['prompt_sha256'][:12]})."


def render_recommendation(agg: dict, reviews: list[dict], meta: dict) -> str:
    variants = agg["ranking"]
    sel = agg["selected"]
    t = agg["tally"]
    by_review = {r["query_id"]: r for r in reviews}
    lines = ["# Recommendation", ""]
    status = "ready to ship" if agg["ready_to_ship"] else "selected as the safer option, not ready to ship: " + "; ".join(agg["ship_blockers"])
    lines += [f"**Selected variant: `{sel}`** ({status}).", ""]
    others = [v for v in variants if v != sel]
    lines += [f"In plain words: `{sel}` won {t[sel].get('wins', 0)} of {len(agg['per_query'])} questions with "
              f"{t[sel].get('vetoes', 0)} vetoed answer(s). "
              + " ".join(f"`{v}` had {t[v].get('vetoes', 0)} vetoed answer(s) and {t[v].get('policy_violation', 0)} banned claim(s)." for v in others), ""]
    if agg.get("overall_tie"):
        lines += [f"**The selection is a tie broken by name.** `{variants[0]}` and `{variants[1]}` are equal on every ranking key, so the harness has no real preference between them.", ""]
    lines += [_source_line(meta), ""]

    lines += ["## Summary", ""]
    llm_cols = not meta["source"].startswith("stub")
    header = ["variant", "query wins", "vetoes", "high-risk vetoes", *TAXONOMY_ORDER, "mean grounding"]
    if llm_cols:
        header += ["mean faithfulness", "mean clarity", "high-risk LLM warnings"]
    rows = []
    for v in variants:
        row = [f"`{v}`", t[v].get("wins", 0), t[v].get("vetoes", 0), t[v].get("high_risk_vetoes", 0), *[t[v].get(c, 0) for c in TAXONOMY_ORDER], agg["mean_grounding"][v]]
        if llm_cols:
            n = len(reviews)
            row += [round(sum(r["faithfulness"][v] for r in reviews) / n, 2), round(sum(r["clarity"][v] for r in reviews) / n, 2), t[v].get("llm_high_risk_warnings", 0)]
        rows.append(row)
    lines += [_table(header, rows), ""]

    lines += ["## Top reasons", ""]
    for i, v in enumerate(variants, 1):
        bits = [f"{t[v].get('high_risk_vetoes', 0)} high-risk veto(es)", f"{t[v].get('vetoes', 0)} veto(es) in total", f"{t[v].get('wins', 0)} query win(s)", f"mean tie-break score {agg['mean_score'][v]}"]
        if llm_cols:
            bits.insert(2, f"{t[v].get('llm_high_risk_warnings', 0)} high-risk LLM warning(s)")
        lines.append(f"{i}. `{v}`: " + ", ".join(bits) + ".")
    lines += ["", "Ranking order: fewest high-risk vetoes, then fewest vetoes, then most wins, then mean score, then name. LLM warnings are ship blockers on the selected variant, never vetoes.", ""]

    lines += ["## Per-query reasoning", ""]
    for pq in agg["per_query"]:
        lines.append(f"### {pq['query_id']} ({pq['risk_level']} risk): pick `{pq['winner']}`, {pq['reason']}")
        lines.append("")
        for c in pq["candidates"]:
            state = f"vetoed by {', '.join(c['vetoed_by'])}" if c["vetoed_by"] else ("clean" if c["tier"] == 0 else "failures: " + ", ".join(c["failures"]))
            if c["vetoed_by"] and len(c["failures"]) > len(c["vetoed_by"]):
                state += "; also " + ", ".join(f for f in c["failures"] if f not in c["vetoed_by"])
            warn = "; LLM warning: " + ", ".join(c["llm_warnings"]) if c.get("llm_warnings") else ""
            lines.append(f"- `{c['variant']}`: {state}; score {c['score']}{warn}")
        lines.append(f"- LLM winner: `{pq['llm_winner']}`. LLM note (unverified): \"{by_review[pq['query_id']]['justification']}\"")
        lines.append("")

    lines += ["## Clarity versus safety", ""]
    trade = [(pq, tr) for pq in agg["per_query"] for tr in pq["tradeoffs"]]
    if not agg["llm_weights_used"]:
        lines.append("Not judged in this run: the LLM review was a stub, so clarity scores are neutral placeholders.")
    elif not trade:
        lines.append("No query where a clearer answer lost on safety.")
    for pq, tr in trade:
        lines.append(f"- {pq['query_id']}: `{tr['variant']}` reads more clearly (clarity {tr['clarity']} vs {tr['winner_clarity']}) but lost, {tr['why_lost']}. The harness prefers the safer answer because a clear wrong answer on a {pq['risk_level']}-risk question does more harm than a plain correct one.")
    lines.append("")

    lines += ["## Known limitations", ""]
    lines += [f"- {x}" for x in LIMITATIONS]
    lines.append("")
    return "\n".join(lines)


def render_review_report(queries: list[dict], answers: dict, retrieval: list[dict], scores: list[dict], reviews: list[dict], agg: dict, meta: dict) -> str:
    by_ret = {r["query_id"]: r["retrieved"] for r in retrieval}
    by_pair = {(s["query_id"], s["variant"]): s for s in scores}
    by_review = {r["query_id"]: r for r in reviews}
    by_agg = {p["query_id"]: p for p in agg["per_query"]}
    lines = ["# Review report", "", _source_line(meta), "",
             "How to read this: `retrieval_hit` means an expected doc was in the top 2, `must_include` means a key fact was present, "
             "`must_not_claim` means no banned claim was made. Risk flags come from `outputs/failure_taxonomy.json`.", ""]
    for q in queries:
        qid = q["query_id"]
        pq, rv = by_agg[qid], by_review[qid]
        lines += [f"## {qid} ({q['risk_level']} risk)", "", f"**Question:** {q['user_question']}", ""]
        lines += ["**Retrieved:**", "", _table(["rank", "doc_id", "title", "score", "expected"], [
            [i + 1, d["doc_id"], d["title"], d["score"], "yes" if d["doc_id"] in q["expected_doc_ids"] else ""] for i, d in enumerate(by_ret[qid])
        ]), ""]
        lines += [f"> [{d['doc_id']}] {d['text']}" for d in by_ret[qid]] + [""]
        for c in pq["candidates"]:
            v = c["variant"]
            s = by_pair[(qid, v)]
            failed = [n for n, ok in (("retrieval_hit", s["retrieval_hit"]), ("must_include", s["must_include_pass"]), ("must_not_claim", s["must_not_claim_pass"])) if not ok]
            lines += [f"### `{v}`", "", f"> {answers[qid][v]}", ""]
            lines.append(f"- Failed checks: {', '.join(failed) or 'none'}; risk flags: {', '.join(s['risk_flags'])}; grounding {s['grounding_score']}")
            lines.append(f"- LLM: faithfulness {rv['faithfulness'][v]}, clarity {rv['clarity'][v]}, overclaim: {'yes' if rv['overclaim_flags'][v] else 'no'}")
            lines.append(f"- Notes: {s['notes']}")
            lines.append("")
        lines += [f"**LLM winner:** `{rv['winner']}`. LLM note (unverified): \"{rv['justification']}\"", "", f"**Final pick:** `{pq['winner']}` ({pq['reason']}).", ""]
    return "\n".join(lines)
