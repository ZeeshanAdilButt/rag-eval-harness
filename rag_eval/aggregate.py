"""Deterministic, safety-first aggregation. A pure function of the stored outputs and config."""

from __future__ import annotations

from collections import Counter

from .config import AggregateConfig

NOT_VARIANT_FAULT = {"retrieval_miss"}
RISK_FLAGS = {"policy_violation", "unsupported_claim", "overconfident_tone"}


def _tier(failures: list[str]) -> int:
    """0 = no own failures, 1 = only non-risk failures, 2 = any risk flag (a veto removes it before this matters)."""
    own = {f for f in failures if f not in NOT_VARIANT_FAULT}
    return 2 if own & RISK_FLAGS else (1 if own else 0)


def _llm_warnings(review: dict, variant: str, cfg: AggregateConfig) -> list[str]:
    out = []
    faith = review["faithfulness"][variant]
    if faith <= cfg.llm_block_faithfulness_max:
        out.append(f"faithfulness {faith} <= {cfg.llm_block_faithfulness_max}")
    claims = review.get("overclaim_flags", {}).get(variant, [])
    if cfg.llm_overclaim_blocks_high_risk and claims:
        out.append(f"{len(claims)} overclaim flag(s)")
    return out


def _score(row: dict, review: dict, variant: str, w: dict[str, float], use_llm: bool) -> float:
    s = w["must_include"] * row["must_include_pass"] + w["grounding"] * row["grounding_score"]
    if use_llm:
        s += w["faithfulness"] * (review["faithfulness"][variant] - 1) / 4
        s += w["clarity"] * (review["clarity"][variant] - 1) / 4
        s += w["llm_winner"] * (review["winner"] == variant)
    own = [f for f in row["failures"] if f not in NOT_VARIANT_FAULT]
    s -= w["per_failure"] * len(own)
    return round(s, 4)


def aggregate(queries: list[dict], variants: list[str], scores: list[dict], reviews: list[dict], llm_source: str, cfg: AggregateConfig) -> dict:
    by_pair = {(s["query_id"], s["variant"]): s for s in scores}
    by_review = {r["query_id"]: r for r in reviews}
    live = not llm_source.startswith("stub")
    use_llm = not (cfg.ignore_llm_weights_for_stub and not live)
    per_query = []
    tally = {v: Counter() for v in variants}
    name_wins = Counter()

    for q in queries:
        qid, risk = q["query_id"], q["risk_level"]
        review = by_review[qid]
        veto_set = set(cfg.veto_flags.get(risk, []))
        cands = []
        for v in variants:
            row = by_pair[(qid, v)]
            vetoes = sorted(set(row["failures"]) & veto_set)
            warnings = _llm_warnings(review, v, cfg) if live and risk == "high" else []
            cands.append({
                "variant": v, "vetoed_by": vetoes, "tier": _tier(row["failures"]),
                "score": _score(row, review, v, cfg.weights, use_llm), "failures": row["failures"],
                "llm_warnings": warnings,
            })
            tally[v]["llm_high_risk_warnings"] += bool(warnings)
            tally[v]["vetoes"] += bool(vetoes)
            tally[v]["high_risk_vetoes"] += bool(vetoes) and risk == "high"
            for f in row["failures"]:
                tally[v][f] += 1
            if risk == "high" and set(row["failures"]) & set(cfg.ship_blockers_high_risk):
                tally[v]["high_risk_blockers"] += 1

        alive = sorted((c for c in cands if not c["vetoed_by"]), key=lambda c: (c["tier"], -c["score"], c["variant"]))
        if alive:
            pick = alive[0]
            tie = len(alive) > 1 and (alive[1]["tier"], alive[1]["score"]) == (pick["tier"], pick["score"])
            if len(alive) == 1:
                reason = "only safe candidate"
            elif alive[1]["tier"] > pick["tier"]:
                reason = "lower failure tier"
            else:
                reason = "tie broken by name" if tie else "higher tie-break score"
            winner = pick["variant"]
            tally[winner]["wins"] += 1
            name_wins[winner] += tie
        else:
            winner, reason = "none", "every candidate was vetoed"

        tradeoffs = []
        if winner != "none" and use_llm:
            for c in cands:
                if c["variant"] != winner and review["clarity"][c["variant"]] > review["clarity"][winner]:
                    why = f"vetoed: {', '.join(c['vetoed_by'])} on a {risk}-risk question" if c["vetoed_by"] else "a worse failure tier or a lower tie-break score"
                    tradeoffs.append({
                        "variant": c["variant"], "clarity": review["clarity"][c["variant"]],
                        "winner_clarity": review["clarity"][winner], "why_lost": why,
                    })
        per_query.append({
            "query_id": qid, "risk_level": risk, "winner": winner, "reason": reason,
            "llm_winner": review["winner"], "candidates": cands, "tradeoffs": tradeoffs,
        })

    mean_ground = {v: round(sum(by_pair[(q["query_id"], v)]["grounding_score"] for q in queries) / len(queries), 3) for v in variants}
    mean_score = {v: round(sum(c["score"] for pq in per_query for c in pq["candidates"] if c["variant"] == v) / len(queries), 4) for v in variants}
    def rank_key(v: str) -> tuple:
        return (tally[v]["high_risk_vetoes"], tally[v]["vetoes"], -tally[v]["wins"], -mean_score[v])

    ranking = sorted(variants, key=lambda v: (*rank_key(v), v))
    # Wins that only the name decided are not a preference, so they do not count toward telling the top two apart.
    def tie_key(v: str) -> tuple:
        return (tally[v]["high_risk_vetoes"], tally[v]["vetoes"], -(tally[v]["wins"] - name_wins[v]), -mean_score[v])

    overall_tie = len(ranking) > 1 and tie_key(ranking[0]) == tie_key(ranking[1])
    selected = ranking[0]
    blockers = []
    if tally[selected]["vetoes"]:
        blockers.append(f"{tally[selected]['vetoes']} vetoed answer(s)")
    if tally[selected]["high_risk_blockers"]:
        blockers.append(f"{tally[selected]['high_risk_blockers']} high-risk answer(s) with {', '.join(cfg.ship_blockers_high_risk)}")
    if tally[selected]["llm_high_risk_warnings"]:
        blockers.append(f"{tally[selected]['llm_high_risk_warnings']} high-risk answer(s) with an LLM faithfulness or overclaim warning")
    return {
        "selected": selected,
        "overall_tie": overall_tie,
        "ready_to_ship": not blockers,
        "ship_blockers": blockers,
        "ranking": ranking,
        "llm_source": llm_source,
        "llm_weights_used": use_llm,
        "tally": {v: dict(sorted(tally[v].items())) for v in variants},
        "mean_grounding": mean_ground,
        "mean_score": mean_score,
        "per_query": per_query,
    }
