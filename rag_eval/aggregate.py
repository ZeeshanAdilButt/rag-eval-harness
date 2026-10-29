"""Deterministic, safety-first aggregation. A pure function of the stored outputs and config."""

from __future__ import annotations

from collections import Counter

from .config import AggregateConfig

NOT_VARIANT_FAULT = {"retrieval_miss"}


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
    use_llm = not (cfg.ignore_llm_weights_for_stub and llm_source.startswith("stub"))
    per_query = []
    tally = {v: Counter() for v in variants}

    for q in queries:
        qid, risk = q["query_id"], q["risk_level"]
        review = by_review[qid]
        veto_set = set(cfg.veto_flags.get(risk, []))
        cands = []
        for v in variants:
            row = by_pair[(qid, v)]
            vetoes = sorted(set(row["failures"]) & veto_set)
            own = [f for f in row["failures"] if f not in NOT_VARIANT_FAULT]
            cands.append({
                "variant": v, "vetoed_by": vetoes, "tier": 0 if not own else 1,
                "score": _score(row, review, v, cfg.weights, use_llm), "failures": row["failures"],
            })
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
            reason = "only safe candidate" if len(alive) == 1 else (
                "lower failure tier" if alive[1]["tier"] > pick["tier"] else "higher tie-break score"
            )
            if tie:
                reason = "tie broken by name"
            winner = pick["variant"]
            tally[winner]["wins"] += 1
        else:
            winner, reason = "none", "every candidate was vetoed"

        tradeoffs = []
        if winner != "none" and use_llm:
            for c in cands:
                if c["variant"] != winner and review["clarity"][c["variant"]] > review["clarity"][winner]:
                    why = f"vetoed: {', '.join(c['vetoed_by'])} on a {risk}-risk question" if c["vetoed_by"] else "more rule-check failures or a lower score"
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
    ranking = sorted(
        variants,
        key=lambda v: (tally[v]["high_risk_vetoes"], tally[v]["vetoes"], -tally[v]["wins"], -mean_score[v], v),
    )
    selected = ranking[0]
    blockers = []
    if tally[selected]["vetoes"]:
        blockers.append(f"{tally[selected]['vetoes']} vetoed answer(s)")
    if tally[selected]["high_risk_blockers"]:
        blockers.append(f"{tally[selected]['high_risk_blockers']} high-risk answer(s) with {', '.join(cfg.ship_blockers_high_risk)}")
    return {
        "selected": selected,
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
