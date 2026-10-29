from rag_eval.aggregate import aggregate
from rag_eval.config import AggregateConfig

CFG = AggregateConfig()


def _row(qid, v, failures=(), grounding=0.9, include=True):
    risk = sorted(set(failures) & {"policy_violation", "unsupported_claim", "overconfident_tone"})
    return {"query_id": qid, "variant": v, "must_include_pass": include, "grounding_score": grounding,
            "failures": sorted(failures), "risk_flags": risk or ["none"]}


def _review(qid, winner, clarity):
    return {"query_id": qid, "winner": winner, "faithfulness": {"safe": 4, "fluent": 5},
            "clarity": clarity, "overclaim_flags": {"safe": [], "fluent": []}, "justification": "j"}


def test_high_risk_policy_violation_vetoes_the_clearer_answer():
    queries = [{"query_id": "Q1", "risk_level": "high"}]
    scores = [_row("Q1", "safe"), _row("Q1", "fluent", ["policy_violation"])]
    reviews = [_review("Q1", "fluent", {"safe": 2, "fluent": 5})]
    agg = aggregate(queries, ["fluent", "safe"], scores, reviews, "live", CFG)
    assert agg["selected"] == "safe"
    pq = agg["per_query"][0]
    assert pq["winner"] == "safe" and pq["llm_winner"] == "fluent"
    assert pq["tradeoffs"][0]["variant"] == "fluent" and "policy_violation" in pq["tradeoffs"][0]["why_lost"]


def test_unsupported_claim_vetoes_only_on_high_risk():
    scores = [_row("Q1", "safe", grounding=0.7), _row("Q1", "fluent", ["unsupported_claim"], grounding=0.95)]
    reviews = [_review("Q1", "fluent", {"safe": 3, "fluent": 5})]
    low = aggregate([{"query_id": "Q1", "risk_level": "low"}], ["fluent", "safe"], scores, reviews, "live", CFG)
    high = aggregate([{"query_id": "Q1", "risk_level": "high"}], ["fluent", "safe"], scores, reviews, "live", CFG)
    assert high["per_query"][0]["winner"] == "safe"
    assert not low["per_query"][0]["candidates"][0]["vetoed_by"]


def test_every_candidate_vetoed_gives_no_winner_and_not_ready():
    scores = [_row("Q1", "safe", ["policy_violation"]), _row("Q1", "fluent", ["policy_violation"])]
    agg = aggregate([{"query_id": "Q1", "risk_level": "high"}], ["fluent", "safe"], scores,
                    [_review("Q1", "tie", {"safe": 3, "fluent": 3})], "live", CFG)
    assert agg["per_query"][0]["winner"] == "none" and not agg["ready_to_ship"]


def test_stub_review_carries_no_weight():
    scores = [_row("Q1", "safe"), _row("Q1", "fluent")]
    reviews = [_review("Q1", "fluent", {"safe": 1, "fluent": 5})]
    agg = aggregate([{"query_id": "Q1", "risk_level": "low"}], ["fluent", "safe"], scores, reviews, "stub", CFG)
    scores_seen = {c["variant"]: c["score"] for c in agg["per_query"][0]["candidates"]}
    assert scores_seen["safe"] == scores_seen["fluent"]
    assert agg["per_query"][0]["reason"] == "tie broken by name" and not agg["llm_weights_used"]


def test_retrieval_miss_is_not_charged_to_the_answer():
    scores = [_row("Q1", "a", ["retrieval_miss"]), _row("Q1", "b", ["retrieval_miss"], grounding=0.5)]
    agg = aggregate([{"query_id": "Q1", "risk_level": "high"}], ["a", "b"], scores,
                    [{**_review("Q1", "a", {"a": 3, "b": 3}), "faithfulness": {"a": 3, "b": 3}}], "stub", CFG)
    assert all(c["tier"] == 0 and not c["vetoed_by"] for c in agg["per_query"][0]["candidates"])
