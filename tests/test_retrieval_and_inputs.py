import json

import pytest

from rag_eval.config import RetrievalConfig
from rag_eval.retrieval import is_hit, retrieve
from rag_eval.schema import Doc, InputError, Query, load_inputs


def _q(text: str) -> Query:
    return Query(query_id="Q", user_question=text, risk_level="low")


def test_ties_break_by_doc_id_and_k_is_respected():
    docs = [Doc(doc_id=i, title="same", text="identical text") for i in ("D3", "D1", "D2")]
    got = retrieve(docs, [_q("identical")], RetrievalConfig(k=2))[0]["retrieved"]
    assert [d["doc_id"] for d in got] == ["D1", "D2"]
    assert got[0]["score"] == got[1]["score"]


def test_relevant_doc_ranks_first_and_fewer_docs_than_k():
    docs = [Doc(doc_id="A", title="Password reset", text="reset your password"), Doc(doc_id="B", title="Demo", text="demo funds")]
    got = retrieve(docs, [_q("How do I reset my password?")], RetrievalConfig(k=5))[0]["retrieved"]
    assert [d["doc_id"] for d in got] == ["A", "B"]


def _write(tmp_path, kb, queries, answers):
    for name, data in (("kb.json", kb), ("queries.json", queries), ("candidate_answers.json", answers)):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")


KB = [{"doc_id": "D1", "title": "t", "text": "x"}]
QS = [{"query_id": "Q1", "user_question": "q?", "expected_doc_ids": ["D1"], "must_include_any": [], "must_not_claim": [], "risk_level": "low"}]


@pytest.mark.parametrize("kb, queries, answers, message", [
    (KB, QS, [{"query_id": "Q1", "answers": {"a": "x", "b": ""}}], "answer for variant 'b' is empty"),
    (KB, QS, [{"query_id": "Q9", "answers": {"a": "x", "b": "y"}}], "unknown query_id"),
    (KB, QS + [{**QS[0], "query_id": "Q2"}], [{"query_id": "Q1", "answers": {"a": "x", "b": "y"}}], "no answers for query_id"),
    (KB, [{**QS[0], "expected_doc_ids": ["D7"]}], [{"query_id": "Q1", "answers": {"a": "x", "b": "y"}}], "unknown expected_doc_ids"),
    (KB, [{k: v for k, v in QS[0].items() if k != "risk_level"}], [{"query_id": "Q1", "answers": {"a": "x", "b": "y"}}], "risk_level"),
    (KB + KB, QS, [{"query_id": "Q1", "answers": {"a": "x", "b": "y"}}], "duplicate doc_id"),
    (KB, QS + [{**QS[0], "query_id": "Q2"}], [{"query_id": "Q1", "answers": {"a": "x", "b": "y"}}, {"query_id": "Q2", "answers": {"a": "x"}}], "need at least two|differ"),
])
def test_bad_inputs_fail_with_a_clear_message(tmp_path, kb, queries, answers, message):
    _write(tmp_path, kb, queries, answers)
    with pytest.raises(InputError, match=message):
        load_inputs(tmp_path)


def test_unicode_inputs_load(tmp_path):
    _write(tmp_path, [{"doc_id": "D1", "title": "Café", "text": "Résumé “quoted” text"}], QS,
           [{"query_id": "Q1", "answers": {"a": "Résumé", "b": "ok"}}])
    assert load_inputs(tmp_path).variants == ["a", "b"]


def test_zero_score_doc_is_returned_in_top_k_but_is_not_a_hit():
    docs = [Doc(doc_id="A", title="Password reset", text="reset your password"), Doc(doc_id="B", title="Demo", text="demo funds")]
    got = retrieve(docs, [_q("How do I reset my password?")], RetrievalConfig(k=2))[0]["retrieved"]
    assert is_hit(got, ["A"]) and not is_hit(got, ["B"]) and not is_hit(got, [])


@pytest.mark.parametrize("raw, stored", [("HIGH", "high"), (" Critical ", "high"), ("Medium", "medium"), ("low", "low")])
def test_risk_level_is_case_insensitive_and_critical_maps_to_high(raw, stored):
    assert Query(query_id="Q", user_question="q?", risk_level=raw).risk_level == stored


def test_unknown_risk_level_is_rejected():
    with pytest.raises(ValueError):
        Query(query_id="Q", user_question="q?", risk_level="urgent")
