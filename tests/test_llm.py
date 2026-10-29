import json
from pathlib import Path

import pytest

from rag_eval.config import load_config
from rag_eval.llm import ProviderError, ProviderReply, label_order, run_review, validate_reply
from rag_eval.retrieval import retrieve
from rag_eval.checks import run_checks
from rag_eval.schema import load_inputs

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def setup():
    cfg = load_config(ROOT / "config" / "eval.toml")
    inputs = load_inputs(ROOT / "data")
    retrieval = retrieve(inputs.kb, inputs.queries, cfg.retrieval)
    scores = run_checks(inputs, retrieval, cfg.checks)
    return cfg, inputs, retrieval, scores


def _valid_payload(inputs, cfg, faith=4):
    reviews = []
    for q in inputs.queries:
        labels = list(label_order(q.query_id, inputs.variants, cfg.llm.blind_labels))
        reviews.append({"query_id": q.query_id, "winner": labels[0],
                        "faithfulness": {l: faith for l in labels}, "clarity": {l: 4 for l in labels},
                        "overclaim_flags": {l: [] for l in labels}, "justification": "ok"})
    return {"reviews": reviews}


class FakeProvider:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, system, user, schema, cfg, api_key):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ProviderReply(json.dumps(reply) if not isinstance(reply, str) else reply, "fake-model", "end_turn", 10, 10)


def test_validate_reply_rejects_unknown_variant_and_bad_scores():
    labels = {"Q1": {"answer_1": "a", "answer_2": "b"}}
    bad = {"reviews": [{"query_id": "Q1", "winner": "prompt_z", "faithfulness": {"answer_1": 9, "answer_2": 3},
                        "clarity": {"answer_1": 3}, "overclaim_flags": {"answer_1": [], "answer_2": []}, "justification": "x"}]}
    _, errors = validate_reply(bad, labels)
    joined = " ".join(errors)
    assert "winner 'prompt_z'" in joined and "outside 1-5" in joined and "clarity keys" in joined
    _, errors = validate_reply({"reviews": []}, labels)
    assert errors == ["missing review for Q1"]


def test_blind_labels_are_stable_and_cover_every_variant():
    m1 = label_order("Q1", ["prompt_a", "prompt_b", "prompt_c"], True)
    assert m1 == label_order("Q1", ["prompt_c", "prompt_a", "prompt_b"], True)
    assert sorted(m1.values()) == ["prompt_a", "prompt_b", "prompt_c"] and list(m1) == ["answer_1", "answer_2", "answer_3"]


def test_no_key_writes_a_labelled_stub(setup, tmp_path, monkeypatch):
    cfg, inputs, retrieval, scores = setup
    monkeypatch.delenv(cfg.llm.api_key_env, raising=False)
    res = run_review(inputs, retrieval, scores, cfg.llm, ROOT, tmp_path / "log.jsonl")
    assert res.meta["source"] == "stub"
    assert all(r["justification"].startswith("[STUB]") for r in res.reviews)
    assert res.new_log_lines[0]["mode"] == "stub"


def test_invalid_reply_retries_once_then_succeeds_and_replays_without_a_call(setup, tmp_path, monkeypatch):
    cfg, inputs, retrieval, scores = setup
    monkeypatch.setenv(cfg.llm.api_key_env, "test-key")
    fake = FakeProvider(["not json", _valid_payload(inputs, cfg)])
    log = tmp_path / "log.jsonl"
    res = run_review(inputs, retrieval, scores, cfg.llm, ROOT, log, provider=fake)
    assert fake.calls == 2 and res.meta["source"] == "live"
    assert [l["valid"] for l in res.new_log_lines] == [False, True]
    assert "test-key" not in json.dumps(res.new_log_lines)
    log.write_text("".join(json.dumps(l) + "\n" for l in res.new_log_lines), encoding="utf-8")

    again = FakeProvider([])
    replay = run_review(inputs, retrieval, scores, cfg.llm, ROOT, log, provider=again)
    assert again.calls == 0 and replay.meta["source"] == "replay" and replay.reviews == res.reviews


def test_two_invalid_replies_fall_back_to_a_stub(setup, tmp_path, monkeypatch):
    cfg, inputs, retrieval, scores = setup
    monkeypatch.setenv(cfg.llm.api_key_env, "k")
    bad = _valid_payload(inputs, cfg, faith=7)
    fake = FakeProvider([bad, bad])
    res = run_review(inputs, retrieval, scores, cfg.llm, ROOT, tmp_path / "log.jsonl", provider=fake)
    assert fake.calls == 2 and res.meta["source"] == "stub_fallback"
    assert res.new_log_lines[-1]["mode"] == "stub_fallback"


def test_auth_error_is_not_retried(setup, tmp_path, monkeypatch):
    cfg, inputs, retrieval, scores = setup
    monkeypatch.setenv(cfg.llm.api_key_env, "k")
    fake = FakeProvider([ProviderError("AUTH", "HTTP 401")])
    res = run_review(inputs, retrieval, scores, cfg.llm, ROOT, tmp_path / "log.jsonl", provider=fake)
    assert fake.calls == 1 and res.meta["source"] == "stub_fallback"
