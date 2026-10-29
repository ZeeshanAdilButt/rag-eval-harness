"""End to end: swapped fixtures, a third variant, tampering, storage claims, config layers."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from rag_eval.config import ConfigError, load_config
from rag_eval.pipeline import run_pipeline
from rag_eval.storage import SQLStore
from rag_eval.validate import validate

ROOT = Path(__file__).resolve().parent.parent


def _project(tmp_path: Path) -> Path:
    for sub in ("data", "config", "prompts"):
        shutil.copytree(ROOT / sub, tmp_path / sub, ignore=shutil.ignore_patterns("local.toml"))
    return tmp_path


def _cfg(root: Path, *extra: str):
    return load_config(root / "config" / "eval.toml", overrides=["llm.mode=stub", "storage.backend=sqlite", *extra])


def test_sample_run_validates_and_is_byte_identical_on_rerun(tmp_path):
    root = _project(tmp_path)
    cfg = _cfg(root)
    run_pipeline(cfg, root, log=lambda *_: None)
    first = {p.name: p.read_bytes() for p in (root / "outputs").iterdir() if p.suffix != ".sqlite"}
    run_pipeline(cfg, root, log=lambda *_: None)
    second = {p.name: p.read_bytes() for p in (root / "outputs").iterdir() if p.suffix != ".sqlite"}
    assert first == second
    assert validate(cfg, root) == []


def test_third_variant_and_renamed_ids_need_no_code_change(tmp_path):
    root = _project(tmp_path)
    data = root / "data"
    queries = json.loads((data / "queries.json").read_text(encoding="utf-8"))
    answers = json.loads((data / "candidate_answers.json").read_text(encoding="utf-8"))
    rename = {q["query_id"]: f"case-{i}" for i, q in enumerate(queries)}
    for q in queries:
        q["query_id"] = rename[q["query_id"]]
    for row in answers:
        row["query_id"] = rename[row["query_id"]]
        row["answers"]["prompt_c"] = row["answers"]["prompt_a"] + " Let me know if you need anything else."
    (data / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    (data / "candidate_answers.json").write_text(json.dumps(answers), encoding="utf-8")

    cfg = _cfg(root)
    result = run_pipeline(cfg, root, log=lambda *_: None)
    assert validate(cfg, root) == []
    scores = json.loads((root / "outputs" / "automated_scores.json").read_text(encoding="utf-8"))
    assert {s["variant"] for s in scores} == {"prompt_a", "prompt_b", "prompt_c"}
    assert {s["query_id"] for s in scores} == set(rename.values())
    assert result["selected"] in {"prompt_a", "prompt_c"}


def test_validation_catches_tampering(tmp_path):
    root = _project(tmp_path)
    cfg = _cfg(root)
    run_pipeline(cfg, root, log=lambda *_: None)
    rec = root / "outputs" / "recommendation.md"
    rec.write_text(rec.read_text(encoding="utf-8").replace("prompt_a", "prompt_b", 1), encoding="utf-8")
    review = root / "outputs" / "llm_review.json"
    data = json.loads(review.read_text(encoding="utf-8"))
    data["reviews"][0]["faithfulness"]["prompt_a"] = 6
    review.write_text(json.dumps(data), encoding="utf-8")
    errors = " ".join(validate(cfg, root))
    assert "outside 1-5" in errors


def test_validation_reports_missing_artifacts(tmp_path):
    root = _project(tmp_path)
    assert any("missing artifact" in e for e in validate(_cfg(root), root))


def test_claim_is_exclusive_until_stale(tmp_path):
    store = SQLStore(sqlite3.connect(tmp_path / "s.sqlite"), "sqlite")
    store.migrate()
    task = store.create_task("run-1", 0, "check", "rule_checker")
    assert store.claim(task, "worker-a", timeout_s=300)
    assert not store.claim(task, "worker-b", timeout_s=300)
    store.update("agent_tasks", {"task_id": task}, {"claimed_at": "2000-01-01T00:00:00+00:00"})
    assert store.claim(task, "worker-b", timeout_s=300)


def test_every_table_gets_rows_for_a_run(tmp_path):
    root = _project(tmp_path)
    result = run_pipeline(_cfg(root), root, log=lambda *_: None)
    conn = sqlite3.connect(root / "outputs" / "runs.sqlite")
    for table in ("runs", "agent_tasks", "kb_docs", "queries", "candidate_answers", "retrievals", "rule_checks",
                  "prompt_templates", "llm_requests", "llm_reviews", "picks", "recommendations"):
        (count,) = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert count > 0, table
    assert conn.execute("SELECT status FROM runs WHERE run_id = ?", (result["run_id"],)).fetchone() == ("done",)


def test_config_layers_and_errors(tmp_path):
    base = ROOT / "config" / "eval.toml"
    cfg = load_config(base, profile="strict", overrides=["retrieval.k=3"], environ={"RAG_EVAL__LLM__MODEL": "m-1"})
    assert cfg.checks.grounding_min == 0.75 and cfg.retrieval.k == 3 and cfg.llm.model == "m-1"
    with pytest.raises(ConfigError, match="retrieval.kk"):
        load_config(base, overrides=["retrieval.kk=3"])
    with pytest.raises(ConfigError, match="unknown profile"):
        load_config(base, profile="nope")


def test_unknown_agent_fails_before_running(tmp_path):
    root = _project(tmp_path)
    cfg = _cfg(root)
    cfg.pipeline.stages[1].agent = "missing_agent"
    with pytest.raises(ValueError, match="unknown agent"):
        run_pipeline(cfg, root, log=lambda *_: None)
