"""Staleness detection in validate and the auto storage fallback."""

import json
import shutil
from pathlib import Path

import pytest

from rag_eval.config import load_config
from rag_eval.pipeline import run_pipeline
from rag_eval.storage import SQLStore, open_store
from rag_eval.validate import validate

ROOT = Path(__file__).resolve().parent.parent


def _project(tmp_path: Path) -> Path:
    for sub in ("data", "config", "prompts"):
        shutil.copytree(ROOT / sub, tmp_path / sub, ignore=shutil.ignore_patterns("local.toml"))
    return tmp_path


def _cfg(root: Path, *extra: str):
    return load_config(root / "config" / "eval.toml", overrides=["llm.mode=stub", "storage.backend=sqlite", *extra])


def test_validate_detects_a_stale_prompt_hash(tmp_path):
    root = _project(tmp_path)
    cfg = _cfg(root)
    run_pipeline(cfg, root, log=lambda *_: None)
    assert validate(cfg, root) == []

    path = root / "data" / "candidate_answers.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["answers"]["prompt_a"] += " Extra sentence."
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert any("prompt hash is stale" in e for e in validate(cfg, root))


def test_validate_detects_a_changed_prompt_template_and_token_budget(tmp_path):
    root = _project(tmp_path)
    cfg = _cfg(root)
    run_pipeline(cfg, root, log=lambda *_: None)
    assert any("prompt hash is stale" in e for e in validate(_cfg(root, "llm.max_tokens_per_pair=60"), root))
    tmpl = root / "prompts" / "llm_review.toml"
    tmpl.write_text(tmpl.read_text(encoding="utf-8").replace("strict reviewer", "lenient reviewer"), encoding="utf-8")
    assert any("prompt hash is stale" in e for e in validate(cfg, root))


def test_auto_storage_falls_back_to_sqlite_with_one_warning(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    secret = "hunter2-do-not-print"
    monkeypatch.setenv("DATABASE_URL", f"postgresql://user:{secret}@127.0.0.1:1/db")
    cfg = load_config(root / "config" / "eval.toml", overrides=["storage.backend=auto", "llm.mode=stub"])
    store = open_store(cfg.storage, root)
    try:
        assert isinstance(store, SQLStore) and store.dialect == "sqlite"
    finally:
        store.close()
    err = capsys.readouterr().err
    assert err.count("warning:") == 1 and err.count("\n") == 1
    assert secret not in err and "postgresql://" not in err


def test_explicit_postgres_still_fails_loudly(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@127.0.0.1:1/db")
    cfg = load_config(root / "config" / "eval.toml", overrides=["storage.backend=postgres", "llm.mode=stub"])
    with pytest.raises(Exception):
        open_store(cfg.storage, root)
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(RuntimeError):
        open_store(cfg.storage, root)
