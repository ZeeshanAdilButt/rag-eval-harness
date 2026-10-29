"""Orchestrator: one run, one claimed task per stage, every run ends in a terminal state."""

from __future__ import annotations

import hashlib
import os
import socket
import uuid
from pathlib import Path

from .agents import REGISTRY, RunContext
from .config import Config
from .schema import load_inputs
from .storage import now, open_store

INPUT_FILES = ("kb.json", "queries.json", "candidate_answers.json")


def check_stages(cfg: Config) -> None:
    """Every agent exists and every agent's inputs are produced by an earlier stage."""
    available: set[str] = set()
    for s in cfg.pipeline.stages:
        if s.agent not in REGISTRY:
            raise ValueError(f"pipeline stage '{s.stage}': unknown agent '{s.agent}'; registered: {sorted(REGISTRY)}")
        missing = set(REGISTRY[s.agent].requires) - available
        if missing:
            raise ValueError(f"pipeline stage '{s.stage}': agent '{s.agent}' needs {sorted(missing)} from an earlier stage")
        available |= set(REGISTRY[s.agent].provides)


def inputs_digest(data_dir: Path) -> str:
    h = hashlib.sha256()
    for name in INPUT_FILES:
        h.update(name.encode() + b"\0" + (data_dir / name).read_bytes())
    return h.hexdigest()


def run_pipeline(cfg: Config, root: Path, log=print) -> dict:
    check_stages(cfg)
    data_dir, out_dir = root / cfg.paths.data_dir, root / cfg.paths.out_dir
    inputs = load_inputs(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(cfg.storage, root)
    run_id = str(uuid.uuid4())
    worker = f"{socket.gethostname()}:{os.getpid()}"
    store.insert("runs", [{
        "run_id": run_id, "status": "running", "started_at": now(), "config": cfg.model_dump(mode="json"),
        "config_sha256": cfg.digest(), "inputs_sha256": inputs_digest(data_dir), "variants": inputs.variants,
    }])
    store.insert("kb_docs", [{"run_id": run_id, **d.model_dump()} for d in inputs.kb])
    store.insert("queries", [{"run_id": run_id, **q.model_dump()} for q in inputs.queries])
    store.insert("candidate_answers", [
        {"run_id": run_id, "query_id": qid, "variant": v, "answer": text}
        for qid, answers in inputs.answers.items() for v, text in sorted(answers.items())
    ])

    ctx = RunContext(cfg=cfg, root=root, out_dir=out_dir, inputs=inputs, store=store, run_id=run_id)
    tasks = [(store.create_task(run_id, i, s.stage, s.agent), s) for i, s in enumerate(cfg.pipeline.stages)]
    try:
        for task_id, stage in tasks:
            if not store.claim(task_id, f"{stage.agent}@{worker}", cfg.storage.claim_timeout_s):
                raise RuntimeError(f"task {task_id} ({stage.stage}) is claimed by another worker")
            summary = REGISTRY[stage.agent]().run(ctx)
            store.update("agent_tasks", {"task_id": task_id}, {"status": "done", "finished_at": now(), "summary": summary})
            log(f"[{stage.stage}] {stage.agent}: {summary}")
    except Exception as exc:
        store.update("agent_tasks", {"run_id": run_id, "status": "running"}, {"status": "failed", "finished_at": now(), "error": str(exc)[:2000]})
        store.update("runs", {"run_id": run_id}, {"status": "failed", "finished_at": now(), "error": str(exc)[:2000]})
        store.close()
        raise
    agg = ctx.artifacts.get("aggregate", {})
    store.update("runs", {"run_id": run_id}, {
        "status": "done", "finished_at": now(), "llm_source": ctx.artifacts.get("llm_meta", {}).get("source"),
        "selected_variant": agg.get("selected"), "ready_to_ship": agg.get("ready_to_ship"),
    })
    store.close()
    return {"run_id": run_id, "storage": store.dialect, **{k: agg.get(k) for k in ("selected", "ready_to_ship")}}
