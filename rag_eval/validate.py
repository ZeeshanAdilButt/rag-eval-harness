"""Artifact validation: re-derive every deterministic output and compare."""

from __future__ import annotations

import json
from pathlib import Path

from .aggregate import aggregate
from .checks import run_checks
from .config import Config
from .llm import read_log, validate_reply
from .report import render_recommendation, render_review_report
from .retrieval import retrieve
from .schema import InputError, load_inputs

ARTIFACTS = ("retrieval.json", "automated_scores.json", "llm_review.json", "llm_calls.jsonl", "recommendation.md", "review_report.md")


def validate(cfg: Config, root: Path) -> list[str]:
    out = root / cfg.paths.out_dir
    errors: list[str] = []

    try:
        inputs = load_inputs(root / cfg.paths.data_dir)
    except InputError as exc:
        return [f"inputs: {exc}"]

    missing = [a for a in ARTIFACTS if not (out / a).exists()]
    taxonomy_path = root / cfg.paths.taxonomy
    if not taxonomy_path.exists():
        missing.append(str(cfg.paths.taxonomy))
    if missing:
        return [f"missing artifact: {m}" for m in missing]

    loaded = {}
    for name in ("retrieval.json", "automated_scores.json", "llm_review.json", str(cfg.paths.taxonomy)):
        path = out / name if name.endswith(".json") and "/" not in name else root / name
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{name}: invalid JSON: {exc}")
    try:
        log = read_log(out / "llm_calls.jsonl")
    except json.JSONDecodeError as exc:
        errors.append(f"llm_calls.jsonl: invalid JSON line: {exc}")
        log = []
    if errors:
        return errors

    qids = [q.query_id for q in inputs.queries]
    retrieval, scores, review = loaded["retrieval.json"], loaded["automated_scores.json"], loaded["llm_review.json"]
    codes = set(loaded[str(cfg.paths.taxonomy)]["codes"])

    # retrieval
    k = min(cfg.retrieval.k, len(inputs.kb))
    if [r.get("query_id") for r in retrieval] != qids:
        errors.append(f"retrieval.json: query ids {[r.get('query_id') for r in retrieval]} != {qids}")
    for r in retrieval:
        got = r.get("retrieved", [])
        if len(got) != k:
            errors.append(f"retrieval.json: {r.get('query_id')} has {len(got)} passages, expected {k}")
        unknown = [d["doc_id"] for d in got if d["doc_id"] not in inputs.docs_by_id]
        if unknown:
            errors.append(f"retrieval.json: {r.get('query_id')} unknown doc ids {unknown}")
    if retrieval != json.loads(json.dumps(retrieve(inputs.kb, inputs.queries, cfg.retrieval))):
        errors.append("retrieval.json: differs from a fresh retrieval (rerun python run.py)")

    # rule checks
    pairs = [(s.get("query_id"), s.get("variant")) for s in scores]
    expected_pairs = [(q, v) for q in qids for v in inputs.variants]
    if sorted(pairs) != sorted(expected_pairs) or len(pairs) != len(set(pairs)):
        errors.append(f"automated_scores.json: expected one row per query x variant ({len(expected_pairs)}), got {len(pairs)}")
    for s in scores:
        bad = [f for f in s.get("failures", []) + s.get("risk_flags", []) if f != "none" and f not in codes]
        if bad:
            errors.append(f"automated_scores.json: {s.get('query_id')}/{s.get('variant')} codes not in taxonomy: {bad}")
    if not errors and scores != json.loads(json.dumps(run_checks(inputs, retrieval, cfg.checks))):
        errors.append("automated_scores.json: differs from a fresh rule-check run")

    # llm review
    meta, reviews = review.get("meta", {}), review.get("reviews", [])
    identity = {q: {v: v for v in inputs.variants} for q in qids}
    _, problems = validate_reply({"reviews": reviews}, identity)
    errors += [f"llm_review.json: {p}" for p in problems]
    if meta.get("source") not in {"live", "replay", "stub", "stub_fallback"}:
        errors.append(f"llm_review.json: meta.source '{meta.get('source')}' is not live, replay, stub or stub_fallback")
    if meta.get("source", "").startswith("stub") and any(not r.get("justification", "").startswith("[STUB]") for r in reviews):
        errors.append("llm_review.json: stub reviews must be labelled [STUB]")
    if not any(l.get("prompt_sha256") == meta.get("prompt_sha256") for l in log):
        errors.append("llm_calls.jsonl: no entry for the prompt hash in llm_review.json")
    if meta.get("source") == "replay" and not any(
        l.get("prompt_sha256") == meta.get("prompt_sha256") and l.get("mode") == "live" and l.get("valid") for l in log
    ):
        errors.append("llm_calls.jsonl: replayed review has no valid live entry")
    if errors:
        return errors

    # reports
    queries = [q.model_dump() for q in inputs.queries]
    agg = aggregate(queries, inputs.variants, scores, reviews, meta["source"], cfg.aggregate)
    if (out / "recommendation.md").read_text(encoding="utf-8") != render_recommendation(agg, reviews, meta):
        errors.append("recommendation.md: not reproducible from the stored outputs")
    if (out / "review_report.md").read_text(encoding="utf-8") != render_review_report(queries, inputs.answers, retrieval, scores, reviews, agg, meta):
        errors.append("review_report.md: not reproducible from the stored outputs")
    return errors
