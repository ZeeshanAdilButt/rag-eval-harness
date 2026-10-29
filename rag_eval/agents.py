"""Pipeline agents. Each stage in config is run by one registered agent.

An agent reads what earlier stages left in the context, writes its own artifact and
rows, and returns a short summary that is stored on its claimed task. To add one:
subclass Agent, decorate it with @register, and name it in [[pipeline.stages]].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .aggregate import aggregate
from .checks import run_checks
from .config import Config
from .llm import PromptTemplate, run_review
from .report import render_recommendation, render_review_report
from .retrieval import retrieve
from .schema import Inputs
from .storage import Store


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


@dataclass
class RunContext:
    cfg: Config
    root: Path
    out_dir: Path
    inputs: Inputs
    store: Store
    run_id: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class Agent:
    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def run(self, ctx: RunContext) -> dict[str, Any]:
        raise NotImplementedError


REGISTRY: dict[str, type[Agent]] = {}


def register(cls: type[Agent]) -> type[Agent]:
    REGISTRY[cls.name] = cls
    return cls


@register
class BM25Retriever(Agent):
    name = "bm25_retriever"
    provides = ("retrieval",)

    def run(self, ctx: RunContext) -> dict[str, Any]:
        result = retrieve(ctx.inputs.kb, ctx.inputs.queries, ctx.cfg.retrieval)
        ctx.artifacts["retrieval"] = result
        write_json(ctx.out_dir / "retrieval.json", result)
        ctx.store.insert("retrievals", [
            {"run_id": ctx.run_id, "query_id": r["query_id"], "rank": i + 1, "doc_id": d["doc_id"], "score": d["score"]}
            for r in result for i, d in enumerate(r["retrieved"])
        ])
        return {"queries": len(result), "k": ctx.cfg.retrieval.k}


@register
class RuleChecker(Agent):
    name = "rule_checker"
    requires = ("retrieval",)
    provides = ("scores",)

    def run(self, ctx: RunContext) -> dict[str, Any]:
        rows = run_checks(ctx.inputs, ctx.artifacts["retrieval"], ctx.cfg.checks)
        ctx.artifacts["scores"] = rows
        write_json(ctx.out_dir / "automated_scores.json", rows)
        vocab = json.loads((ctx.root / ctx.cfg.paths.taxonomy).read_text(encoding="utf-8"))
        write_json(ctx.out_dir / "failure_taxonomy.json", {
            "version": vocab["version"], "codes": vocab["codes"],
            "assignments": [{"query_id": r["query_id"], "variant": r["variant"], "tags": r["failures"]} for r in rows],
        })
        ctx.store.insert("rule_checks", [{"run_id": ctx.run_id, **r} for r in rows])
        return {"pairs": len(rows), "flagged": sum(r["risk_flags"] != ["none"] for r in rows)}


@register
class LLMReviewer(Agent):
    """The only agent allowed to call a model, and it makes one call per run."""

    name = "llm_reviewer"
    requires = ("retrieval", "scores")
    provides = ("reviews", "llm_meta")

    def run(self, ctx: RunContext) -> dict[str, Any]:
        log_path = ctx.out_dir / "llm_calls.jsonl"
        res = run_review(ctx.inputs, ctx.artifacts["retrieval"], ctx.artifacts["scores"], ctx.cfg.llm, ctx.root, log_path)
        if res.new_log_lines:
            with log_path.open("a", encoding="utf-8", newline="\n") as fh:
                for line in res.new_log_lines:
                    fh.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
        ctx.artifacts["reviews"], ctx.artifacts["llm_meta"] = res.reviews, res.meta
        write_json(ctx.out_dir / "llm_review.json", [{**r, "review_source": res.meta["source"]} for r in res.reviews])
        write_json(ctx.out_dir / "llm_review_meta.json", res.meta)

        tmpl = PromptTemplate.load(ctx.root / ctx.cfg.llm.prompt_file)
        ctx.store.insert("prompt_templates", [{"sha256": tmpl.sha256, "name": tmpl.name, "version": tmpl.version,
                                               "template": (ctx.root / ctx.cfg.llm.prompt_file).read_text(encoding="utf-8")}])
        logged = res.new_log_lines or [{"prompt_sha256": res.meta["prompt_sha256"], "provider": res.meta["provider"],
                                        "model": res.meta["model"], "temperature": res.meta["temperature"],
                                        "mode": res.meta["source"], "attempt": 0, "valid": True, "error": res.meta["reason"]}]
        cols = {"prompt_sha256", "provider", "model", "model_reported", "temperature", "mode", "attempt", "valid", "error_code",
                "error", "finish_reason", "tokens_in", "tokens_out", "latency_ms", "request", "response"}
        ctx.store.insert("llm_requests", [{"run_id": ctx.run_id, "prompt_template_sha256": tmpl.sha256,
                                           **{k: v for k, v in line.items() if k in cols}} for line in logged])
        ctx.store.insert("llm_reviews", [
            {"run_id": ctx.run_id, "query_id": r["query_id"], "variant": v, "source": res.meta["source"],
             "faithfulness": r["faithfulness"][v], "clarity": r["clarity"][v], "overclaim_flags": r["overclaim_flags"][v],
             "is_llm_winner": r["winner"] == v, "justification": r["justification"]}
            for r in res.reviews for v in ctx.inputs.variants
        ])
        return {"source": res.meta["source"], "api_calls": sum(1 for l in res.new_log_lines if l["mode"] == "live")}


@register
class SafetyFirstAggregator(Agent):
    name = "safety_first_aggregator"
    requires = ("scores", "reviews", "llm_meta")
    provides = ("aggregate",)

    def run(self, ctx: RunContext) -> dict[str, Any]:
        queries = [q.model_dump() for q in ctx.inputs.queries]
        agg = aggregate(queries, ctx.inputs.variants, ctx.artifacts["scores"], ctx.artifacts["reviews"],
                        ctx.artifacts["llm_meta"]["source"], ctx.cfg.aggregate)
        ctx.artifacts["aggregate"] = agg
        ctx.store.insert("picks", [
            {"run_id": ctx.run_id, "query_id": p["query_id"], "risk_level": p["risk_level"], "picked_variant": p["winner"],
             "reason": p["reason"], "llm_winner": p["llm_winner"], "candidates": p["candidates"], "tradeoffs": p["tradeoffs"]}
            for p in agg["per_query"]
        ])
        return {"selected": agg["selected"], "ready_to_ship": agg["ready_to_ship"]}


@register
class MarkdownReporter(Agent):
    name = "markdown_reporter"
    requires = ("retrieval", "scores", "reviews", "llm_meta", "aggregate")
    provides = ("recommendation",)

    def run(self, ctx: RunContext) -> dict[str, Any]:
        agg, meta = ctx.artifacts["aggregate"], ctx.artifacts["llm_meta"]
        rec = render_recommendation(agg, ctx.artifacts["reviews"], meta)
        report = render_review_report([q.model_dump() for q in ctx.inputs.queries], ctx.inputs.answers,
                                      ctx.artifacts["retrieval"], ctx.artifacts["scores"], ctx.artifacts["reviews"], agg, meta)
        (ctx.out_dir / "recommendation.md").write_text(rec, encoding="utf-8", newline="\n")
        (ctx.out_dir / "review_report.md").write_text(report, encoding="utf-8", newline="\n")
        ctx.artifacts["recommendation"] = rec
        ctx.store.insert("recommendations", [{
            "run_id": ctx.run_id, "selected_variant": agg["selected"], "ready_to_ship": agg["ready_to_ship"],
            "ship_blockers": agg["ship_blockers"], "ranking": agg["ranking"], "tally": agg["tally"], "markdown": rec,
        }])
        return {"files": ["recommendation.md", "review_report.md"]}
