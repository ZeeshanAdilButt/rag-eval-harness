"""Layered, validated configuration.

Precedence, lowest to highest:
  1. defaults in these models
  2. config/eval.toml
  3. [profiles.<name>] in that file (--profile)
  4. environment variables RAG_EVAL__<SECTION>__<KEY> (for example RAG_EVAL__LLM__MODEL)
  5. --set section.key=value on the command line

Every layer is validated by pydantic, so a typo or a bad value fails at startup.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "RAG_EVAL__"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    data_dir: str = "data"
    out_dir: str = "outputs"
    taxonomy: str = "config/failure_taxonomy.json"


class RetrievalConfig(_Section):
    k: int = Field(2, ge=1)
    k1: float = Field(1.5, gt=0)
    b: float = Field(0.75, ge=0, le=1)


class ChecksConfig(_Section):
    grounding_min: float = Field(0.6, ge=0, le=1)
    sentence_support_min: float = Field(0.5, ge=0, le=1)
    sentence_min_tokens: int = Field(3, ge=1)
    relevance_min: float = Field(0.2, ge=0, le=1)
    certainty_markers: list[str] = Field(default_factory=list)
    refusal_markers: list[str] = Field(default_factory=list)


class LLMConfig(_Section):
    mode: Literal["auto", "live", "replay", "stub"] = "auto"
    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    model: str = "claude-haiku-5-5"
    temperature: float = Field(0.0, ge=0, le=1)
    max_tokens: int = Field(4000, ge=256)
    timeout_s: float = Field(60, gt=0)
    transport_retries: int = Field(1, ge=0, le=3)
    validation_retries: int = Field(1, ge=0, le=1)
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = "https://api.anthropic.com/v1/messages"
    prompt_file: str = "prompts/llm_review.toml"
    blind_labels: bool = True


class AggregateConfig(_Section):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "must_include": 0.30,
            "grounding": 0.30,
            "faithfulness": 0.20,
            "clarity": 0.10,
            "llm_winner": 0.10,
            "per_failure": 0.10,
        }
    )
    veto_flags: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "low": ["policy_violation"],
            "medium": ["policy_violation"],
            "high": ["policy_violation", "unsupported_claim", "overconfident_tone"],
        }
    )
    ship_blockers_high_risk: list[str] = Field(default_factory=lambda: ["missed_key_fact"])
    ignore_llm_weights_for_stub: bool = True


class StorageConfig(_Section):
    backend: Literal["auto", "sqlite", "postgres", "none"] = "auto"
    sqlite_path: str = "outputs/runs.sqlite"
    database_url_env: str = "DATABASE_URL"
    claim_timeout_s: int = Field(300, ge=1)


class StageConfig(_Section):
    stage: str
    agent: str


class PipelineConfig(_Section):
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(stage="retrieve", agent="bm25_retriever"),
            StageConfig(stage="check", agent="rule_checker"),
            StageConfig(stage="review", agent="llm_reviewer"),
            StageConfig(stage="aggregate", agent="safety_first_aggregator"),
            StageConfig(stage="report", agent="markdown_reporter"),
        ]
    )


class Config(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    aggregate: AggregateConfig = Field(default_factory=AggregateConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode()
        ).hexdigest()

    def scoring_view(self) -> dict[str, Any]:
        """The parts that change scores; paths, storage and LLM transport do not."""
        return self.model_dump(include={"retrieval", "checks", "aggregate"})


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_path(tree: dict, dotted: str, value: Any) -> None:
    parts = [p for p in dotted.lower().split(".") if p]
    if len(parts) < 2:
        raise ConfigError(f"override '{dotted}' must look like section.key")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(
    path: Path | None = None,
    profile: str | None = None,
    overrides: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    tree: dict[str, Any] = {}
    if path is not None and path.exists():
        with path.open("rb") as fh:
            tree = tomllib.load(fh)
    profiles = tree.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise ConfigError(f"unknown profile '{profile}'; known: {sorted(profiles)}")
        tree = _deep_merge(tree, profiles[profile])

    env = os.environ if environ is None else environ
    for key in sorted(env):
        if key.startswith(ENV_PREFIX):
            dotted = key[len(ENV_PREFIX):].replace("__", ".")
            _set_path(tree, dotted, _parse_value(env[key]))

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        _set_path(tree, dotted.strip(), _parse_value(raw.strip()))

    try:
        return Config.model_validate(tree)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"invalid config: {problems}") from None
