"""The single LLM review: prompt, schema, providers, replay cache, retry and stub."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from .config import LLMConfig

ErrorCode = Literal["NO_KEY", "TIMEOUT", "RATE_LIMITED", "SERVER", "AUTH", "BAD_REQUEST", "TRUNCATED", "INVALID"]
RETRYABLE: frozenset[str] = frozenset({"TIMEOUT", "RATE_LIMITED", "SERVER"})


class ProviderError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class ProviderReply:
    text: str
    model_reported: str | None = None
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict, cfg: LLMConfig, api_key: str) -> ProviderReply: ...


# ---------------------------------------------------------------- prompt and schema


@dataclass
class PromptTemplate:
    name: str
    version: str
    system: str
    query_block: str
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "PromptTemplate":
        raw = path.read_bytes().replace(b"\r\n", b"\n")  # same hash on every OS
        data = tomllib.loads(raw.decode("utf-8"))
        return cls(
            name=data["name"],
            version=str(data["version"]),
            system=data["system"].strip(),
            query_block=data["query_block"].strip(),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


def label_order(query_id: str, variants: list[str], blind: bool) -> dict[str, str]:
    """Map label -> variant. Blind mode shuffles by hash so position and name carry no signal."""
    order = sorted(variants, key=lambda v: hashlib.sha256(f"{query_id}:{v}".encode()).hexdigest()) if blind else list(variants)
    return {(f"answer_{i + 1}" if blind else v): v for i, v in enumerate(order)}


def _esc(text: str) -> str:
    """Minimal escaping so answer or evidence text cannot close or forge a tag."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def build_prompt(inputs, retrieval: list[dict], scores: list[dict], tmpl: PromptTemplate, blind: bool) -> tuple[str, dict[str, dict[str, str]]]:
    by_query = {r["query_id"]: r["retrieved"] for r in retrieval}
    by_pair = {(s["query_id"], s["variant"]): s for s in scores}
    docs = inputs.docs_by_id
    blocks, labels = [], {}
    for q in inputs.queries:
        mapping = label_order(q.query_id, inputs.variants, blind)
        labels[q.query_id] = mapping
        retrieved = by_query[q.query_id]
        lines = [f"- [{d['doc_id']}] {_esc(d['title'])}: {_esc(d['text'])}" for d in retrieved]
        got = {d["doc_id"] for d in retrieved}
        if q.expected_doc_ids and not got & set(q.expected_doc_ids):
            lines += [f"- [{i}] {_esc(docs[i].title)}: {_esc(docs[i].text)} (expected doc, not retrieved)"
                      for i in q.expected_doc_ids if i in docs]
        evidence = "<evidence>\n" + "\n".join(lines) + "\n</evidence>"
        answers = "\n".join(f'<answer label="{label}">{_esc(inputs.answers[q.query_id][v])}</answer>' for label, v in mapping.items())
        checks = "\n".join(
            f"- {label}: must_include={by_pair[(q.query_id, v)]['must_include_pass']}, "
            f"must_not_claim={by_pair[(q.query_id, v)]['must_not_claim_pass']}, "
            f"grounding={by_pair[(q.query_id, v)]['grounding_score']}, "
            f"flags={by_pair[(q.query_id, v)]['risk_flags']}"
            for label, v in mapping.items()
        )
        blocks.append(
            tmpl.query_block.format(
                query_id=q.query_id, risk_level=q.risk_level, question=q.user_question,
                evidence=evidence, answers=answers, checks=checks,
            )
        )
    return "\n\n".join(blocks), labels


def response_schema(query_ids: list[str], label_sets: list[list[str]]) -> dict:
    """JSON schema for the reply, built from the data so N variants work."""
    labels = sorted({label for ls in label_sets for label in ls})
    score = {"type": "integer", "minimum": 1, "maximum": 5}
    per_label = lambda item: {"type": "object", "properties": {l: item for l in labels}, "required": labels, "additionalProperties": False}  # noqa: E731
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(query_ids),
                "maxItems": len(query_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "query_id": {"type": "string", "enum": query_ids},
                        "justification": {"type": "string", "maxLength": 600},
                        "winner": {"type": "string", "enum": [*labels, "tie"]},
                        "faithfulness": per_label(score),
                        "clarity": per_label(score),
                        "overclaim_flags": per_label({"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 3}),
                    },
                    "required": ["query_id", "justification", "winner", "faithfulness", "clarity", "overclaim_flags"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


class Review(BaseModel):
    query_id: str
    winner: str
    faithfulness: dict[str, int]
    clarity: dict[str, int]
    overclaim_flags: dict[str, list[str]]
    justification: str = Field(max_length=600)


class ReviewSet(BaseModel):
    reviews: list[Review]


def validate_reply(payload: object, labels: dict[str, dict[str, str]]) -> tuple[list[Review], list[str]]:
    """Pydantic shape check, then the semantic checks the schema cannot express."""
    try:
        parsed = ReviewSet.model_validate(payload)
    except ValidationError as exc:
        return [], [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    errors: list[str] = []
    seen: dict[str, Review] = {}
    for r in parsed.reviews:
        if r.query_id not in labels:
            errors.append(f"unknown query_id {r.query_id}")
            continue
        if r.query_id in seen:
            errors.append(f"duplicate review for {r.query_id}")
        seen[r.query_id] = r
        allowed = set(labels[r.query_id])
        if r.winner not in allowed | {"tie"}:
            errors.append(f"{r.query_id}: winner '{r.winner}' not in {sorted(allowed)} or 'tie'")
        for name in ("faithfulness", "clarity", "overclaim_flags"):
            if set(getattr(r, name)) != allowed:
                errors.append(f"{r.query_id}: {name} keys {sorted(getattr(r, name))} != {sorted(allowed)}")
        for name in ("faithfulness", "clarity"):
            for label, value in getattr(r, name).items():
                if not 1 <= value <= 5:
                    errors.append(f"{r.query_id}: {name}.{label}={value} outside 1-5")
        for label, flags in r.overclaim_flags.items():
            if len(flags) > 3 or any(len(f) > 200 for f in flags):
                errors.append(f"{r.query_id}: overclaim_flags.{label} too long")
    for qid in labels:
        if qid not in seen:
            errors.append(f"missing review for {qid}")
    return list(seen.values()), errors


def _deblind_text(text: str, m: dict[str, str]) -> str:
    """Replace any label the model wrote in prose with the backticked variant name."""
    swap = {l.lower(): v for l, v in m.items() if l != v}
    if not swap:
        return text
    pattern = "(?<![A-Za-z0-9_])(" + "|".join(re.escape(l) for l in sorted(swap, key=len, reverse=True)) + ")(?![A-Za-z0-9_])"
    return re.sub(pattern, lambda mo: f"`{swap[mo.group(1).lower()]}`", text, flags=re.IGNORECASE)


def unblind(reviews: list[Review], labels: dict[str, dict[str, str]]) -> list[dict]:
    out = []
    for r in sorted(reviews, key=lambda r: list(labels).index(r.query_id)):
        m = labels[r.query_id]
        out.append(
            {
                "query_id": r.query_id,
                "winner": m.get(r.winner, "tie"),
                "faithfulness": {m[k]: v for k, v in sorted(r.faithfulness.items(), key=lambda kv: m[kv[0]])},
                "clarity": {m[k]: v for k, v in sorted(r.clarity.items(), key=lambda kv: m[kv[0]])},
                "overclaim_flags": {m[k]: v for k, v in sorted(r.overclaim_flags.items(), key=lambda kv: m[kv[0]])},
                "justification": _deblind_text(r.justification, m),
            }
        )
    return out


# ---------------------------------------------------------------- stub


def stub_reviews(inputs, scores: list[dict], reason: str) -> list[dict]:
    """Deterministic placeholder derived from the rule checks. Carries no independent signal."""
    by_pair = {(s["query_id"], s["variant"]): s for s in scores}
    out = []
    for q in inputs.queries:
        faith, flags = {}, {}
        for v in inputs.variants:
            s = by_pair[(q.query_id, v)]
            f = 1 + round(4 * s["grounding_score"]) - (2 if not s["must_not_claim_pass"] else 0)
            faith[v] = max(1, min(5, f))
            flags[v] = [x for x in s["risk_flags"] if x != "none"]
        best = max(faith.values())
        top = [v for v in inputs.variants if faith[v] == best]
        out.append(
            {
                "query_id": q.query_id,
                "winner": top[0] if len(top) == 1 else "tie",
                "faithfulness": faith,
                "clarity": {v: 3 for v in inputs.variants},
                "overclaim_flags": flags,
                "justification": f"[STUB] Derived from rule checks ({reason}); not an independent review.",
            }
        )
    return out


# ---------------------------------------------------------------- providers


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        code: ErrorCode = (
            "RATE_LIMITED" if exc.code == 429 else "AUTH" if exc.code in (401, 403)
            else "SERVER" if exc.code >= 500 else "BAD_REQUEST"
        )
        raise ProviderError(code, f"HTTP {exc.code}: {detail}") from None
    except (TimeoutError, urllib.error.URLError) as exc:
        raise ProviderError("TIMEOUT", str(exc)) from None


class AnthropicProvider:
    name = "anthropic"

    def complete(self, system: str, user: str, schema: dict, cfg: LLMConfig, api_key: str) -> ProviderReply:
        body = {
            "model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
            "tools": [{"name": "submit_review", "description": "Submit the structured review.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "submit_review"},
        }
        data = _post(cfg.base_url, {"x-api-key": api_key, "anthropic-version": "2023-06-01"}, body, cfg.timeout_s)
        if data.get("stop_reason") == "max_tokens":
            raise ProviderError("TRUNCATED", "reply hit max_tokens")
        block = next((b for b in data.get("content", []) if b.get("type") == "tool_use"), None)
        text = json.dumps(block["input"], sort_keys=True) if block else json.dumps(data.get("content"))
        usage = data.get("usage", {})
        return ProviderReply(text, data.get("model"), data.get("stop_reason"), usage.get("input_tokens"), usage.get("output_tokens"))


REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _strict_schema(node: object) -> object:
    """Drop length and count keywords some strict-mode providers reject; validate_reply enforces them."""
    if isinstance(node, dict):
        return {k: _strict_schema(v) for k, v in node.items() if k not in ("maxLength", "minLength", "minItems", "maxItems")}
    if isinstance(node, list):
        return [_strict_schema(v) for v in node]
    return node


class OpenAICompatibleProvider:
    """OpenRouter, OpenAI, vLLM, Ollama: anything that speaks /chat/completions with json_schema."""

    name = "openai_compatible"

    def complete(self, system: str, user: str, schema: dict, cfg: LLMConfig, api_key: str) -> ProviderReply:
        body = {
            "model": cfg.model, "max_completion_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": _strict_schema(schema)}},
        }
        # Reasoning models (o-series, gpt-5) reject a temperature; the rest get the pinned 0.
        if not cfg.model.split("/")[-1].startswith(REASONING_PREFIXES):
            body["temperature"] = cfg.temperature
        data = _post(cfg.base_url, {"authorization": f"Bearer {api_key}"}, body, cfg.timeout_s)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ProviderError("TRUNCATED", "reply hit max_tokens")
        usage = data.get("usage", {})
        return ProviderReply(
            choice.get("message", {}).get("content") or "", data.get("model"), choice.get("finish_reason"),
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )


PROVIDERS: dict[str, Provider] = {"anthropic": AnthropicProvider(), "openai_compatible": OpenAICompatibleProvider()}


# ---------------------------------------------------------------- cache and orchestration


def effective_max_tokens(cfg: LLMConfig, n_queries: int, n_variants: int) -> int:
    """Ask for what the pairs need, never more than the configured cap."""
    return min(cfg.max_tokens, cfg.max_tokens_base + cfg.max_tokens_per_pair * n_queries * n_variants)


def effective_cfg(cfg: LLMConfig, inputs) -> LLMConfig:
    return cfg.model_copy(update={"max_tokens": effective_max_tokens(cfg, len(inputs.queries), len(inputs.variants))})


@dataclass
class Prepared:
    tmpl: PromptTemplate
    user: str
    labels: dict[str, dict[str, str]]
    schema: dict
    cfg: LLMConfig  # effective: max_tokens already scaled
    digest: str


def prepare(inputs, retrieval: list[dict], scores: list[dict], cfg: LLMConfig, root: Path) -> Prepared:
    """Everything that defines the request, shared by the run and by validation."""
    tmpl = PromptTemplate.load(root / cfg.prompt_file)
    user, labels = build_prompt(inputs, retrieval, scores, tmpl, cfg.blind_labels)
    schema = response_schema([q.query_id for q in inputs.queries], [list(m) for m in labels.values()])
    ecfg = effective_cfg(cfg, inputs)
    return Prepared(tmpl, user, labels, schema, ecfg, request_hash(tmpl.system, user, schema, ecfg))


def request_hash(system: str, user: str, schema: dict, cfg: LLMConfig) -> str:
    canonical = {
        "provider": cfg.provider, "model": cfg.model, "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens, "system": system, "user": user, "schema": schema,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class ReviewResult:
    reviews: list[dict]
    meta: dict
    new_log_lines: list[dict] = field(default_factory=list)


def run_review(inputs, retrieval, scores, cfg: LLMConfig, root: Path, log_path: Path, provider: Provider | None = None) -> ReviewResult:
    prep = prepare(inputs, retrieval, scores, cfg, root)
    tmpl, user, labels, schema, digest = prep.tmpl, prep.user, prep.labels, prep.schema, prep.digest
    cfg = prep.cfg
    meta = {
        "prompt_sha256": digest, "prompt_name": tmpl.name, "prompt_version": tmpl.version,
        "prompt_template_sha256": tmpl.sha256, "provider": cfg.provider, "model": cfg.model,
        "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "labels": labels,
    }
    log = read_log(log_path)
    base_line = {"prompt_sha256": digest, "provider": cfg.provider, "model": cfg.model, "temperature": cfg.temperature}

    def stub(mode: str, reason: str, lines: list[dict]) -> ReviewResult:
        line = {**base_line, "mode": mode, "attempt": 0, "valid": True, "error": reason, "request": None, "response": None}
        mine = [l for l in log if l.get("prompt_sha256") == digest]
        if not (mine and {k: v for k, v in mine[-1].items()} == line):
            lines = [*lines, line]
        return ReviewResult(stub_reviews(inputs, scores, reason), {**meta, "source": mode, "reason": reason}, lines)

    if cfg.mode in ("auto", "replay"):
        for entry in reversed(log):
            if entry.get("prompt_sha256") == digest and entry.get("mode") == "live" and entry.get("valid"):
                reviews, errors = validate_reply(json.loads(entry["response"]), labels)
                if not errors:
                    return ReviewResult(unblind(reviews, labels), {**meta, "source": "replay", "reason": "cached reply with the same prompt hash"})
        if cfg.mode == "replay":
            return stub("stub", "replay mode and no cached reply", [])
    if cfg.mode == "stub":
        return stub("stub", "stub mode requested", [])

    api_key = (cfg.api_key.get_secret_value() if cfg.api_key else os.environ.get(cfg.api_key_env, "")).strip()
    if not api_key:
        if cfg.mode == "live":
            raise ProviderError("NO_KEY", f"{cfg.api_key_env} is not set and llm.mode is live")
        return stub("stub", f"{cfg.api_key_env} not set", [])

    provider = provider or PROVIDERS[cfg.provider]
    lines: list[dict] = []
    prompt_user = user
    last_error = ""
    attempt = 0
    for v_try in range(cfg.validation_retries + 1):
        reply = None
        for t_try in range(cfg.transport_retries + 1):
            attempt += 1
            started = time.monotonic()
            try:
                reply = provider.complete(tmpl.system, prompt_user, schema, cfg, api_key)
                break
            except ProviderError as exc:
                last_error = str(exc)
                lines.append({**base_line, "mode": "live", "attempt": attempt, "valid": False, "error": last_error,
                              "error_code": exc.code, "latency_ms": int((time.monotonic() - started) * 1000),
                              "request": prompt_user, "response": None})
                if exc.code not in RETRYABLE or t_try == cfg.transport_retries:
                    return stub("stub_fallback", last_error, lines)
                time.sleep(min(8.0, 2.0 ** t_try))
        assert reply is not None
        try:
            payload = json.loads(reply.text)
            reviews, errors = validate_reply(payload, labels)
        except json.JSONDecodeError as exc:
            reviews, errors = [], [f"reply is not JSON: {exc}"]
        lines.append({**base_line, "mode": "live", "attempt": attempt, "valid": not errors,
                      "error": "; ".join(errors) or None, "error_code": "INVALID" if errors else None,
                      "latency_ms": int((time.monotonic() - started) * 1000), "model_reported": reply.model_reported,
                      "finish_reason": reply.finish_reason, "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out,
                      "request": prompt_user, "response": reply.text})
        if not errors:
            return ReviewResult(unblind(reviews, labels), {**meta, "source": "live", "reason": f"attempt {attempt}"}, lines)
        last_error = "; ".join(errors)
        prompt_user = f"{user}\n\nYour previous reply failed validation. Fix exactly these problems and resubmit:\n- " + "\n- ".join(errors[:20])
    return stub("stub_fallback", f"invalid reply after retry: {last_error}", lines)
