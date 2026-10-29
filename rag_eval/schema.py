"""Input models and loaders. Every error names the file and the record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RiskLevel = Literal["low", "medium", "high"]


_RISK_ALIASES = {"critical": "high"}


class InputError(ValueError):
    """Raised when an input file breaks the schema."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


def _ident(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        raise ValueError("must be non-empty and contain no whitespace")
    return value


class Doc(_Strict):
    doc_id: str
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)

    _check_id = field_validator("doc_id")(_ident)


class Query(_Strict):
    query_id: str
    user_question: str = Field(min_length=1)
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_include_any: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    risk_level: RiskLevel

    _check_id = field_validator("query_id")(_ident)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        return _RISK_ALIASES.get(value, value)

    @field_validator("must_include_any", "must_not_claim")
    @classmethod
    def _no_blank_phrases(cls, value: list[str]) -> list[str]:
        if any(not p.strip() for p in value):
            raise ValueError("phrases must not be blank")
        return value


class CandidateRow(_Strict):
    query_id: str
    answers: dict[str, str]

    @field_validator("answers")
    @classmethod
    def _answers(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("no answers")
        for variant, text in value.items():
            _ident(variant)
            if not text.strip():
                raise ValueError(f"answer for variant '{variant}' is empty")
        return value


class Inputs(BaseModel):
    kb: list[Doc]
    queries: list[Query]
    answers: dict[str, dict[str, str]]  # query_id -> variant -> answer
    variants: list[str]

    @property
    def docs_by_id(self) -> dict[str, Doc]:
        return {d.doc_id: d for d in self.kb}


def _read_list(path: Path) -> list[object]:
    if not path.exists():
        raise InputError(f"{path.name}: file not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path.name}: invalid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise InputError(f"{path.name}: expected a non-empty JSON array")
    return data


def _parse(model: type[BaseModel], rows: list[object], fname: str, key: str) -> list:
    out = []
    for i, row in enumerate(rows):
        label = row.get(key, f"#{i}") if isinstance(row, dict) else f"#{i}"
        try:
            out.append(model.model_validate(row))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'record'}: {e['msg']}" for e in exc.errors()
            )
            raise InputError(f"{fname}: record {label}: {problems}") from None
    return out


def _unique(ids: list[str], fname: str, key: str) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise InputError(f"{fname}: duplicate {key} '{i}'")
        seen.add(i)


def load_inputs(data_dir: Path) -> Inputs:
    kb = _parse(Doc, _read_list(data_dir / "kb.json"), "kb.json", "doc_id")
    queries = _parse(Query, _read_list(data_dir / "queries.json"), "queries.json", "query_id")
    rows = _parse(
        CandidateRow, _read_list(data_dir / "candidate_answers.json"), "candidate_answers.json", "query_id"
    )
    _unique([d.doc_id for d in kb], "kb.json", "doc_id")
    _unique([q.query_id for q in queries], "queries.json", "query_id")
    _unique([r.query_id for r in rows], "candidate_answers.json", "query_id")

    doc_ids = {d.doc_id for d in kb}
    for q in queries:
        unknown = sorted(set(q.expected_doc_ids) - doc_ids)
        if unknown:
            raise InputError(f"queries.json: query {q.query_id}: unknown expected_doc_ids {unknown}")

    answers = {r.query_id: r.answers for r in rows}
    query_ids = [q.query_id for q in queries]
    extra = sorted(set(answers) - set(query_ids))
    if extra:
        raise InputError(f"candidate_answers.json: answers for unknown query_id(s) {extra}")
    missing = [qid for qid in query_ids if qid not in answers]
    if missing:
        raise InputError(f"candidate_answers.json: no answers for query_id(s) {missing}")

    variants = sorted(answers[query_ids[0]])
    for qid in query_ids:
        if sorted(answers[qid]) != variants:
            raise InputError(
                f"candidate_answers.json: query {qid}: variants {sorted(answers[qid])} "
                f"differ from {variants}; every query needs the same variants"
            )
    if len(variants) < 2:
        raise InputError("candidate_answers.json: need at least two variants to compare")
    return Inputs(kb=kb, queries=queries, answers=answers, variants=variants)
