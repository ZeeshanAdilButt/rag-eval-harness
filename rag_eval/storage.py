"""Run storage: Postgres when DATABASE_URL is set, SQLite otherwise, or nothing.

The JSON files in outputs/ are the contract the graders check. The database is the
queryable history: every run, every agent's claimed task, the inputs it saw, the LLM
requests it sent and what it picked. One schema, rendered for both dialects.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StorageConfig

SCHEMA_VERSION = 1

# table -> (columns, extra constraints). Types: PK, TEXT, INT, REAL, BOOL, JSON, TS.
TABLES: dict[str, tuple[list[tuple[str, str]], list[str]]] = {
    "runs": ([
        ("run_id", "TEXT PRIMARY KEY"), ("status", "TEXT"), ("started_at", "TS"), ("finished_at", "TS"),
        ("config", "JSON"), ("config_sha256", "TEXT"), ("inputs_sha256", "TEXT"), ("variants", "JSON"),
        ("llm_source", "TEXT"), ("selected_variant", "TEXT"), ("ready_to_ship", "BOOL"), ("error", "TEXT"),
    ], []),
    "agent_tasks": ([
        ("task_id", "PK"), ("run_id", "TEXT"), ("seq", "INT"), ("stage", "TEXT"), ("agent", "TEXT"),
        ("status", "TEXT"), ("claimed_by", "TEXT"), ("claimed_at", "TS"), ("finished_at", "TS"),
        ("attempts", "INT"), ("summary", "JSON"), ("error", "TEXT"),
    ], ["UNIQUE (run_id, seq)"]),
    "kb_docs": ([("run_id", "TEXT"), ("doc_id", "TEXT"), ("title", "TEXT"), ("text", "TEXT")], ["PRIMARY KEY (run_id, doc_id)"]),
    "queries": ([
        ("run_id", "TEXT"), ("query_id", "TEXT"), ("user_question", "TEXT"), ("risk_level", "TEXT"),
        ("expected_doc_ids", "JSON"), ("must_include_any", "JSON"), ("must_not_claim", "JSON"),
    ], ["PRIMARY KEY (run_id, query_id)"]),
    "candidate_answers": ([("run_id", "TEXT"), ("query_id", "TEXT"), ("variant", "TEXT"), ("answer", "TEXT")], ["PRIMARY KEY (run_id, query_id, variant)"]),
    "retrievals": ([("run_id", "TEXT"), ("query_id", "TEXT"), ("rank", "INT"), ("doc_id", "TEXT"), ("score", "REAL")], ["PRIMARY KEY (run_id, query_id, rank)"]),
    "rule_checks": ([
        ("run_id", "TEXT"), ("query_id", "TEXT"), ("variant", "TEXT"), ("retrieval_hit", "BOOL"),
        ("must_include_pass", "BOOL"), ("must_not_claim_pass", "BOOL"), ("grounding_score", "REAL"),
        ("risk_flags", "JSON"), ("failures", "JSON"), ("notes", "TEXT"),
    ], ["PRIMARY KEY (run_id, query_id, variant)"]),
    "prompt_templates": ([("sha256", "TEXT PRIMARY KEY"), ("name", "TEXT"), ("version", "TEXT"), ("template", "TEXT")], []),
    "llm_requests": ([
        ("request_id", "PK"), ("run_id", "TEXT"), ("prompt_sha256", "TEXT"), ("prompt_template_sha256", "TEXT"),
        ("provider", "TEXT"), ("model", "TEXT"), ("model_reported", "TEXT"), ("temperature", "REAL"), ("mode", "TEXT"),
        ("attempt", "INT"), ("valid", "BOOL"), ("error_code", "TEXT"), ("error", "TEXT"), ("finish_reason", "TEXT"),
        ("tokens_in", "INT"), ("tokens_out", "INT"), ("latency_ms", "INT"), ("request", "TEXT"), ("response", "TEXT"),
    ], []),
    "llm_reviews": ([
        ("run_id", "TEXT"), ("query_id", "TEXT"), ("variant", "TEXT"), ("source", "TEXT"), ("faithfulness", "INT"),
        ("clarity", "INT"), ("overclaim_flags", "JSON"), ("is_llm_winner", "BOOL"), ("justification", "TEXT"),
    ], ["PRIMARY KEY (run_id, query_id, variant)"]),
    "picks": ([
        ("run_id", "TEXT"), ("query_id", "TEXT"), ("risk_level", "TEXT"), ("picked_variant", "TEXT"), ("reason", "TEXT"),
        ("llm_winner", "TEXT"), ("candidates", "JSON"), ("tradeoffs", "JSON"),
    ], ["PRIMARY KEY (run_id, query_id)"]),
    "recommendations": ([
        ("run_id", "TEXT PRIMARY KEY"), ("selected_variant", "TEXT"), ("ready_to_ship", "BOOL"), ("ship_blockers", "JSON"),
        ("ranking", "JSON"), ("tally", "JSON"), ("markdown", "TEXT"),
    ], []),
}

_TYPES = {
    "sqlite": {"PK": "INTEGER PRIMARY KEY AUTOINCREMENT", "TEXT": "TEXT", "INT": "INTEGER", "REAL": "REAL", "BOOL": "BOOLEAN", "JSON": "TEXT", "TS": "TEXT"},
    "postgres": {"PK": "BIGSERIAL PRIMARY KEY", "TEXT": "TEXT", "INT": "INTEGER", "REAL": "DOUBLE PRECISION", "BOOL": "BOOLEAN", "JSON": "JSONB", "TS": "TIMESTAMPTZ"},
}


def ddl(dialect: str) -> list[str]:
    types = _TYPES[dialect]
    stmts = []
    for table, (cols, extra) in TABLES.items():
        parts = [f"{name} {types.get(kind, kind)}" for name, kind in cols] + extra
        stmts.append(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(parts)})")
    stmts.append("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    return stmts


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """No-op store. Subclasses persist."""

    dialect = "none"

    def migrate(self) -> None: ...
    def insert(self, table: str, rows: list[dict[str, Any]]) -> None: ...
    def update(self, table: str, key: dict[str, Any], values: dict[str, Any]) -> None: ...

    def claim(self, task_id: int, worker: str, timeout_s: int) -> bool:
        return True

    def create_task(self, run_id: str, seq: int, stage: str, agent: str) -> int:
        return seq

    def close(self) -> None: ...


class SQLStore(Store):
    def __init__(self, conn: Any, dialect: str) -> None:
        self.conn, self.dialect = conn, dialect
        self.ph = "?" if dialect == "sqlite" else "%s"
        self._json_cols = {t: {n for n, k in cols if k == "JSON"} for t, (cols, _) in TABLES.items()}

    def _adapt(self, table: str, col: str, value: Any) -> Any:
        if col in self._json_cols.get(table, ()) and value is not None:
            if self.dialect == "postgres":
                from psycopg.types.json import Jsonb
                return Jsonb(value)
            return json.dumps(value, sort_keys=True)
        return value

    def _exec(self, sql: str, params: list[Any]) -> Any:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur

    def migrate(self) -> None:
        for stmt in ddl(self.dialect):
            self._exec(stmt, [])
        cur = self._exec("SELECT version FROM schema_version", [])
        if cur.fetchone() is None:
            self._exec(f"INSERT INTO schema_version (version) VALUES ({self.ph})", [SCHEMA_VERSION])
        self.conn.commit()

    def insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            cols = list(row)
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join([self.ph] * len(cols))})"
            if table == "prompt_templates":
                sql += " ON CONFLICT (sha256) DO NOTHING"
            self._exec(sql, [self._adapt(table, c, row[c]) for c in cols])
        self.conn.commit()

    def update(self, table: str, key: dict[str, Any], values: dict[str, Any]) -> None:
        sets = ", ".join(f"{c} = {self.ph}" for c in values)
        where = " AND ".join(f"{c} = {self.ph}" for c in key)
        self._exec(f"UPDATE {table} SET {sets} WHERE {where}",
                   [self._adapt(table, c, v) for c, v in values.items()] + list(key.values()))
        self.conn.commit()

    def create_task(self, run_id: str, seq: int, stage: str, agent: str) -> int:
        sql = (f"INSERT INTO agent_tasks (run_id, seq, stage, agent, status, attempts) "
               f"VALUES ({self.ph}, {self.ph}, {self.ph}, {self.ph}, 'pending', 0)")
        params = [run_id, seq, stage, agent]
        if self.dialect == "postgres":
            cur = self._exec(sql + " RETURNING task_id", params)
            task_id = cur.fetchone()[0]
        else:
            task_id = self._exec(sql, params).lastrowid
        self.conn.commit()
        return int(task_id)

    def claim(self, task_id: int, worker: str, timeout_s: int) -> bool:
        """One conditional UPDATE: a pending task, or a running one whose claim went stale."""
        ts = now()
        if self.dialect == "postgres":
            stale = f"claimed_at < now() - make_interval(secs => {self.ph})"
        else:
            stale = f"claimed_at < strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-' || {self.ph} || ' seconds')"
        sql = (f"UPDATE agent_tasks SET status = 'running', claimed_by = {self.ph}, claimed_at = {self.ph}, "
               f"attempts = attempts + 1 WHERE task_id = {self.ph} AND "
               f"(status = 'pending' OR (status = 'running' AND {stale}))")
        cur = self._exec(sql, [worker, ts, task_id, timeout_s])
        self.conn.commit()
        return cur.rowcount == 1

    def close(self) -> None:
        self.conn.close()


def _open_sqlite(cfg: StorageConfig, root: Path) -> Store:
    path = root / cfg.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLStore(sqlite3.connect(path), "sqlite")


def _open_postgres(url: str) -> Store:
    import psycopg  # optional dependency, only needed for Postgres

    return SQLStore(psycopg.connect(url, connect_timeout=3), "postgres")


def open_store(cfg: StorageConfig, root: Path) -> Store:
    url = os.environ.get(cfg.database_url_env, "").strip()
    backend = cfg.backend
    if backend == "none":
        return Store()
    if backend == "postgres" and not url:
        raise RuntimeError(f"storage.backend is postgres but {cfg.database_url_env} is not set")
    store: Store | None = None
    if backend == "postgres" or (backend == "auto" and url):
        try:
            store = _open_postgres(url)
        except Exception as exc:  # never echo str(exc): a driver error can carry the URL
            if backend == "postgres":
                raise
            why = "psycopg is not installed" if isinstance(exc, ImportError) else f"connection failed ({type(exc).__name__})"
            print(f"warning: {cfg.database_url_env} is set but {why}; falling back to SQLite", file=sys.stderr)
    if store is None:
        store = _open_sqlite(cfg, root)
    store.migrate()
    return store
