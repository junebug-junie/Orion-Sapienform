from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from orion.schemas.self_experiments import SelfExperimentRecordV1, SelfExperimentSpecV1

from .settings import settings

_TERMINAL_DEDUPE_STATUSES = ("rejected", "discarded", "expired")
_DEDUPE_UNIQUE_INDEX = "idx_experiments_dedupe_key_active_unique"
_DEDUPE_LOOKUP_INDEX = "idx_experiments_dedupe_key"

InsertOutcome = Literal["created", "dedupe_hit"]


def _dedupe_index_sql() -> str:
    terminal = ", ".join(f"'{s}'" for s in _TERMINAL_DEDUPE_STATUSES)
    return f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {_DEDUPE_UNIQUE_INDEX}
        ON experiments(dedupe_key)
        WHERE dedupe_key IS NOT NULL
          AND dedupe_key != ''
          AND status NOT IN ({terminal})
    """


def _dedupe_active_row(conn: sqlite3.Connection, dedupe_key: str) -> sqlite3.Row | None:
    terminal = ", ".join("?" for _ in _TERMINAL_DEDUPE_STATUSES)
    return conn.execute(
        f"""
        SELECT * FROM experiments
        WHERE dedupe_key = ?
          AND status NOT IN ({terminal})
        ORDER BY created_at_utc DESC
        LIMIT 1
        """,
        (dedupe_key, *_TERMINAL_DEDUPE_STATUSES),
    ).fetchone()


def _collapse_duplicate_active_dedupe_keys(conn: sqlite3.Connection) -> None:
    """Keep newest active row per dedupe_key before adding the unique index."""
    rows = conn.execute(
        """
        SELECT id, dedupe_key, status, created_at_utc
        FROM experiments
        WHERE dedupe_key IS NOT NULL AND dedupe_key != ''
        ORDER BY dedupe_key, created_at_utc DESC
        """
    ).fetchall()
    seen_active: set[str] = set()
    to_delete: list[str] = []
    for row in rows:
        dedupe_key = str(row["dedupe_key"])
        status = str(row["status"])
        if status in _TERMINAL_DEDUPE_STATUSES:
            continue
        if dedupe_key in seen_active:
            to_delete.append(str(row["id"]))
        else:
            seen_active.add(dedupe_key)
    for experiment_id in to_delete:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))


def _ensure_dedupe_constraints(conn: sqlite3.Connection) -> None:
    _collapse_duplicate_active_dedupe_keys(conn)
    conn.execute(_dedupe_index_sql())
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_DEDUPE_LOOKUP_INDEX} ON experiments(dedupe_key)"
    )


def connect() -> sqlite3.Connection:
    db_path = Path(settings.experiments_store_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                skill_id TEXT,
                experiment_type TEXT,
                question TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                provenance_json TEXT NOT NULL,
                args_json TEXT NOT NULL,
                spec_json TEXT,
                dedupe_key TEXT,
                dispatch_attempts INTEGER NOT NULL DEFAULT 0,
                context_exec_request_json TEXT,
                context_exec_result_json TEXT,
                context_exec_run_id TEXT,
                context_exec_status TEXT,
                artifact_type TEXT,
                artifact_summary TEXT,
                proposal_id TEXT,
                proposal_status TEXT,
                attention_required INTEGER NOT NULL DEFAULT 0,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT
            )
            """
        )
        existing = _table_columns(conn, "experiments")
        migrations: list[tuple[str, str]] = [
            ("skill_id", "ALTER TABLE experiments ADD COLUMN skill_id TEXT"),
            ("experiment_type", "ALTER TABLE experiments ADD COLUMN experiment_type TEXT"),
            ("question", "ALTER TABLE experiments ADD COLUMN question TEXT"),
            ("spec_json", "ALTER TABLE experiments ADD COLUMN spec_json TEXT"),
            ("dedupe_key", "ALTER TABLE experiments ADD COLUMN dedupe_key TEXT"),
            ("dispatch_attempts", "ALTER TABLE experiments ADD COLUMN dispatch_attempts INTEGER NOT NULL DEFAULT 0"),
            ("context_exec_request_json", "ALTER TABLE experiments ADD COLUMN context_exec_request_json TEXT"),
            ("context_exec_result_json", "ALTER TABLE experiments ADD COLUMN context_exec_result_json TEXT"),
            ("context_exec_run_id", "ALTER TABLE experiments ADD COLUMN context_exec_run_id TEXT"),
            ("context_exec_status", "ALTER TABLE experiments ADD COLUMN context_exec_status TEXT"),
            ("artifact_type", "ALTER TABLE experiments ADD COLUMN artifact_type TEXT"),
            ("artifact_summary", "ALTER TABLE experiments ADD COLUMN artifact_summary TEXT"),
            ("proposal_id", "ALTER TABLE experiments ADD COLUMN proposal_id TEXT"),
            ("proposal_status", "ALTER TABLE experiments ADD COLUMN proposal_status TEXT"),
            ("attention_required", "ALTER TABLE experiments ADD COLUMN attention_required INTEGER NOT NULL DEFAULT 0"),
            ("updated_at_utc", "ALTER TABLE experiments ADD COLUMN updated_at_utc TEXT"),
            ("completed_at_utc", "ALTER TABLE experiments ADD COLUMN completed_at_utc TEXT"),
        ]
        for col, sql in migrations:
            if col not in existing:
                conn.execute(sql)
        _ensure_dedupe_constraints(conn)
        conn.commit()


def find_by_dedupe_key(dedupe_key: str) -> SelfExperimentRecordV1 | None:
    with connect() as conn:
        row = _dedupe_active_row(conn, dedupe_key)
    if row is None:
        return None
    return row_to_record(row)


def _insert_record_conn(conn: sqlite3.Connection, record: SelfExperimentRecordV1) -> None:
    spec = record.spec
    conn.execute(
        """
        INSERT INTO experiments (
            id, skill_id, experiment_type, question, status, reason,
            provenance_json, args_json, spec_json, dedupe_key, dispatch_attempts,
            context_exec_request_json, context_exec_result_json, context_exec_run_id,
            context_exec_status, artifact_type, artifact_summary, proposal_id,
            proposal_status, attention_required, created_at_utc, updated_at_utc,
            completed_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.experiment_id,
            spec.requested_skill_id,
            spec.experiment_type,
            spec.question,
            record.status,
            record.reason,
            json.dumps(spec.provenance, sort_keys=True),
            json.dumps(spec.args, sort_keys=True),
            spec.model_dump_json(),
            record.dedupe_key,
            record.dispatch_attempts,
            json.dumps(record.context_exec_request) if record.context_exec_request else None,
            json.dumps(record.artifact_payload) if record.artifact_payload else None,
            record.context_exec_run_id,
            record.context_exec_status,
            record.artifact_type,
            record.artifact_summary,
            record.proposal_id,
            record.proposal_status,
            1 if record.attention_required else 0,
            record.created_at_utc,
            record.updated_at_utc,
            record.completed_at_utc,
        ),
    )


def insert_record_dedupe_safe(record: SelfExperimentRecordV1) -> tuple[SelfExperimentRecordV1, InsertOutcome]:
    """Atomically insert or return existing active row for the same dedupe_key."""
    if not record.dedupe_key:
        with connect() as conn:
            _insert_record_conn(conn, record)
            conn.commit()
        return record, "created"

    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = _dedupe_active_row(conn, record.dedupe_key)
            if existing is not None:
                conn.commit()
                return row_to_record(existing), "dedupe_hit"
            try:
                _insert_record_conn(conn, record)
                conn.commit()
                return record, "created"
            except sqlite3.IntegrityError:
                conn.rollback()
                conn.execute("BEGIN IMMEDIATE")
                existing = _dedupe_active_row(conn, record.dedupe_key)
                if existing is None:
                    raise
                conn.commit()
                return row_to_record(existing), "dedupe_hit"
        except Exception:
            conn.rollback()
            raise


def insert_record(record: SelfExperimentRecordV1) -> None:
    insert_record_dedupe_safe(record)


def update_record(record: SelfExperimentRecordV1) -> None:
    spec = record.spec
    with connect() as conn:
        conn.execute(
            """
            UPDATE experiments SET
                skill_id = ?,
                experiment_type = ?,
                question = ?,
                status = ?,
                reason = ?,
                provenance_json = ?,
                args_json = ?,
                spec_json = ?,
                dedupe_key = ?,
                dispatch_attempts = ?,
                context_exec_request_json = ?,
                context_exec_result_json = ?,
                context_exec_run_id = ?,
                context_exec_status = ?,
                artifact_type = ?,
                artifact_summary = ?,
                proposal_id = ?,
                proposal_status = ?,
                attention_required = ?,
                updated_at_utc = ?,
                completed_at_utc = ?
            WHERE id = ?
            """,
            (
                spec.requested_skill_id,
                spec.experiment_type,
                spec.question,
                record.status,
                record.reason,
                json.dumps(spec.provenance, sort_keys=True),
                json.dumps(spec.args, sort_keys=True),
                spec.model_dump_json(),
                record.dedupe_key,
                record.dispatch_attempts,
                json.dumps(record.context_exec_request) if record.context_exec_request else None,
                json.dumps(record.artifact_payload) if record.artifact_payload else None,
                record.context_exec_run_id,
                record.context_exec_status,
                record.artifact_type,
                record.artifact_summary,
                record.proposal_id,
                record.proposal_status,
                1 if record.attention_required else 0,
                record.updated_at_utc,
                record.completed_at_utc,
                record.experiment_id,
            ),
        )
        conn.commit()


def get_record(experiment_id: str) -> SelfExperimentRecordV1 | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        return None
    return row_to_record(row)


def list_records(
    *,
    limit: int,
    status: str | None = None,
    experiment_type: str | None = None,
    source: str | None = None,
    date: str | None = None,
    correlation_id: str | None = None,
    attention_required: bool | None = None,
) -> list[SelfExperimentRecordV1]:
    where: list[str] = []
    values: list[Any] = []
    if status:
        where.append("status = ?")
        values.append(status)
    if experiment_type:
        where.append("experiment_type = ?")
        values.append(experiment_type)
    if source:
        where.append("spec_json LIKE ?")
        values.append(f'%"source": "{source}"%')
    if date:
        where.append("(spec_json LIKE ? OR provenance_json LIKE ?)")
        values.extend([f"%{date}%", f"%{date}%"])
    if correlation_id:
        where.append("(spec_json LIKE ? OR provenance_json LIKE ?)")
        values.extend([f"%{correlation_id}%", f"%{correlation_id}%"])
    if attention_required is not None:
        where.append("attention_required = ?")
        values.append(1 if attention_required else 0)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    query = f"""
        SELECT * FROM experiments
        {where_sql}
        ORDER BY created_at_utc DESC
        LIMIT ?
    """
    values.append(limit)
    with connect() as conn:
        rows = conn.execute(query, values).fetchall()
    return [row_to_record(row) for row in rows]


def row_to_record(row: sqlite3.Row) -> SelfExperimentRecordV1:
    spec_raw = row["spec_json"]
    if spec_raw:
        spec = SelfExperimentSpecV1.model_validate_json(spec_raw)
    else:
        skill_id = str(row["skill_id"] or "")
        spec = SelfExperimentSpecV1(
            experiment_id=str(row["id"]),
            experiment_type="skill_probe",
            question=str(row["question"] or f"Run read-only skill probe: {skill_id}"),
            requested_skill_id=skill_id or None,
            provenance=json.loads(row["provenance_json"] or "{}"),
            args=json.loads(row["args_json"] or "{}"),
            created_at_utc=str(row["created_at_utc"]),
        )

    ctx_req_raw = row["context_exec_request_json"]
    ctx_req = json.loads(ctx_req_raw) if ctx_req_raw else None
    artifact_raw = row["context_exec_result_json"]
    artifact_payload = json.loads(artifact_raw) if artifact_raw else None

    updated = row["updated_at_utc"] or row["created_at_utc"]
    return SelfExperimentRecordV1(
        experiment_id=str(row["id"]),
        spec=spec,
        status=row["status"],  # type: ignore[arg-type]
        reason=str(row["reason"]) if row["reason"] else None,
        dedupe_key=str(row["dedupe_key"] or ""),
        dispatch_attempts=int(row["dispatch_attempts"] or 0),
        context_exec_request=ctx_req if isinstance(ctx_req, dict) else None,
        context_exec_run_id=str(row["context_exec_run_id"]) if row["context_exec_run_id"] else None,
        context_exec_status=str(row["context_exec_status"]) if row["context_exec_status"] else None,
        artifact_type=str(row["artifact_type"]) if row["artifact_type"] else None,
        artifact_summary=str(row["artifact_summary"]) if row["artifact_summary"] else None,
        artifact_payload=artifact_payload if isinstance(artifact_payload, dict) else None,
        proposal_id=str(row["proposal_id"]) if row["proposal_id"] else None,
        proposal_status=str(row["proposal_status"]) if row["proposal_status"] else None,
        attention_required=bool(row["attention_required"]),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(updated),
        completed_at_utc=str(row["completed_at_utc"]) if row["completed_at_utc"] else None,
    )
