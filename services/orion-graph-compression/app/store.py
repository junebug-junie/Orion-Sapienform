from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("orion.graph-compression.store")


class CompressionStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(postgres_uri, pool_pre_ping=True)

    def ensure_tables(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stale_queue (
                    id SERIAL PRIMARY KEY,
                    region_id TEXT,
                    scope TEXT,
                    reason TEXT,
                    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    priority INT DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS compression_artifacts (
                    region_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    fuseki_graph_uri TEXT NOT NULL DEFAULT 'http://conjourney.net/graph/orion/compressions',
                    summary_kind TEXT NOT NULL,
                    salience FLOAT,
                    trust_tier TEXT,
                    compression_version TEXT,
                    generated_at TIMESTAMPTZ NOT NULL,
                    stale BOOLEAN NOT NULL DEFAULT false
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS compression_jobs (
                    job_id TEXT PRIMARY KEY,
                    region_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    llm_tokens_used INT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    error TEXT
                )
            """))

    def enqueue_stale(
        self,
        *,
        scope: str | None = None,
        region_id: str | None = None,
        reason: str,
        priority: int = 0,
    ) -> None:
        # Coalesce: skip the insert when an equivalent pending mark already exists
        # for the same (scope, region_id) target. Under normal RDF write traffic the
        # listener would otherwise enqueue a row per write, growing the queue without
        # bound (the worker re-clusters the whole scope regardless of row count).
        # IS NOT DISTINCT FROM gives NULL-safe equality (scope-wide marks have NULL
        # region_id; region marks have NULL scope).
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO stale_queue (region_id, scope, reason, queued_at, priority)"
                    " SELECT :region_id, :scope, :reason, :queued_at, :priority"
                    " WHERE NOT EXISTS ("
                    "   SELECT 1 FROM stale_queue sq"
                    "   WHERE sq.scope IS NOT DISTINCT FROM :scope"
                    "     AND sq.region_id IS NOT DISTINCT FROM :region_id"
                    " )"
                ),
                {
                    "region_id": region_id,
                    "scope": scope,
                    "reason": reason,
                    "queued_at": datetime.now(timezone.utc),
                    "priority": priority,
                },
            )

    def drain_stale_queue(self, *, batch_size: int) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT id, region_id, scope, reason, priority"
                        " FROM stale_queue"
                        " ORDER BY priority DESC, id ASC"
                        " LIMIT :batch_size"
                    ),
                    {"batch_size": batch_size},
                )
                .mappings()
                .fetchall()
            )
        return [dict(r) for r in rows]

    def delete_stale_queue_items(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM stale_queue WHERE id = ANY(:ids)"),
                {"ids": ids},
            )

    def upsert_artifact(
        self,
        *,
        region_id: str,
        scope: str,
        kind: str,
        summary_kind: str,
        salience: float,
        trust_tier: str,
        compression_version: str,
        generated_at: datetime,
        stale: bool = False,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO compression_artifacts
                        (region_id, scope, kind, summary_kind, salience, trust_tier,
                         compression_version, generated_at, stale)
                    VALUES
                        (:region_id, :scope, :kind, :summary_kind, :salience, :trust_tier,
                         :compression_version, :generated_at, :stale)
                    ON CONFLICT (region_id) DO UPDATE SET
                        scope = EXCLUDED.scope,
                        kind = EXCLUDED.kind,
                        summary_kind = EXCLUDED.summary_kind,
                        salience = EXCLUDED.salience,
                        trust_tier = EXCLUDED.trust_tier,
                        compression_version = EXCLUDED.compression_version,
                        generated_at = EXCLUDED.generated_at,
                        stale = EXCLUDED.stale
                """),
                {
                    "region_id": region_id,
                    "scope": scope,
                    "kind": kind,
                    "summary_kind": summary_kind,
                    "salience": salience,
                    "trust_tier": trust_tier,
                    "compression_version": compression_version,
                    "generated_at": generated_at,
                    "stale": stale,
                },
            )

    def list_artifacts(
        self,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM compression_artifacts"
        params: dict[str, Any] = {"limit": limit}
        if scope:
            sql += " WHERE scope = :scope"
            params["scope"] = scope
        sql += " ORDER BY generated_at DESC LIMIT :limit"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_artifact(self, region_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM compression_artifacts WHERE region_id = :id"),
                    {"id": region_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def record_job(
        self,
        *,
        job_id: str,
        region_id: str,
        status: str,
        llm_tokens_used: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO compression_jobs
                        (job_id, region_id, status, llm_tokens_used, started_at, finished_at, error)
                    VALUES
                        (:job_id, :region_id, :status, :llm_tokens_used, :started_at, :finished_at, :error)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        llm_tokens_used = EXCLUDED.llm_tokens_used,
                        finished_at = EXCLUDED.finished_at,
                        error = EXCLUDED.error
                """),
                {
                    "job_id": job_id,
                    "region_id": region_id,
                    "status": status,
                    "llm_tokens_used": llm_tokens_used,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": error,
                },
            )

    def artifact_count(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM compression_artifacts")).first()
        return int(row[0]) if row else 0

    def stale_queue_depth(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM stale_queue")).first()
        return int(row[0]) if row else 0
