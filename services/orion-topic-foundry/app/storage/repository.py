from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID, uuid4

from psycopg2.extras import Json, RealDictCursor, execute_values

from app.models import DatasetSpec, ModelCreateRequest, RunRecord, SegmentRecord
from app.storage.ddl import (
    BOUNDARY_CACHE_DDL,
    CONVERSATIONS_DDL,
    CONVERSATION_BLOCKS_DDL,
    CONVERSATION_OVERRIDES_DDL,
    CONVERSATION_ROLLUPS_DDL,
    DATASETS_DDL,
    DRIFT_DDL,
    EVENTS_DDL,
    EDGES_DDL,
    MODEL_EVENTS_DDL,
    MODELS_DDL,
    RUNS_DDL,
    SEGMENTS_DDL,
    TOPICS_DDL,
    WINDOW_FILTERS_DDL,
)
from app.storage.pg import pg_conn


logger = logging.getLogger("orion-topic-foundry.repository")


def ensure_tables() -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DATASETS_DDL)
            cur.execute(MODELS_DDL)
            cur.execute(RUNS_DDL)
            cur.execute(SEGMENTS_DDL)
            cur.execute(BOUNDARY_CACHE_DDL)
            cur.execute(MODEL_EVENTS_DDL)
            cur.execute(DRIFT_DDL)
            cur.execute(EVENTS_DDL)
            cur.execute(EDGES_DDL)
            cur.execute(CONVERSATIONS_DDL)
            cur.execute(CONVERSATION_BLOCKS_DDL)
            cur.execute(CONVERSATION_OVERRIDES_DDL)
            cur.execute(TOPICS_DDL)
            cur.execute(WINDOW_FILTERS_DDL)
            cur.execute(CONVERSATION_ROLLUPS_DDL)
            cur.execute("ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT EXISTS spec_hash VARCHAR")
            cur.execute("ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT EXISTS stage VARCHAR")
            cur.execute("ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT EXISTS run_scope VARCHAR")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_topic_foundry_runs_spec_hash ON topic_foundry_runs (spec_hash)"
            )
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS title TEXT")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS aspects JSONB")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS sentiment JSONB")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS meaning JSONB")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS enrichment JSONB")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS enrichment_version TEXT")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS snippet TEXT")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS chars INTEGER")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS row_ids_count INTEGER")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS start_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE topic_foundry_segments ADD COLUMN IF NOT EXISTS end_at TIMESTAMPTZ")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_aspects ON topic_foundry_segments USING GIN (aspects)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id_created_at "
                "ON topic_foundry_segments (run_id, created_at)"
            )
            cur.execute("ALTER TABLE topic_foundry_models ADD COLUMN IF NOT EXISTS enrichment_spec JSONB")
            cur.execute("ALTER TABLE topic_foundry_models ADD COLUMN IF NOT EXISTS model_meta JSONB")
            cur.execute("ALTER TABLE topic_foundry_datasets ADD COLUMN IF NOT EXISTS boundary_column VARCHAR")
            cur.execute("ALTER TABLE topic_foundry_datasets ADD COLUMN IF NOT EXISTS boundary_strategy VARCHAR")
            cur.execute("ALTER TABLE topic_foundry_datasets ADD COLUMN IF NOT EXISTS timezone VARCHAR")
            cur.execute("UPDATE topic_foundry_datasets SET timezone = 'UTC' WHERE timezone IS NULL")
            cur.execute("ALTER TABLE topic_foundry_datasets ALTER COLUMN timezone SET DEFAULT 'UTC'")
            cur.execute("ALTER TABLE topic_foundry_datasets ALTER COLUMN timezone SET NOT NULL")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS title TEXT")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS aspects JSONB")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS sentiment JSONB")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS meaning JSONB")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS enrichment JSONB")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE topic_foundry_topics ADD COLUMN IF NOT EXISTS enrichment_version TEXT")


def create_dataset(dataset: DatasetSpec) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_datasets (
                    dataset_id, name, source_table, id_column, time_column, text_columns, timezone,
                    boundary_column, boundary_strategy, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(dataset.dataset_id),
                    dataset.name,
                    dataset.source_table,
                    dataset.id_column,
                    dataset.time_column,
                    Json(dataset.text_columns),
                    dataset.timezone or "UTC",
                    dataset.boundary_column,
                    dataset.boundary_strategy,
                    dataset.created_at,
                ),
            )


def update_dataset(dataset: DatasetSpec) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE topic_foundry_datasets
                SET name = %s,
                    source_table = %s,
                    id_column = %s,
                    time_column = %s,
                    text_columns = %s,
                    timezone = %s,
                    boundary_column = %s,
                    boundary_strategy = %s
                WHERE dataset_id = %s
                """,
                (
                    dataset.name,
                    dataset.source_table,
                    dataset.id_column,
                    dataset.time_column,
                    Json(dataset.text_columns),
                    dataset.timezone or "UTC",
                    dataset.boundary_column,
                    dataset.boundary_strategy,
                    str(dataset.dataset_id),
                ),
            )


def fetch_dataset(dataset_id: UUID) -> Optional[DatasetSpec]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_datasets WHERE dataset_id = %s",
                (str(dataset_id),),
            )
            row = cur.fetchone()
    if not row:
        return None
    return DatasetSpec(
        dataset_id=UUID(row["dataset_id"]),
        name=row["name"],
        source_table=row["source_table"],
        id_column=row["id_column"],
        time_column=row["time_column"],
        text_columns=row["text_columns"],
        timezone=row.get("timezone") or "UTC",
        boundary_column=row.get("boundary_column"),
        boundary_strategy=row.get("boundary_strategy"),
        created_at=row["created_at"],
    )


def list_datasets(*, limit: int = 200) -> List[DatasetSpec]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_datasets ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall() or []
    return [
        DatasetSpec(
            dataset_id=UUID(row["dataset_id"]),
            name=row["name"],
            source_table=row["source_table"],
            id_column=row["id_column"],
            time_column=row["time_column"],
            text_columns=row["text_columns"],
            timezone=row.get("timezone") or "UTC",
            boundary_column=row.get("boundary_column"),
            boundary_strategy=row.get("boundary_strategy"),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def create_model(model_id: UUID, request: ModelCreateRequest, created_at: datetime) -> None:
    enrichment_spec = request.enrichment_spec.model_dump(mode="json") if request.enrichment_spec else None
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_models (
                    model_id, name, version, stage, dataset_id, model_spec, windowing_spec, enrichment_spec, model_meta, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(model_id),
                    request.name,
                    request.version,
                    request.stage,
                    str(request.dataset_id),
                    Json(request.model_spec.model_dump(mode="json")),
                    Json(request.windowing_spec.model_dump(mode="json")),
                    Json(enrichment_spec) if enrichment_spec is not None else None,
                    Json(request.model_meta) if request.model_meta is not None else None,
                    Json(request.metadata),
                    created_at,
                ),
            )


def fetch_model_versions(name: str) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT model_id, name, version, stage, created_at
                FROM topic_foundry_models
                WHERE name = %s
                ORDER BY created_at DESC
                """,
                (name,),
            )
            rows = cur.fetchall() or []
    return rows


def fetch_model(model_id: UUID) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_models WHERE model_id = %s",
                (str(model_id),),
            )
            return cur.fetchone()


def list_models(*, name: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    query = "SELECT * FROM topic_foundry_models"
    params: List[Any] = []
    if name:
        query += " WHERE name = %s"
        params.append(name)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def fetch_active_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM topic_foundry_models
                WHERE name = %s AND stage = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            )
            return cur.fetchone()


def create_run(run: RunRecord) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_runs (
                    run_id, model_id, dataset_id, specs, spec_hash, status, stage, run_scope, stats, artifact_paths,
                    created_at, started_at, completed_at, error
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(run.run_id),
                    str(run.model_id),
                    str(run.dataset_id),
                    Json(run.specs.model_dump(mode="json")),
                    run.spec_hash,
                    run.status,
                    run.stage,
                    run.run_scope,
                    Json(run.stats),
                    Json(run.artifact_paths),
                    run.created_at,
                    run.started_at,
                    run.completed_at,
                    run.error,
                ),
            )


def update_run(run: RunRecord) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE topic_foundry_runs
                SET status = %s,
                    stage = %s,
                    run_scope = %s,
                    stats = %s,
                    artifact_paths = %s,
                    started_at = %s,
                    completed_at = %s,
                    error = %s
                WHERE run_id = %s
                """,
                (
                    run.status,
                    run.stage,
                    run.run_scope,
                    Json(run.stats),
                    Json(run.artifact_paths),
                    run.started_at,
                    run.completed_at,
                    run.error,
                    str(run.run_id),
                ),
            )


def fetch_run(run_id: UUID) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM topic_foundry_runs WHERE run_id = %s", (str(run_id),))
            return cur.fetchone()


def fetch_run_by_spec_hash(spec_hash: str) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM topic_foundry_runs
                WHERE spec_hash = %s AND status IN ('queued', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (spec_hash,),
            )
            return cur.fetchone()


def fetch_latest_completed_run(model_id: UUID) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM topic_foundry_runs
                WHERE model_id = %s AND status = 'complete'
                ORDER BY completed_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """,
                (str(model_id),),
            )
            return cur.fetchone()


def fetch_latest_completed_run_by_scope(model_id: UUID, scope: str) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM topic_foundry_runs
                WHERE model_id = %s AND status = 'complete' AND run_scope = %s
                ORDER BY completed_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """,
                (str(model_id), scope),
            )
            return cur.fetchone()


def list_runs(*, limit: int = 50) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall() or []


def list_runs_paginated(
    *,
    limit: int,
    offset: int,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    model_name: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[int]]:
    filters: List[str] = []
    params: List[Any] = []
    if status:
        filters.append("r.status = %s")
        params.append(status)
    if stage:
        filters.append("r.stage = %s")
        params.append(stage)
    if model_name:
        filters.append("m.name = %s")
        params.append(model_name)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            r.*,
            m.name AS model_name,
            m.version AS model_version,
            m.stage AS model_stage,
            d.name AS dataset_name,
            d.source_table AS dataset_source_table
        FROM topic_foundry_runs r
        JOIN topic_foundry_models m ON r.model_id = m.model_id
        JOIN topic_foundry_datasets d ON r.dataset_id = d.dataset_id
        {where_clause}
        ORDER BY r.created_at DESC
        LIMIT %s OFFSET %s
    """
    count_query = f"""
        SELECT COUNT(*) AS total
        FROM topic_foundry_runs r
        JOIN topic_foundry_models m ON r.model_id = m.model_id
        JOIN topic_foundry_datasets d ON r.dataset_id = d.dataset_id
        {where_clause}
    """
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params + [limit, offset])
            rows = cur.fetchall() or []
            cur.execute(count_query, params)
            total_row = cur.fetchone()
            total = int(total_row["total"]) if total_row else None
    return rows, total


def insert_segments(segments: Iterable[SegmentRecord]) -> None:
    rows = [
        (
            str(seg.segment_id),
            str(seg.run_id),
            seg.size,
            Json(seg.provenance),
            seg.label,
            seg.topic_id,
            seg.topic_prob,
            seg.is_outlier,
            seg.title,
            Json(seg.aspects) if seg.aspects is not None else None,
            Json(seg.sentiment) if seg.sentiment is not None else None,
            Json(seg.meaning) if seg.meaning is not None else None,
            Json(seg.enrichment) if seg.enrichment is not None else None,
            seg.enriched_at,
            seg.enrichment_version,
            seg.snippet,
            seg.chars,
            seg.row_ids_count,
            seg.start_at,
            seg.end_at,
            seg.created_at,
        )
        for seg in segments
    ]
    if not rows:
        return
    with pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO topic_foundry_segments (
                    segment_id, run_id, size, provenance, label, topic_id, topic_prob, is_outlier, title, aspects,
                    sentiment, meaning, enrichment, enriched_at, enrichment_version, snippet, chars, row_ids_count,
                    start_at, end_at, created_at
                ) VALUES %s
                """,
                rows,
            )


def fetch_segments(
    run_id: UUID,
    *,
    aspect: Optional[str] = None,
    has_enrichment: Optional[bool] = None,
    q: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sort_fields = {
        "start_at": "start_at",
        "end_at": "end_at",
        "size": "size",
        "friction": "COALESCE((sentiment->>'friction')::float, 0)",
        "valence": "COALESCE((sentiment->>'valence')::float, 0)",
        "created_at": "created_at",
    }
    order_field = sort_fields.get(sort_by, "created_at")
    order_dir = "ASC" if sort_dir.lower() == "asc" else "DESC"
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM topic_foundry_segments WHERE run_id = %s"
            params: List[Any] = [str(run_id)]
            if has_enrichment is True:
                query += " AND enriched_at IS NOT NULL"
            elif has_enrichment is False:
                query += " AND enriched_at IS NULL"
            if aspect:
                query += " AND aspects @> %s::jsonb"
                params.append(json.dumps([aspect]))
            if q:
                query += " AND (title ILIKE %s OR aspects::text ILIKE %s OR snippet ILIKE %s)"
                like = f"%{q}%"
                params.extend([like, like, like])
            query += f" ORDER BY {order_field} {order_dir} NULLS LAST"
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            if offset:
                query += " OFFSET %s"
                params.append(offset)
            cur.execute(query, params)
            return cur.fetchall() or []


def count_segments(
    run_id: UUID,
    *,
    aspect: Optional[str] = None,
    has_enrichment: Optional[bool] = None,
    q: Optional[str] = None,
) -> int:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            query = "SELECT COUNT(*) FROM topic_foundry_segments WHERE run_id = %s"
            params: List[Any] = [str(run_id)]
            if has_enrichment is True:
                query += " AND enriched_at IS NOT NULL"
            elif has_enrichment is False:
                query += " AND enriched_at IS NULL"
            if aspect:
                query += " AND aspects @> %s::jsonb"
                params.append(json.dumps([aspect]))
            if q:
                query += " AND (title ILIKE %s OR aspects::text ILIKE %s OR snippet ILIKE %s)"
                like = f"%{q}%"
                params.extend([like, like, like])
            cur.execute(query, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0


def list_topics(
    run_id: UUID,
    *,
    limit: int = 200,
    offset: int = 0,
    scope: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[int]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT topic_id, count, scope, parent_topic_id
                FROM topic_foundry_topics
                WHERE run_id = %s
            """
            params: List[Any] = [str(run_id)]
            if scope:
                query += " AND scope = %s"
                params.append(scope)
            query += " ORDER BY count DESC NULLS LAST LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(query, params)
            rows = cur.fetchall() or []
            if rows:
                count_query = "SELECT COUNT(*) AS total FROM topic_foundry_topics WHERE run_id = %s"
                count_params: List[Any] = [str(run_id)]
                if scope:
                    count_query += " AND scope = %s"
                    count_params.append(scope)
                cur.execute(count_query, count_params)
                total_row = cur.fetchone()
                total = int(total_row["total"]) if total_row else None
            else:
                cur.execute(
                    """
                    SELECT topic_id,
                           COUNT(*) AS count,
                           SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) AS outliers
                    FROM topic_foundry_segments
                    WHERE run_id = %s
                    GROUP BY topic_id
                    ORDER BY count DESC
                    LIMIT %s OFFSET %s
                    """,
                    (str(run_id), limit, offset),
                )
                rows = cur.fetchall() or []
                cur.execute(
                    "SELECT COUNT(DISTINCT topic_id) AS total FROM topic_foundry_segments WHERE run_id = %s",
                    (str(run_id),),
                )
                total_row = cur.fetchone()
                total = int(total_row["total"]) if total_row else None
    return rows, total


def insert_topics(run_id: UUID, topics: List[Dict[str, Any]]) -> None:
    if not topics:
        return
    rows = [
        (
            str(run_id),
            int(topic["topic_id"]),
            topic.get("scope") or "macro",
            topic.get("parent_topic_id"),
            Json(topic.get("centroid")),
            topic.get("count"),
            topic.get("label"),
        )
        for topic in topics
    ]
    with pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO topic_foundry_topics (
                    run_id, topic_id, scope, parent_topic_id, centroid, count, label
                ) VALUES %s
                ON CONFLICT (run_id, topic_id, scope) DO UPDATE SET
                    parent_topic_id = EXCLUDED.parent_topic_id,
                    centroid = EXCLUDED.centroid,
                    count = EXCLUDED.count,
                    label = EXCLUDED.label
                """,
                rows,
            )


def fetch_topics(run_id: UUID, scope: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM topic_foundry_topics WHERE run_id = %s"
    params: List[Any] = [str(run_id)]
    if scope:
        query += " AND scope = %s"
        params.append(scope)
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def update_topic_enrichment(
    run_id: UUID,
    topic_id: int,
    scope: str,
    *,
    enrichment: Dict[str, Any],
    enrichment_version: str,
) -> None:
    enriched_at = utc_now()
    enrichment["enriched_at"] = enriched_at.isoformat()
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE topic_foundry_topics
                SET title = %s,
                    aspects = %s,
                    sentiment = %s,
                    meaning = %s,
                    enrichment = %s,
                    enriched_at = %s,
                    enrichment_version = %s
                WHERE run_id = %s AND topic_id = %s AND scope = %s
                """,
                (
                    enrichment.get("title"),
                    Json(enrichment.get("aspects")),
                    Json(enrichment.get("sentiment")),
                    Json(enrichment.get("meaning")),
                    Json(enrichment),
                    enriched_at,
                    enrichment_version,
                    str(run_id),
                    int(topic_id),
                    scope,
                ),
            )


def insert_window_filters(run_id: Optional[UUID], filters: List[Dict[str, Any]]) -> None:
    if not filters:
        return
    rows = [
        (
            str(uuid4()),
            str(run_id) if run_id else None,
            f.get("segment_id"),
            f.get("policy") or "keep",
            Json(f.get("decision") or {}),
        )
        for f in filters
    ]
    with pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO topic_foundry_window_filters (
                    filter_id, run_id, segment_id, policy, decision
                ) VALUES %s
                """,
                rows,
            )


def upsert_conversation_rollups(run_id: UUID, rollups: Dict[str, Any]) -> None:
    if not rollups:
        return
    rows = [
        (conversation_id, str(run_id), Json(payload))
        for conversation_id, payload in rollups.items()
    ]
    with pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO topic_foundry_conversation_rollups (
                    conversation_id, run_id, payload
                ) VALUES %s
                ON CONFLICT (conversation_id) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    payload = EXCLUDED.payload
                """,
                rows,
            )


def fetch_topic_segments(
    run_id: UUID,
    topic_id: int,
    *,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM topic_foundry_segments WHERE run_id = %s AND topic_id = %s"
            params: List[Any] = [str(run_id), topic_id]
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(query, params)
            return cur.fetchall() or []


def list_aspect_counts(run_id: UUID) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COALESCE(a.aspect, '(none)') AS key, COUNT(*) AS count
                FROM topic_foundry_segments s
                LEFT JOIN LATERAL (
                    SELECT jsonb_array_elements_text(COALESCE(s.aspects, '[]'::jsonb)) AS aspect
                ) AS a ON TRUE
                WHERE s.run_id = %s AND s.aspects IS NOT NULL
                GROUP BY key
                ORDER BY count DESC
                """,
                (str(run_id),),
            )
            return cur.fetchall() or []


def segment_facets(
    run_id: UUID,
    *,
    aspect: Optional[str] = None,
    has_enrichment: Optional[bool] = None,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    base_query = "FROM topic_foundry_segments s"
    where_clauses = ["s.run_id = %s"]
    params: List[Any] = [str(run_id)]
    if has_enrichment is True:
        where_clauses.append("s.enriched_at IS NOT NULL")
    elif has_enrichment is False:
        where_clauses.append("s.enriched_at IS NULL")
    if aspect:
        where_clauses.append("s.aspects @> %s::jsonb")
        params.append(json.dumps([aspect]))
    if q:
        where_clauses.append("(s.title ILIKE %s OR s.aspects::text ILIKE %s OR s.snippet ILIKE %s)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where_clause_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COALESCE(a.aspect, '(none)') AS key, COUNT(*) AS count
                {base_query}
                LEFT JOIN LATERAL (
                    SELECT jsonb_array_elements_text(COALESCE(s.aspects, '[]'::jsonb)) AS aspect
                ) AS a ON TRUE
                {where_clause_sql}
                GROUP BY key
                ORDER BY count DESC
                """,
                params,
            )
            aspects = [dict(row) for row in cur.fetchall() or []]

            cur.execute(
                f"""
                SELECT (s.meaning->>'intent') AS key, COUNT(*) AS count
                {base_query}
                {where_clause_sql} AND s.meaning ? 'intent'
                GROUP BY key
                ORDER BY count DESC
                """,
                params,
            )
            intents = [dict(row) for row in cur.fetchall() or []]

            cur.execute(
                f"""
                SELECT
                    CASE
                        WHEN COALESCE((s.sentiment->>'friction')::float, 0) <= 0.3 THEN '0-0.3'
                        WHEN COALESCE((s.sentiment->>'friction')::float, 0) <= 0.7 THEN '0.3-0.7'
                        ELSE '0.7-1.0'
                    END AS key,
                    COUNT(*) AS count
                {base_query}
                {where_clause_sql}
                GROUP BY key
                ORDER BY key
                """,
                params,
            )
            friction = [dict(row) for row in cur.fetchall() or []]

            cur.execute(
                f"SELECT COUNT(*) AS total, COUNT(enriched_at) AS enriched {base_query} {where_clause_sql}",
                params,
            )
            totals_row = cur.fetchone() or {}

    return {
        "aspects": aspects,
        "intents": intents,
        "friction_buckets": friction,
        "totals": {
            "segments": int(totals_row.get("total") or 0),
            "enriched": int(totals_row.get("enriched") or 0),
        },
    }


def fetch_segment(segment_id: UUID) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_segments WHERE segment_id = %s",
                (str(segment_id),),
            )
            return cur.fetchone()


def update_segment_enrichment(segment_id: UUID, *, enrichment: Dict[str, Any], enrichment_version: str) -> None:
    enriched_at = utc_now()
    enrichment["enriched_at"] = enriched_at.isoformat()
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE topic_foundry_segments
                SET title = %s,
                    aspects = %s,
                    sentiment = %s,
                    meaning = %s,
                    enrichment = %s,
                    enriched_at = %s,
                    enrichment_version = %s
                WHERE segment_id = %s
                """,
                (
                    enrichment.get("title"),
                    Json(enrichment.get("aspects")),
                    Json(enrichment.get("sentiment")),
                    Json(enrichment.get("meaning")),
                    Json(enrichment),
                    enriched_at,
                    enrichment_version,
                    str(segment_id),
                ),
            )


def fetch_boundary_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_boundary_cache WHERE cache_key = %s",
                (cache_key,),
            )
            return cur.fetchone()


def insert_boundary_cache(
    *,
    cache_key: str,
    run_id: Optional[UUID],
    spec_hash: Optional[str],
    dataset_id: Optional[UUID],
    model_id: Optional[UUID],
    boundary_index: int,
    context_hash: str,
    decision: Dict[str, Any],
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_boundary_cache (
                    cache_key, run_id, spec_hash, dataset_id, model_id,
                    boundary_index, context_hash, decision
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                (
                    cache_key,
                    str(run_id) if run_id else None,
                    spec_hash,
                    str(dataset_id) if dataset_id else None,
                    str(model_id) if model_id else None,
                    boundary_index,
                    context_hash,
                    Json(decision),
                ),
            )


def insert_drift_record(
    *,
    drift_id: UUID,
    model_id: UUID,
    window_start: datetime,
    window_end: datetime,
    js_divergence: float,
    outlier_pct: float,
    threshold_js: Optional[float],
    threshold_outlier: Optional[float],
    topic_shares: Dict[str, Any],
    created_at: datetime,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_drift (
                    drift_id, model_id, window_start, window_end, js_divergence, outlier_pct,
                    threshold_js, threshold_outlier, topic_shares, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(drift_id),
                    str(model_id),
                    window_start,
                    window_end,
                    js_divergence,
                    outlier_pct,
                    threshold_js,
                    threshold_outlier,
                    Json(topic_shares),
                    created_at,
                ),
            )


def list_drift_records(model_name: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT drift.*
                FROM topic_foundry_drift drift
                JOIN topic_foundry_models model ON drift.model_id = model.model_id
                WHERE model.name = %s
                ORDER BY drift.created_at DESC
                LIMIT %s
                """,
                (model_name, limit),
            )
            return cur.fetchall() or []


def replace_edges_for_run(*, run_id: UUID, edges: List[Dict[str, Any]]) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM topic_foundry_edges
                WHERE segment_id IN (
                    SELECT segment_id FROM topic_foundry_segments WHERE run_id = %s
                )
                """,
                (str(run_id),),
            )
            for edge in edges:
                cur.execute(
                    """
                    INSERT INTO topic_foundry_edges (
                        edge_id, segment_id, subject, predicate, object, confidence, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(edge["edge_id"]),
                        str(edge["segment_id"]),
                        edge["subject"],
                        edge["predicate"],
                        edge["object"],
                        edge["confidence"],
                        edge["created_at"],
                    ),
                )


def list_edges(run_id: UUID, *, limit: int = 200) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT edge.*
                FROM topic_foundry_edges edge
                JOIN topic_foundry_segments seg ON edge.segment_id = seg.segment_id
                WHERE seg.run_id = %s
                ORDER BY edge.created_at DESC
                LIMIT %s
                """,
                (str(run_id), limit),
            )
            return cur.fetchall() or []


def list_edges_filtered(
    run_id: UUID,
    *,
    q: Optional[str],
    predicate: Optional[str],
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    query = """
        SELECT edge.*
        FROM topic_foundry_edges edge
        JOIN topic_foundry_segments seg ON edge.segment_id = seg.segment_id
        WHERE seg.run_id = %s
    """
    params: List[Any] = [str(run_id)]
    if predicate:
        query += " AND edge.predicate = %s"
        params.append(predicate)
    if q:
        query += " AND (edge.subject ILIKE %s OR edge.object ILIKE %s)"
        like = f"%{q}%"
        params.extend([like, like])
    query += " ORDER BY edge.created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def create_event(
    *,
    event_id: UUID,
    kind: str,
    run_id: Optional[UUID],
    model_id: Optional[UUID],
    drift_id: Optional[UUID],
    payload: Optional[Dict[str, Any]],
    bus_status: Optional[str],
    bus_error: Optional[str],
    created_at: datetime,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_events (
                    event_id, kind, run_id, model_id, drift_id, payload, bus_status, bus_error, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(event_id),
                    kind,
                    str(run_id) if run_id else None,
                    str(model_id) if model_id else None,
                    str(drift_id) if drift_id else None,
                    Json(payload) if payload is not None else None,
                    bus_status,
                    bus_error,
                    created_at,
                ),
            )


def list_events(
    *,
    limit: int = 100,
    offset: int = 0,
    kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM topic_foundry_events"
    params: List[Any] = []
    if kind:
        query += " WHERE kind = %s"
        params.append(kind)
    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def create_model_event(
    *,
    event_id: UUID,
    model_id: UUID,
    name: str,
    version: str,
    from_stage: Optional[str],
    to_stage: str,
    reason: Optional[str],
    created_at: datetime,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_model_events (
                    event_id, model_id, name, version, from_stage, to_stage, reason, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(event_id),
                    str(model_id),
                    name,
                    version,
                    from_stage,
                    to_stage,
                    reason,
                    created_at,
                ),
            )


def promote_model_stage(
    *,
    model_id: UUID,
    to_stage: str,
    reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM topic_foundry_models WHERE model_id = %s FOR UPDATE", (str(model_id),))
            model = cur.fetchone()
            if not model:
                return None
            from_stage = model.get("stage")
            name = model["name"]
            version = model["version"]
            now = utc_now()

            if to_stage == "active":
                cur.execute(
                    """
                    SELECT model_id, name, version, stage FROM topic_foundry_models
                    WHERE name = %s AND stage = 'active' AND model_id <> %s
                    FOR UPDATE
                    """,
                    (name, str(model_id)),
                )
                active_rows = cur.fetchall() or []
                for row in active_rows:
                    cur.execute(
                        "UPDATE topic_foundry_models SET stage = %s WHERE model_id = %s",
                        ("candidate", row["model_id"]),
                    )
                    cur.execute(
                        """
                        INSERT INTO topic_foundry_model_events (
                            event_id, model_id, name, version, from_stage, to_stage, reason, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid4()),
                            str(row["model_id"]),
                            row["name"],
                            row["version"],
                            row.get("stage"),
                            "candidate",
                            f"Demoted due to promotion of {name}:{version} to active.",
                            now,
                        ),
                    )

            if from_stage != to_stage:
                cur.execute(
                    "UPDATE topic_foundry_models SET stage = %s WHERE model_id = %s",
                    (to_stage, str(model_id)),
                )
                cur.execute(
                    """
                    INSERT INTO topic_foundry_model_events (
                        event_id, model_id, name, version, from_stage, to_stage, reason, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid4()),
                        str(model_id),
                        name,
                        version,
                        from_stage,
                        to_stage,
                        reason,
                        now,
                    ),
                )
            cur.execute("SELECT * FROM topic_foundry_models WHERE model_id = %s", (str(model_id),))
            return cur.fetchone()
def replace_conversations(dataset_id: UUID, conversations: List[Dict[str, Any]]) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM topic_foundry_conversation_blocks WHERE conversation_id IN (SELECT conversation_id FROM topic_foundry_conversations WHERE dataset_id = %s)", (str(dataset_id),))
            cur.execute("DELETE FROM topic_foundry_conversations WHERE dataset_id = %s", (str(dataset_id),))
            for convo in conversations:
                cur.execute(
                    """
                    INSERT INTO topic_foundry_conversations (
                        conversation_id, dataset_id, observed_start_at, observed_end_at, block_count, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(convo["conversation_id"]),
                        str(dataset_id),
                        convo.get("observed_start_at"),
                        convo.get("observed_end_at"),
                        convo.get("block_count", 0),
                        convo.get("created_at"),
                    ),
                )
                for block in convo.get("blocks", []):
                    cur.execute(
                        """
                        INSERT INTO topic_foundry_conversation_blocks (
                            conversation_id, block_index, row_ids, timestamps, role_summary, text_snippet
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(convo["conversation_id"]),
                            block["block_index"],
                            Json(block["row_ids"]),
                            Json(block["timestamps"]),
                            block.get("role_summary"),
                            block.get("text_snippet"),
                        ),
                    )


def list_conversations(
    dataset_id: UUID,
    *,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM topic_foundry_conversations WHERE dataset_id = %s"
    params: List[Any] = [str(dataset_id)]
    if start_at:
        query += " AND observed_end_at >= %s"
        params.append(start_at)
    if end_at:
        query += " AND observed_start_at <= %s"
        params.append(end_at)
    query += " ORDER BY observed_start_at ASC LIMIT %s"
    params.append(limit)
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def fetch_conversation(conversation_id: UUID) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_conversations WHERE conversation_id = %s",
                (str(conversation_id),),
            )
            convo = cur.fetchone()
            if not convo:
                return None
            cur.execute(
                "SELECT * FROM topic_foundry_conversation_blocks WHERE conversation_id = %s ORDER BY block_index",
                (str(conversation_id),),
            )
            convo["blocks"] = cur.fetchall() or []
            return convo


def create_conversation_override(
    *,
    override_id: UUID,
    dataset_id: UUID,
    kind: str,
    payload: Dict[str, Any],
    reason: Optional[str],
    created_at: datetime,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_foundry_conversation_overrides (
                    override_id, dataset_id, kind, payload, reason, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(override_id),
                    str(dataset_id),
                    kind,
                    Json(payload),
                    reason,
                    created_at,
                ),
            )


def list_conversation_overrides(dataset_id: UUID) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_foundry_conversation_overrides WHERE dataset_id = %s ORDER BY created_at",
                (str(dataset_id),),
            )
            return cur.fetchall() or []


UTC_TZINFO = datetime.fromisoformat("1970-01-01T00:00:00+00:00").tzinfo


def utc_now() -> datetime:
    return datetime.utcnow().replace(tzinfo=UTC_TZINFO)
