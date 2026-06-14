"""Idempotent spark_telemetry persistence keyed by correlation_id."""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.spark_telemetry import SparkTelemetrySQL


def upsert_spark_telemetry(sess: Session, data: dict[str, Any]) -> bool:
    mapper = inspect(SparkTelemetrySQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
    corr_id = filtered.get("correlation_id")
    if not corr_id:
        sess.add(SparkTelemetrySQL(**filtered))
        sess.commit()
        return True

    stmt = insert(SparkTelemetrySQL).values(**filtered)
    update_cols = {
        "phi": stmt.excluded.phi,
        "novelty": stmt.excluded.novelty,
        "trace_mode": stmt.excluded.trace_mode,
        "trace_verb": stmt.excluded.trace_verb,
        "stimulus_summary": stmt.excluded.stimulus_summary,
        "timestamp": stmt.excluded.timestamp,
        "metadata_": text(
            "(COALESCE(spark_telemetry.metadata::jsonb, '{}'::jsonb) "
            "|| COALESCE(EXCLUDED.metadata::jsonb, '{}'::jsonb))::json"
        ),
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[SparkTelemetrySQL.correlation_id],
        set_=update_cols,
    )
    sess.execute(stmt)
    sess.commit()
    return True
