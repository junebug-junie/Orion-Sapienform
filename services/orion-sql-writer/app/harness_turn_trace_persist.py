"""Idempotent harness_turn_trace persistence keyed by correlation_id.

Four independent bus kinds (harness.run.v1, harness.verdict.molecule.v1,
harness.turn.outcome.v1, harness.post_turn.closure.v1) each fill in one
column of the same row, in whatever order they actually arrive -- upsert
by correlation_id, never wait for the others. Same on_conflict_do_update
idiom as app/spark_telemetry_persist.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.harness_turn_trace import HarnessTurnTraceSQL

# schema_version on the wire payload -> column this write should fill.
# Every one of the four molecules declares a distinct schema_version literal
# (orion/schemas/harness_finalize.py), so this is a real, guaranteed-present
# discriminator, not a guess.
_COLUMN_BY_SCHEMA_VERSION: dict[str, str] = {
    "harness.run.v1": "run_artifact",
    "harness.verdict.molecule.v1": "verdict_molecule",
    "harness.turn.outcome.v1": "outcome_molecule",
    "harness.post_turn.closure.v1": "closure",
}


def column_for_schema_version(schema_version: str | None) -> str | None:
    return _COLUMN_BY_SCHEMA_VERSION.get(str(schema_version or "").strip())


def upsert_harness_turn_trace(sess: Session, payload: dict[str, Any]) -> bool:
    corr_id = payload.get("correlation_id")
    column = column_for_schema_version(payload.get("schema_version"))
    if not corr_id or column is None:
        return False

    now = datetime.now(timezone.utc)
    stmt = insert(HarnessTurnTraceSQL).values(
        correlation_id=str(corr_id),
        created_at=now,
        updated_at=now,
        **{column: payload},
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[HarnessTurnTraceSQL.correlation_id],
        set_={column: payload, "updated_at": now},
    )
    sess.execute(stmt)
    sess.commit()
    return True
