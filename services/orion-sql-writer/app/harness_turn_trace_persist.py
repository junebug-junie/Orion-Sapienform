"""Idempotent harness_turn_trace persistence keyed by correlation_id.

Four independent bus kinds (harness.run.v1, harness.verdict.molecule.v1,
harness.turn.outcome.v1, harness.post_turn.closure.v1) each fill in one
column of the same row, in whatever order they actually arrive -- upsert
by correlation_id, never wait for the others. Same on_conflict_do_update
idiom as app/spark_telemetry_persist.py.

One exception (#3596): a promise act points at its quote by position
(``harness_turn_trace:<corr>#promises_made[<i>]``) and Prime's hand-back links promises by
the same string, so a re-published run artifact for the same correlation id must not move
those positions. The first non-empty ``reflection.promises_made`` stored for a corr wins;
everything else in the artifact updates as before. A re-publish that brought a different
list logs ``harness_turn_trace_promises_kept_first``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import literal_column
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app import write_health
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


logger = logging.getLogger("sql-writer.harness_turn_trace")

# ON CONFLICT value for run_artifact: the incoming artifact, except that an already-stored non-empty
# reflection.promises_made is kept (first stored list wins, #3596). jsonb_set on '{reflection}' also
# creates the key when the incoming artifact has no reflection object.
_STORED_MADE = "harness_turn_trace.run_artifact #> '{reflection,promises_made}'"
# A re-publish with no reflection object (a judge failure, an artifact without one) keeps the stored reflection
# whole, so no promise list ever sits on a turn without the reflection_source that says it was judged.
_STORED_REFLECTION = "harness_turn_trace.run_artifact -> 'reflection'"
RUN_ARTIFACT_ON_CONFLICT_SQL = f"""CASE
  WHEN jsonb_typeof({_STORED_MADE}) = 'array' AND jsonb_array_length({_STORED_MADE}) > 0
  THEN jsonb_set(excluded.run_artifact, '{{reflection}}',
                 CASE WHEN jsonb_typeof(excluded.run_artifact -> 'reflection') = 'object'
                      THEN (excluded.run_artifact -> 'reflection')
                           || jsonb_build_object('promises_made', {_STORED_MADE})
                      ELSE {_STORED_REFLECTION} END)
  ELSE excluded.run_artifact
END"""


def column_for_schema_version(schema_version: str | None) -> str | None:
    return _COLUMN_BY_SCHEMA_VERSION.get(str(schema_version or "").strip())


def upsert_harness_turn_trace(sess: Session, payload: dict[str, Any]) -> bool:
    corr_id = payload.get("correlation_id")
    column = column_for_schema_version(payload.get("schema_version"))
    if not corr_id or column is None:
        # No fallback row for this reject: tell the storage-write organ it was lost.
        write_health.mark_failed("validation error for harness turn trace: missing correlation_id or unknown schema_version")
        return False

    now = datetime.now(timezone.utc)
    filled: dict[str, Any] = {column: payload}
    # Only the run artifact carries the turn's workflow label, and only newer producers set it:
    # the column is named in the statement when there is a value, so an artifact without one
    # never overwrites a stored label with NULL.
    workflow = str(payload.get("workflow") or "").strip() if column == "run_artifact" else ""
    if workflow:
        filled["workflow"] = workflow
    stmt = insert(HarnessTurnTraceSQL).values(
        correlation_id=str(corr_id),
        created_at=now,
        updated_at=now,
        **filled,
    )
    set_ = {**filled, "updated_at": now}
    if column == "run_artifact":
        set_["run_artifact"] = literal_column(RUN_ARTIFACT_ON_CONFLICT_SQL)
    stmt = stmt.on_conflict_do_update(
        index_elements=[HarnessTurnTraceSQL.correlation_id],
        set_=set_,
    )
    if column == "run_artifact":
        stmt = stmt.returning(literal_column("run_artifact #> '{reflection,promises_made}'"))
    result = sess.execute(stmt)
    if column == "run_artifact":
        _log_if_kept_first(str(corr_id), payload, result)
    sess.commit()
    return True


def _log_if_kept_first(corr_id: str, payload: dict[str, Any], result: Any) -> None:
    """One warning when the stored promises_made differs from the incoming one: a re-published run
    artifact whose list the first-stored-wins rule kept out (#3596). Never raises."""
    try:
        stored = result.scalar()
    except Exception:  # noqa: BLE001
        return
    if not isinstance(stored, list):
        return
    incoming = ((payload.get("reflection") or {}) if isinstance(payload.get("reflection"), dict) else {}).get(
        "promises_made")
    # A longer incoming list means a re-run promised more: those extra items have no stored quote, so their acts
    # read as unreadable in the view (and as the v1 rule at the horizon). Logged the same way.
    if stored and stored != incoming:
        logger.warning("harness_turn_trace_promises_kept_first corr=%s stored=%d incoming=%s", corr_id,
                       len(stored), len(incoming) if isinstance(incoming, list) else "none")
