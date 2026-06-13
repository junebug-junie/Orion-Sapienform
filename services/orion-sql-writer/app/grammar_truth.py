"""Runtime snapshot and bounded retention for grammar production observe mode."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from app.db import grammar_engine
from app.settings import get_settings

logger = logging.getLogger("sql-writer.grammar_truth")

_BATCH_DELETE_EVENTS = text(
    """
    DELETE FROM grammar_events
    WHERE event_id IN (
        SELECT event_id
        FROM grammar_events
        WHERE created_at < :cutoff
        ORDER BY created_at ASC, event_id ASC
        LIMIT :batch_size
    )
    """
)

_COUNT_DEBT = text(
    """
    SELECT COUNT(*)::bigint
    FROM grammar_events
    WHERE created_at < :cutoff
    """
)

_VERIFY_INCOMING_FK = text(
    """
    SELECT COUNT(*)::bigint
    FROM pg_constraint c
    JOIN pg_class ref ON ref.oid = c.confrelid
    JOIN pg_namespace n ON n.oid = ref.relnamespace
    WHERE n.nspname = 'public'
      AND ref.relname = 'grammar_events'
      AND c.contype = 'f'
      AND c.conrelid <> ref.oid
    """
)


@dataclass
class GrammarRetentionState:
    enabled: bool = False
    configured_days: int = 0
    cutoff_at: datetime | None = None
    last_run_at: datetime | None = None
    rows_pruned_last_run: int = 0
    batches_attempted: int = 0
    elapsed_sec: float = 0.0
    remaining_debt: int | None = None
    failure_reason: str | None = None
    fk_delete_verified: bool = False
    fk_delete_verification_note: str | None = None
    capped_by_startup_limit: bool = False
    capped_by_elapsed_limit: bool = False
    batch_size: int = 0
    max_batches_per_startup: int = 0


_retention_state = GrammarRetentionState()


def retention_state() -> GrammarRetentionState:
    return _retention_state


def reset_retention_state_for_tests() -> None:
    global _retention_state
    _retention_state = GrammarRetentionState()


def _verify_grammar_events_delete_safe(conn) -> tuple[bool, str]:
    incoming = int(conn.execute(_VERIFY_INCOMING_FK).scalar_one())
    if incoming != 0:
        return False, f"unexpected_incoming_fk_to_grammar_events:{incoming}"
    return (
        True,
        "grammar_events_delete_is_child_safe;"
        " derived_tables_reference_grammar_traces_not_events;"
        " no ON DELETE CASCADE on grammar_events",
    )


def apply_grammar_events_retention(retention_days: int) -> GrammarRetentionState:
    """Bounded startup retention for grammar_events older than retention_days."""
    global _retention_state
    settings = get_settings()
    batch_size = max(1, int(settings.grammar_events_retention_batch_size))
    max_batches = max(1, int(settings.grammar_events_retention_max_batches_per_startup))
    max_elapsed_sec = max(1.0, float(settings.grammar_events_retention_max_elapsed_sec))

    state = GrammarRetentionState(
        enabled=retention_days > 0,
        configured_days=retention_days,
        batch_size=batch_size,
        max_batches_per_startup=max_batches,
    )

    if retention_days <= 0:
        _retention_state = state
        return state

    started = time.monotonic()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    state.cutoff_at = cutoff
    total_deleted = 0
    batches = 0

    try:
        with grammar_engine.connect() as conn:
            safe, note = _verify_grammar_events_delete_safe(conn)
            state.fk_delete_verified = safe
            state.fk_delete_verification_note = note
            if not safe:
                state.failure_reason = note
                state.last_run_at = datetime.now(timezone.utc)
                state.elapsed_sec = time.monotonic() - started
                _retention_state = state
                logger.error("grammar_retention_aborted fk_check_failed note=%s", note)
                return state

        while batches < max_batches:
            if (time.monotonic() - started) >= max_elapsed_sec:
                state.capped_by_elapsed_limit = True
                logger.warning(
                    "grammar_retention_elapsed_cap reached elapsed_sec=%.2f cap=%.2f batches=%s",
                    time.monotonic() - started,
                    max_elapsed_sec,
                    batches,
                )
                break
            with grammar_engine.begin() as conn:
                result = conn.execute(
                    _BATCH_DELETE_EVENTS,
                    {"cutoff": cutoff, "batch_size": batch_size},
                )
                deleted = int(result.rowcount or 0)
            total_deleted += deleted
            batches += 1
            logger.info(
                "grammar_retention_batch batch=%s deleted=%s cutoff=%s",
                batches,
                deleted,
                cutoff.isoformat(),
            )
            if deleted < batch_size:
                break

        with grammar_engine.connect() as conn:
            state.remaining_debt = int(conn.execute(_COUNT_DEBT, {"cutoff": cutoff}).scalar_one())

        if state.remaining_debt and state.remaining_debt > 0:
            state.capped_by_startup_limit = batches >= max_batches

        state.rows_pruned_last_run = total_deleted
        state.batches_attempted = batches
        state.last_run_at = datetime.now(timezone.utc)
        state.elapsed_sec = time.monotonic() - started

        logger.info(
            "grammar_retention_complete cutoff=%s rows_pruned=%s batches=%s "
            "elapsed_sec=%.2f remaining_debt=%s capped_batches=%s capped_elapsed=%s",
            cutoff.isoformat(),
            total_deleted,
            batches,
            state.elapsed_sec,
            state.remaining_debt,
            state.capped_by_startup_limit,
            state.capped_by_elapsed_limit,
        )
    except Exception as exc:
        state.failure_reason = str(exc)
        state.last_run_at = datetime.now(timezone.utc)
        state.elapsed_sec = time.monotonic() - started
        state.rows_pruned_last_run = total_deleted
        state.batches_attempted = batches
        logger.exception(
            "grammar_retention_failed cutoff=%s rows_pruned=%s batches=%s",
            cutoff.isoformat(),
            total_deleted,
            batches,
        )

    _retention_state = state
    return state


def _fallback_counts() -> dict[str, int]:
    """Count grammar.event.v1 fallbacks using typed created_at_ts when available."""
    with grammar_engine.connect() as conn:
        total = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM bus_fallback_log
                WHERE kind = 'grammar.event.v1'
                """
            ),
        ).scalar_one()
        recent_5m = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM bus_fallback_log
                WHERE kind = 'grammar.event.v1'
                  AND created_at_ts > (NOW() - INTERVAL '5 minutes')
                """
            ),
        ).scalar_one()
        recent_30m = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM bus_fallback_log
                WHERE kind = 'grammar.event.v1'
                  AND created_at_ts > (NOW() - INTERVAL '30 minutes')
                """
            ),
        ).scalar_one()
        recent_60m = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM bus_fallback_log
                WHERE kind = 'grammar.event.v1'
                  AND created_at_ts > (NOW() - INTERVAL '60 minutes')
                """
            ),
        ).scalar_one()
    return {
        "total": int(total),
        "last_5m": int(recent_5m),
        "last_30m": int(recent_30m),
        "last_60m": int(recent_60m),
    }


def _latest_events_by_source() -> list[dict[str, Any]]:
    with grammar_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT source_service,
                       MAX(created_at) AS latest_created_at,
                       COUNT(*) AS event_count
                FROM grammar_events
                GROUP BY source_service
                ORDER BY latest_created_at DESC NULLS LAST
                """
            ),
        ).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        ts = row["latest_created_at"]
        out.append(
            {
                "source_service": row["source_service"],
                "latest_created_at": ts.isoformat() if ts is not None else None,
                "event_count": int(row["event_count"]),
            }
        )
    return out


def _grammar_index_valid() -> dict[str, Any]:
    with grammar_engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'grammar_events'
                  AND indexname = 'idx_grammar_events_source_created'
                """
            ),
        ).mappings().first()
    return {
        "idx_grammar_events_source_created": row is not None,
        "indexdef": row["indexdef"] if row else None,
    }


def _retention_truth_block(settings) -> dict[str, Any]:
    rs = _retention_state
    return {
        "enabled": rs.enabled or settings.grammar_events_retention_days > 0,
        "configured_days": settings.grammar_events_retention_days,
        "batch_size": settings.grammar_events_retention_batch_size,
        "max_batches_per_startup": settings.grammar_events_retention_max_batches_per_startup,
        "max_elapsed_sec": settings.grammar_events_retention_max_elapsed_sec,
        "cutoff_at": rs.cutoff_at.isoformat() if rs.cutoff_at else None,
        "last_run_at": rs.last_run_at.isoformat() if rs.last_run_at else None,
        "rows_pruned_last_run": rs.rows_pruned_last_run,
        "batches_attempted": rs.batches_attempted,
        "elapsed_sec": rs.elapsed_sec,
        "remaining_debt": rs.remaining_debt,
        "failure_reason": rs.failure_reason,
        "fk_delete_verified": rs.fk_delete_verified,
        "fk_delete_verification_note": rs.fk_delete_verification_note,
        "capped_by_startup_limit": rs.capped_by_startup_limit,
        "capped_by_elapsed_limit": rs.capped_by_elapsed_limit,
    }


def build_grammar_truth_snapshot() -> dict[str, Any]:
    from app import worker as worker_mod

    settings = get_settings()
    queue = worker_mod.grammar_queue_snapshot()
    subscribed = list(settings.effective_subscribe_channels)
    rs = _retention_state

    degraded_reasons: list[str] = []
    if not settings.orion_bus_enabled:
        degraded_reasons.append("orion_bus_disabled")
    if not settings.sql_writer_enable_grammar_channel:
        degraded_reasons.append("grammar_channel_disabled")
    if "orion:grammar:event" not in subscribed:
        degraded_reasons.append("grammar_event_not_subscribed")
    if "orion:grammar:accepted-pressure" in subscribed:
        if not settings.sql_writer_allow_accepted_pressure_ingest:
            degraded_reasons.append("accepted_pressure_subscribed_without_explicit_allow")
    if queue.get("total_depth", 0) > 400:
        degraded_reasons.append("grammar_queue_backpressure")

    index_info = _grammar_index_valid()
    if not index_info["idx_grammar_events_source_created"]:
        degraded_reasons.append("grammar_source_created_index_missing")

    if rs.failure_reason:
        degraded_reasons.append("grammar_retention_failed")
    if rs.remaining_debt and rs.remaining_debt > 0:
        degraded_reasons.append("grammar_retention_debt_remaining")
    if settings.grammar_events_retention_days > 0 and rs.last_run_at is None:
        degraded_reasons.append("grammar_retention_not_run")

    return {
        "ok": not degraded_reasons,
        "degraded": bool(degraded_reasons),
        "degraded_reasons": degraded_reasons,
        "grammar_channel_enabled": settings.sql_writer_enable_grammar_channel,
        "subscribed_channels": subscribed,
        "grammar_worker_count": settings.sql_writer_grammar_workers,
        "grammar_queue": queue,
        "grammar_fallbacks": _fallback_counts(),
        "latest_by_source_service": _latest_events_by_source(),
        "grammar_index": index_info,
        "grammar_retention": _retention_truth_block(settings),
        "known_risks": {
            "retention_prunes_grammar_events_only": (
                "Retention deletes grammar_events rows only; orphan grammar_traces and derived "
                "storage may remain until a future pruner."
            ),
            "retention_debt_requires_restarts": (
                "Remaining retention debt may require repeated startups or a future background pruner."
            ),
            "publish_orion_bus_grammar_default_off": (
                "PUBLISH_ORION_BUS_GRAMMAR remains false in orion-bus code default; enable in deployed env."
            ),
            "accepted_pressure_not_canonical_ingress": (
                "orion:grammar:accepted-pressure is reducer output, not canonical sql-writer ingestion."
            ),
        },
        "effective_flags": {
            "orion_bus_enabled": settings.orion_bus_enabled,
            "sql_writer_enable_grammar_channel": settings.sql_writer_enable_grammar_channel,
            "grammar_subscribed": "orion:grammar:event" in subscribed,
            "accepted_pressure_subscribed": "orion:grammar:accepted-pressure" in subscribed,
            "sql_writer_allow_accepted_pressure_ingest": settings.sql_writer_allow_accepted_pressure_ingest,
            "sql_writer_grammar_workers": settings.sql_writer_grammar_workers,
            "grammar_events_retention_days": settings.grammar_events_retention_days,
        },
    }
