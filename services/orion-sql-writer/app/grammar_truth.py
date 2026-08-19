"""Runtime snapshot and bounded retention for grammar production observe mode."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db import engine as default_engine
from app.db import grammar_engine
from app.settings import get_settings

logger = logging.getLogger("sql-writer.grammar_truth")


def _batch_delete_sql(table: str, id_column: str) -> Any:
    # Table/column names are hardcoded call-site constants below, never caller
    # input -- an f-string here is not a SQL-injection surface.
    return text(
        f"""
        DELETE FROM {table}
        WHERE {id_column} IN (
            SELECT {id_column}
            FROM {table}
            WHERE created_at < :cutoff
            ORDER BY created_at ASC, {id_column} ASC
            LIMIT :batch_size
        )
        """
    )


def _count_debt_sql(table: str) -> Any:
    return text(f"SELECT COUNT(*)::bigint FROM {table} WHERE created_at < :cutoff")


_VERIFY_INCOMING_FK = text(
    """
    SELECT COUNT(*)::bigint
    FROM pg_constraint c
    JOIN pg_class ref ON ref.oid = c.confrelid
    JOIN pg_namespace n ON n.oid = ref.relnamespace
    WHERE n.nspname = 'public'
      AND ref.relname = :table
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
# grammar_events keeps the original single-slot global above (retention_state()
# and _retention_truth_block() already report on it by name in the health
# snapshot -- preserved as-is rather than reshaped, so that existing contract
# doesn't change). grammar_edges/grammar_atoms/substrate_organ_emissions are new
# as of this patch and get their own slots here instead, keyed by table name.
_extra_retention_state: dict[str, GrammarRetentionState] = {}


def retention_state() -> GrammarRetentionState:
    return _retention_state


def extra_retention_state(table: str) -> GrammarRetentionState | None:
    return _extra_retention_state.get(table)


def reset_retention_state_for_tests() -> None:
    global _retention_state
    _retention_state = GrammarRetentionState()
    _extra_retention_state.clear()


def _verify_delete_safe(conn, *, table: str) -> tuple[bool, str]:
    """No table in this repo has ever had a real ON DELETE CASCADE from grammar_events/
    grammar_edges/grammar_atoms/substrate_organ_emissions (confirmed live 2026-08-19:
    zero rows in pg_constraint for any of them) -- but that could change, so this stays
    a runtime check rather than a one-time assumption baked into the retention code."""
    incoming = int(conn.execute(_VERIFY_INCOMING_FK, {"table": table}).scalar_one())
    if incoming != 0:
        return False, f"unexpected_incoming_fk_to_{table}:{incoming}"
    return True, f"{table}_delete_is_child_safe; no incoming FK constraints found"


def _apply_bounded_table_retention(
    *,
    engine: Engine,
    table: str,
    id_column: str,
    retention_days: int,
    batch_size: int,
    max_batches: int,
    max_elapsed_sec: float,
) -> GrammarRetentionState:
    """Bounded startup retention for `table`, rows older than retention_days.

    Shared implementation behind apply_grammar_events_retention() and its three
    siblings below -- same batched-delete/FK-safety-check/elapsed-and-batch-cap
    shape for all four tables, since none of them need genuinely different
    retention logic, just different (engine, table, id_column) targets.
    """
    batch_size = max(1, int(batch_size))
    max_batches = max(1, int(max_batches))
    max_elapsed_sec = max(1.0, float(max_elapsed_sec))

    state = GrammarRetentionState(
        enabled=retention_days > 0,
        configured_days=retention_days,
        batch_size=batch_size,
        max_batches_per_startup=max_batches,
    )
    if retention_days <= 0:
        return state

    started = time.monotonic()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    state.cutoff_at = cutoff
    total_deleted = 0
    batches = 0
    delete_sql = _batch_delete_sql(table, id_column)
    debt_sql = _count_debt_sql(table)

    try:
        with engine.connect() as conn:
            safe, note = _verify_delete_safe(conn, table=table)
            state.fk_delete_verified = safe
            state.fk_delete_verification_note = note
            if not safe:
                state.failure_reason = note
                state.last_run_at = datetime.now(timezone.utc)
                state.elapsed_sec = time.monotonic() - started
                logger.error("%s_retention_aborted fk_check_failed note=%s", table, note)
                return state

        while batches < max_batches:
            if (time.monotonic() - started) >= max_elapsed_sec:
                state.capped_by_elapsed_limit = True
                logger.warning(
                    "%s_retention_elapsed_cap reached elapsed_sec=%.2f cap=%.2f batches=%s",
                    table,
                    time.monotonic() - started,
                    max_elapsed_sec,
                    batches,
                )
                break
            with engine.begin() as conn:
                result = conn.execute(delete_sql, {"cutoff": cutoff, "batch_size": batch_size})
                deleted = int(result.rowcount or 0)
            total_deleted += deleted
            batches += 1
            logger.info(
                "%s_retention_batch batch=%s deleted=%s cutoff=%s",
                table,
                batches,
                deleted,
                cutoff.isoformat(),
            )
            if deleted < batch_size:
                break

        with engine.connect() as conn:
            state.remaining_debt = int(conn.execute(debt_sql, {"cutoff": cutoff}).scalar_one())

        if state.remaining_debt and state.remaining_debt > 0:
            state.capped_by_startup_limit = batches >= max_batches

        state.rows_pruned_last_run = total_deleted
        state.batches_attempted = batches
        state.last_run_at = datetime.now(timezone.utc)
        state.elapsed_sec = time.monotonic() - started

        logger.info(
            "%s_retention_complete cutoff=%s rows_pruned=%s batches=%s "
            "elapsed_sec=%.2f remaining_debt=%s capped_batches=%s capped_elapsed=%s",
            table,
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
            "%s_retention_failed cutoff=%s rows_pruned=%s batches=%s",
            table,
            cutoff.isoformat(),
            total_deleted,
            batches,
        )

    return state


def apply_grammar_events_retention(retention_days: int) -> GrammarRetentionState:
    """Bounded startup retention for grammar_events older than retention_days."""
    global _retention_state
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_events",
        id_column="event_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )
    _retention_state = state
    return state


def apply_grammar_edges_retention(retention_days: int) -> GrammarRetentionState:
    """Bounded startup retention for grammar_edges older than retention_days.

    Reuses the grammar_events batch/cap knobs (settings.py comment explains why) --
    only the retention_days value is independently configurable per table.
    """
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_edges",
        id_column="edge_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )
    _extra_retention_state["grammar_edges"] = state
    return state


def apply_grammar_atoms_retention(retention_days: int) -> GrammarRetentionState:
    """Bounded startup retention for grammar_atoms older than retention_days."""
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_atoms",
        id_column="atom_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )
    _extra_retention_state["grammar_atoms"] = state
    return state


def apply_substrate_organ_emissions_retention(retention_days: int) -> GrammarRetentionState:
    """Bounded startup retention for substrate_organ_emissions older than retention_days.

    Uses the plain `engine` (not `grammar_engine`) -- this table is biometrics/substrate
    data, not part of the grammar lane, same reasoning as goal_provenance_streak_ticks'
    retention above using `engine` too. Already has idx_substrate_organ_emissions_created
    (created_at DESC) from its original table definition, so no new index is needed for
    this one, unlike the three grammar_* tables.
    """
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=default_engine,
        table="substrate_organ_emissions",
        id_column="emission_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )
    _extra_retention_state["substrate_organ_emissions"] = state
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
    # idx_grammar_events_created_at is what apply_grammar_events_retention()'s batched
    # DELETE actually needs (WHERE created_at < cutoff ORDER BY created_at, event_id) --
    # idx_grammar_events_source_created can't serve that scan without a source_service
    # predicate. Confirmed live 2026-08-19: retention failed on every startup with
    # `QueryCanceled: canceling statement due to statement timeout` until this index
    # existed. Both are reported so a missing *either* one is visible, not just the one
    # this snapshot originally tracked.
    with grammar_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE (tablename, indexname) IN (
                    ('grammar_events', 'idx_grammar_events_source_created'),
                    ('grammar_events', 'idx_grammar_events_created_at'),
                    ('grammar_edges', 'idx_grammar_edges_created_at'),
                    ('grammar_atoms', 'idx_grammar_atoms_created_at')
                )
                """
            ),
        ).mappings().all()
    by_name = {row["indexname"]: row["indexdef"] for row in rows}
    return {
        "idx_grammar_events_source_created": "idx_grammar_events_source_created" in by_name,
        "idx_grammar_events_created_at": "idx_grammar_events_created_at" in by_name,
        "idx_grammar_edges_created_at": "idx_grammar_edges_created_at" in by_name,
        "idx_grammar_atoms_created_at": "idx_grammar_atoms_created_at" in by_name,
        "indexdef": by_name.get("idx_grammar_events_source_created"),
    }


def _retention_block(rs: GrammarRetentionState, *, configured_days: int, batch_size: int,
                      max_batches: int, max_elapsed_sec: float) -> dict[str, Any]:
    return {
        "enabled": rs.enabled or configured_days > 0,
        "configured_days": configured_days,
        "batch_size": batch_size,
        "max_batches_per_startup": max_batches,
        "max_elapsed_sec": max_elapsed_sec,
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


def _retention_truth_block(settings) -> dict[str, Any]:
    return _retention_block(
        _retention_state,
        configured_days=settings.grammar_events_retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )


_EXTRA_RETENTION_TABLES = ("grammar_edges", "grammar_atoms", "substrate_organ_emissions")


def _other_retention_truth_blocks(settings) -> dict[str, dict[str, Any]]:
    days_by_table = {
        "grammar_edges": settings.grammar_edges_retention_days,
        "grammar_atoms": settings.grammar_atoms_retention_days,
        "substrate_organ_emissions": settings.substrate_organ_emissions_retention_days,
    }
    return {
        table: _retention_block(
            _extra_retention_state.get(table, GrammarRetentionState()),
            configured_days=days_by_table[table],
            batch_size=settings.grammar_events_retention_batch_size,
            max_batches=settings.grammar_events_retention_max_batches_per_startup,
            max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
        )
        for table in _EXTRA_RETENTION_TABLES
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
    if not index_info["idx_grammar_events_created_at"]:
        degraded_reasons.append("grammar_events_created_at_index_missing")
    if not index_info["idx_grammar_edges_created_at"]:
        degraded_reasons.append("grammar_edges_created_at_index_missing")
    if not index_info["idx_grammar_atoms_created_at"]:
        degraded_reasons.append("grammar_atoms_created_at_index_missing")

    if rs.failure_reason:
        degraded_reasons.append("grammar_retention_failed")
    if rs.remaining_debt and rs.remaining_debt > 0:
        degraded_reasons.append("grammar_retention_debt_remaining")
    if settings.grammar_events_retention_days > 0 and rs.last_run_at is None:
        degraded_reasons.append("grammar_retention_not_run")

    other_retention = _other_retention_truth_blocks(settings)
    for table, block in other_retention.items():
        if block["failure_reason"]:
            degraded_reasons.append(f"{table}_retention_failed")
        if block["remaining_debt"] and block["remaining_debt"] > 0:
            degraded_reasons.append(f"{table}_retention_debt_remaining")
        if block["configured_days"] > 0 and block["last_run_at"] is None:
            degraded_reasons.append(f"{table}_retention_not_run")

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
        "other_table_retention": other_retention,
        "known_risks": {
            "retention_prunes_grammar_traces_never": (
                "grammar_events/grammar_edges/grammar_atoms/substrate_organ_emissions all have "
                "bounded retention as of this patch; grammar_traces itself (and "
                "grammar_temporal_hops/grammar_compactions/grammar_projections) still do not -- "
                "orphan trace rows and other derived storage may remain until a future pruner."
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
