"""Runtime snapshot and bounded retention for grammar production observe mode."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

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


# grammar_events is not an archive -- it is a cursor-driven consumption queue. Five reducer
# lanes walk it forward (services/orion-substrate-runtime/app/store.py:352-390), each with
# its OWN filter, and each committing its position to substrate_reduction_cursor.
#
# WHAT THIS FLOOR ASKS, AND WHY THE OBVIOUS VERSION IS WRONG.
# The first version of this asked "where is the oldest cursor" -- MIN(last_event_created_at)
# -- and clamped the retention cutoff to it. That is the wrong question, and code review
# caught it before deploy with live data.
#
# A cursor advances only when its lane consumes a MATCHING row. So a cursor sitting still
# means "this lane's source has emitted nothing lately", which is indistinguishable from
# "this reducer is stalled". chat_grammar_consumer reads source_service='orion-hub' with
# trace_id LIKE 'hub.chat:%' -- literally Juniper talking to Orion. Measured live: that lane
# went silent for 1 day 19.6 hours (2026-08-16 -> 2026-08-18) while the system was healthy
# and ingesting ~400-500k grammar_events/day, and it sat 14m57s behind while the other four
# were inside 9 seconds. Against a 3-day window that is a 1.65x margin on ORDINARY BEHAVIOUR.
# A quiet weekend would have pinned retention at the chat cursor and stopped pruning, and
# (before the debt fix below) reported remaining_debt=0 while doing it.
#
# The right question is not "where is the cursor" but "does this lane have unconsumed rows
# below the cutoff". A silent lane has none -- everything older than its cursor is consumed
# and nothing newer exists -- so it correctly imposes NO floor. A genuinely stalled lane has
# real rows sitting above its cursor and below the cutoff, and those are exactly the rows
# that must survive.
#
# Consequence when a reducer IS stuck: disk grows instead of events disappearing. That is
# the correct trade -- growing disk is observable and recoverable, silently-skipped events
# are neither.
#
# LANE TABLE. Deliberately duplicated here rather than imported from
# orion/substrate/*/constants.py: importing those executes orion/substrate/__init__.py,
# which pulls the graph-DB store, materializer and reconciler into this thin writer. That
# exact mistake crash-looped two services on 2026-08-19. Drift is caught by a test instead
# (tests/test_grammar_retention_periodic.py::TestTheLaneTableMatchesTheRealConsumers), which
# imports the real constants at test time where heavy imports cost nothing.
# NOTE the execution lane covers THREE source services, not one. Live grammar_events shows
# orion-harness-governor writing cortex.exec: traces alongside orion-cortex-exec (528 vs
# 125,179 rows/day), and EXECUTION_SOURCE_SERVICES also includes orion-hub. A first draft of
# this table guessed ("orion-cortex-exec",) and would have under-protected exactly the lane
# it was meant to protect -- the floor probe would have missed unconsumed governor/hub rows
# and reported the lane clear. Caught by looking at live data, not by reading the code.
GRAMMAR_LANES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("biometrics_grammar_consumer", ("orion-biometrics",), "biometrics.node:"),
    (
        "execution_grammar_reducer",
        ("orion-cortex-exec", "orion-harness-governor", "orion-hub"),
        "cortex.exec:",
    ),
    ("transport_grammar_reducer", ("orion-bus",), "bus.transport:"),
    ("chat_grammar_consumer", ("orion-hub",), "hub.chat:"),
    ("route_grammar_consumer", ("orion-cortex-orch",), "orch.route:"),
)

_LANE_UNCONSUMED_SQL = text(
    """
    SELECT MIN(created_at)
      FROM grammar_events
     WHERE created_at > :cursor_ts
       AND created_at < :time_cutoff
       AND source_service = ANY(:sources)
       AND trace_id LIKE :trace_like
    """
)

_CURSOR_ROWS_SQL = text(
    "SELECT cursor_name, last_event_created_at FROM substrate_reduction_cursor"
)


def _grammar_events_cursor_floor(
    conn: Any, time_cutoff: datetime
) -> tuple[datetime | None, bool]:
    """Oldest row still owed to some reducer lane, as (floor, resolved).

    The two failure modes must not be conflated:

      (None, True)   every lane is fully consumed below `time_cutoff` -- nothing is owed,
                     so the time cutoff governs and retention proceeds normally.
      (None, False)  the answer could not be determined (table unreachable, timeout,
                     unexpected shape). The caller must SKIP the prune, not fall back to
                     the time cutoff.

    That second case is why this returns a pair. Falling back when the answer is unknown
    would delete rows that may be unconsumed -- the exact silent loss this prevents.
    Skipping costs one 60-second cycle; guessing costs events no reducer will see again.

    A lane with a NULL cursor imposes no floor on purpose: substrate-runtime seeds a new
    lane at the TAIL (store.py:184-190), never at the beginning of time, so it never wants
    historical rows.
    """
    try:
        rows = conn.execute(_CURSOR_ROWS_SQL).mappings().all()
    except Exception as exc:
        logger.warning("grammar_events_cursor_rows_unavailable err=%s", exc)
        return None, False

    positions: dict[str, Any] = {}
    for row in rows:
        try:
            positions[row["cursor_name"]] = row["last_event_created_at"]
        except Exception as exc:
            logger.warning("grammar_events_cursor_row_unreadable err=%s", exc)
            return None, False

    floor: datetime | None = None
    for cursor_name, sources, trace_prefix in GRAMMAR_LANES:
        cursor_ts = positions.get(cursor_name)
        if cursor_ts is None:
            # No row for this lane, or a tail-seeded NULL. Nothing owed either way.
            continue
        if not isinstance(cursor_ts, datetime):
            logger.warning(
                "grammar_events_cursor_unexpected_type lane=%s type=%s",
                cursor_name,
                type(cursor_ts).__name__,
            )
            return None, False
        if cursor_ts.tzinfo is None:
            cursor_ts = cursor_ts.replace(tzinfo=timezone.utc)
        try:
            oldest = conn.execute(
                _LANE_UNCONSUMED_SQL,
                {
                    "cursor_ts": cursor_ts,
                    "time_cutoff": time_cutoff,
                    "sources": list(sources),
                    "trace_like": f"{trace_prefix}%",
                },
            ).scalar()
        except Exception as exc:
            logger.warning("grammar_events_lane_probe_failed lane=%s err=%s", cursor_name, exc)
            return None, False
        if oldest is None:
            continue  # this lane owes nothing below the cutoff
        if not isinstance(oldest, datetime):
            logger.warning("grammar_events_lane_probe_unexpected_type lane=%s", cursor_name)
            return None, False
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        logger.warning(
            "grammar_events_lane_owed lane=%s oldest_unconsumed=%s -- reducer is behind",
            cursor_name,
            oldest.isoformat(),
        )
        if floor is None or oldest < floor:
            floor = oldest
    return floor, True


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
    cursor_floor_at: datetime | None = None
    cursor_floor_applied: bool = False
    debt_count_failed_reason: str | None = None
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


# substrate_proposal_frames is not an archive either -- it is the FIRST stage of a four-stage
# pipeline, and three later stages can still reach back into it by frame_id long after the row
# was written:
#
#   proposal -> policy decision -> execution dispatch -> feedback
#
# Each hop has its own pending marker and partial index, added by
# services/orion-sql-db/manual_migration_substrate_pending_markers.sql (2026-08-19). That
# migration's own header records why a TIME window is the wrong instrument here, and says so in
# capitals: a time bound was attempted first and reverted. Measured over 7 days at the time, the
# dispatch->feedback hop ran p50 34.6 hours and max 11.3 DAYS behind, with 8 of 30 days entirely
# in that regime. Any "recent rows only" cutoff strands the backlog, and strands it permanently
# once fresh work keeps the fast path busy. Retention must not re-introduce the exact bound that
# migration removed.
#
# So the floor asks the markers, not the clock: what is the oldest row any stage still owes work
# on? Everything at or above that timestamp stays, no matter how old it is.
#
# WHY A PENDING MARKER IS SAFE TO FLOOR ON, WHERE A REDUCTION CURSOR WAS NOT.
# The grammar cursor floor above deliberately does NOT ask "where is the oldest cursor", because
# a cursor stops moving when its SOURCE goes quiet, which is indistinguishable from a stall. A
# pending marker has no such ambiguity: it is set at insert and cleared in the same transaction
# as the downstream write, so a pending row is unconsumed work by construction, and a quiet
# source produces no pending rows at all rather than a stuck floor. The failure mode that does
# remain is a leaked marker -- a row stuck pending forever would pin retention forever -- which
# is why a binding floor logs at WARNING with the stage name and how far back it reaches, the
# same way the cursor floor does. Live 2026-08-20, all three stages were current: oldest pending
# was 1 / 96 / 15 rows, all under four minutes old.
_SUBSTRATE_CHAIN_PENDING: tuple[tuple[str, str, str], ...] = (
    ("proposal->policy", "substrate_proposal_frames", "policy_pending"),
    ("policy->dispatch", "substrate_policy_decision_frames", "dispatch_pending"),
    ("dispatch->feedback", "substrate_execution_dispatch_frames", "feedback_pending"),
)


def _substrate_chain_floor(conn: Any, time_cutoff: datetime) -> tuple[datetime | None, bool]:
    """Oldest row any downstream substrate stage still owes work on, as (floor, resolved).

    Same two-value contract as _grammar_events_cursor_floor: (None, True) means every stage is
    caught up and the time cutoff governs; (None, False) means the question could not be
    answered and the caller must refuse to prune.

    Each probe rides the stage's partial index (idx_*_pending, 32 kB live), so this is three
    tiny index-only scans, not three scans of a 1.7 GB table.
    """
    floor: datetime | None = None
    for stage, table, marker in _SUBSTRATE_CHAIN_PENDING:
        try:
            # Table/column names are module constants above, never caller input.
            oldest = conn.execute(
                # created_at, NOT generated_at, even though the partial index is on
                # generated_at. The delete predicate below keys on created_at, and the two
                # columns are on different clocks: generated_at is when the stage produced the
                # frame, created_at is when this row was written. Measured live 2026-08-20 they
                # diverge by up to 724s on substrate_execution_dispatch_frames, and 24 policy
                # rows have created_at strictly BEFORE generated_at. Comparing a floor from one
                # clock against a cutoff on the other is how you delete rows you meant to keep.
                # The pending set is tiny (1/96/15 rows live), so the index still does the
                # narrowing and the heap fetch for created_at is negligible.
                text(f"SELECT MIN(created_at) FROM {table} WHERE {marker}")  # noqa: S608
            ).scalar()
        except Exception:  # noqa: BLE001 - unresolved is a distinct outcome from "no floor"
            logger.exception(
                "substrate_chain_floor_probe_failed stage=%s table=%s -- refusing to prune",
                stage,
                table,
            )
            return None, False
        if oldest is None:
            continue
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        if floor is None or oldest < floor:
            floor = oldest
    return floor, True


def _apply_bounded_table_retention(
    *,
    engine: Engine,
    table: str,
    id_column: str,
    retention_days: int,
    batch_size: int,
    max_batches: int,
    max_elapsed_sec: float,
    respect_cursor_floor: bool = False,
    floor_resolver: Callable[[Any, datetime], tuple[datetime | None, bool]] | None = None,
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

    time_cutoff = cutoff
    # respect_cursor_floor is the grammar lane's spelling of the same idea; keeping it as a
    # named flag rather than folding it into floor_resolver keeps every existing call site and
    # its test unchanged.
    resolve_floor = floor_resolver
    if respect_cursor_floor and resolve_floor is None:
        resolve_floor = _grammar_events_cursor_floor
    if resolve_floor is not None:
        with engine.connect() as conn:
            floor, resolved = resolve_floor(conn, time_cutoff)
        state.cursor_floor_at = floor
        if not resolved:
            # Fail safe: no floor, no prune. Retention runs every cycle, so skipping one
            # is cheap; deleting rows we cannot prove were consumed is not recoverable.
            state.failure_reason = "cursor_floor_unresolved"
            state.last_run_at = datetime.now(timezone.utc)
            state.elapsed_sec = time.monotonic() - started
            logger.warning(
                "%s_retention_skipped cursor floor unresolved -- refusing to prune a "
                "cursor-consumed table without knowing what has been consumed",
                table,
            )
            return state
        if floor is not None and floor < cutoff:  # a lane is genuinely behind
            # A reducer is further behind than the retention window. Delete only up to
            # where it has actually consumed, and say so -- this is the difference between
            # "retention cannot keep up" and "a reducer is stuck", which look identical
            # from disk usage alone.
            state.cursor_floor_applied = True
            logger.warning(
                "%s_retention_cursor_floor_binds time_cutoff=%s cursor_floor=%s "
                "reducer_behind_sec=%.0f -- deleting only up to the cursor",
                table,
                cutoff.isoformat(),
                floor.isoformat(),
                (cutoff - floor).total_seconds(),
            )
            cutoff = floor

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

        # Diagnostics only, and deliberately non-fatal. Live 2026-08-19:
        # substrate_organ_emissions' debt COUNT exceeded the 10s grammar statement_timeout
        # (SQL_WRITER_GRAMMAR_STATEMENT_TIMEOUT_MS) and raised, which logged the whole run
        # as `retention_failed` and reported remaining_debt=None -- even though all 100,000
        # deletes had already committed. A reporting query must not be able to mask a
        # successful prune as a failure, nor hide the debt number that tells you whether
        # retention is converging.
        # Measured against `time_cutoff`, NOT the possibly-clamped `cutoff`. Using the
        # clamped value makes the debt instrument read 0 in exactly the state where it must
        # scream: the floor pinned retention, nothing was pruned, and millions of rows sit
        # above the floor. The number has to mean "rows past the retention window", not
        # "rows I was allowed to touch this cycle".
        try:
            with engine.connect() as conn:
                state.remaining_debt = int(
                    conn.execute(debt_sql, {"cutoff": time_cutoff}).scalar_one()
                )
        except Exception as exc:
            state.remaining_debt = None
            state.debt_count_failed_reason = str(exc)
            logger.warning(
                "%s_retention_debt_count_failed (prune itself succeeded, rows_pruned=%s) err=%s",
                table,
                total_deleted,
                exc,
            )

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


def apply_grammar_events_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
    """Bounded startup retention for grammar_events older than retention_days."""
    global _retention_state
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_events",
        id_column="event_id",
        respect_cursor_floor=True,
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
    )
    _retention_state = state
    return state


def apply_grammar_edges_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
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
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
    )
    _extra_retention_state["grammar_edges"] = state
    return state


def apply_grammar_atoms_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
    """Bounded startup retention for grammar_atoms older than retention_days."""
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_atoms",
        id_column="atom_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
    )
    _extra_retention_state["grammar_atoms"] = state
    return state


def apply_substrate_organ_emissions_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
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
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
    )
    _extra_retention_state["substrate_organ_emissions"] = state
    return state


def apply_substrate_proposal_frames_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
    """Bounded retention for substrate_proposal_frames older than retention_days.

    THE ONLY UNPRUNED TABLE IN THE SUBSTRATE PIPELINE THAT HAD NO BOUND AT ALL. Live
    2026-08-20: 474,230 rows / 1,758 MB, oldest 2026-07-23, growing ~27k rows and ~105 MB a
    day with nothing to stop it.

    Uses the plain `engine` (not `grammar_engine`), same as
    apply_substrate_organ_emissions_retention above -- this is substrate data, not a grammar
    lane.

    Floored on the pipeline's pending markers via _substrate_chain_floor, NOT on time alone.
    See the long comment on that function for why: three later stages reach back into this
    table by frame_id, and the migration that introduced the markers explicitly reverted a
    time-bounded version of this same idea. A row still owed to any stage survives regardless
    of age.

    No new index needed: idx_substrate_proposal_frames_created_at (created_at DESC) already
    exists from manual_migration_substrate_frames_created_at_index.sql, and a DESC btree
    serves a `created_at < :cutoff` range scan fine. Confirmed live before shipping.
    """
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=default_engine,
        table="substrate_proposal_frames",
        id_column="frame_id",
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
        floor_resolver=_substrate_chain_floor,
    )
    _extra_retention_state["substrate_proposal_frames"] = state
    return state



def apply_grammar_traces_retention(
    retention_days: int,
    *,
    max_batches: int | None = None,
    max_elapsed_sec: float | None = None,
) -> GrammarRetentionState:
    """Bounded retention for grammar_traces older than retention_days.

    THE PARENT ROW, NOT A LEAF. Every other table here is a leaf: pruning it removes
    exactly its own rows. grammar_traces is the row the Grammar Atlas lists and then
    expands (orion/grammar/query.py::list_traces -> get_trace), so leaving it out of
    retention did not just waste disk -- it produced hollow cognition surfaces. Measured
    live 2026-08-20, before this function existed: 487,970 traces, oldest 2026-07-23,
    205,465 of them (42%) with zero atoms. That is a UI panel rendered with no real
    backing artifact, which CLAUDE.md names as an invalid success state outright.

    THE CURSOR FLOOR IS ON, AND THE OBVIOUS REASON FOR IT IS WRONG. An earlier version of
    this docstring said "a trace's created_at is <= the created_at of every event under it,
    so the same floor keeps the parent alive at least as long as its owed children". Code
    review falsified that on live data: 3,627 of 3,627 sampled traces (100%) violate it,
    and the claim is self-contradictory anyway -- a SMALLER created_at is deleted SOONER.

    The two columns are on different clocks. ledger.py:71 stamps grammar_traces.created_at
    with wall-clock WRITE time; ledger.py:117 stamps grammar_events.created_at with
    OCCURRENCE time (observed_at or emitted_at). Occurrence precedes write, so a trace row
    is consistently NEWER than its own earliest event -- measured lag 0.15s to 9.4s.

    What that actually buys, precisely: a trace outlives its EARLIEST event by
    construction. It does NOT guarantee the trace outlives every LATER unconsumed event,
    and the floor does not close that gap either -- the floor is MIN(created_at) over
    unconsumed events, so a trace whose first event was already consumed can sit below the
    floor and be deleted while a later sibling event survives. Measured at the live 3-day
    cutoff: 4 such events, 8.6s of lag, gone on the next 60s cycle.

    The floor stays on anyway, for the case it was built for: a genuinely stalled lane,
    where it holds trace rows back by hours or days rather than seconds. Nothing reads
    across the seam in the meantime -- reducers key on grammar_events and never join
    grammar_traces, and the Atlas keys on grammar_traces -- so a briefly parentless event
    is inert. Do not upgrade this to a correctness guarantee it does not provide.

    There are no FK constraints to enforce any of this: the checked-in
    manual_migration_grammar_atlas.sql declares `references grammar_traces(trace_id)` on
    six child tables, but the live database has ZERO foreign keys touching any grammar_*
    table (verified 2026-08-20 against pg_constraint) -- the tables were created by
    SQLAlchemy metadata first, and the migration's FK clauses never took effect. So
    _verify_delete_safe passes here, and ordering is a correctness argument this code has
    to make for itself rather than one the database will make for it.
    """
    settings = get_settings()
    state = _apply_bounded_table_retention(
        engine=grammar_engine,
        table="grammar_traces",
        id_column="trace_id",
        respect_cursor_floor=True,
        retention_days=retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=(
            settings.grammar_events_retention_max_batches_per_startup
            if max_batches is None
            else max_batches
        ),
        max_elapsed_sec=(
            settings.grammar_events_retention_max_elapsed_sec
            if max_elapsed_sec is None
            else max_elapsed_sec
        ),
    )
    _extra_retention_state["grammar_traces"] = state
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
        # Added 2026-08-20. Without these, a pinned floor is indistinguishable on /health
        # from "healthy, nothing to do": rows_pruned=0, failure_reason=null, neither cap
        # set. cursor_floor_applied is the only field that says "a reducer is behind and
        # retention is deliberately holding back".
        "cursor_floor_at": rs.cursor_floor_at.isoformat() if rs.cursor_floor_at else None,
        "cursor_floor_applied": rs.cursor_floor_applied,
        "debt_count_failed_reason": rs.debt_count_failed_reason,
    }


def _retention_truth_block(settings) -> dict[str, Any]:
    return _retention_block(
        _retention_state,
        configured_days=settings.grammar_events_retention_days,
        batch_size=settings.grammar_events_retention_batch_size,
        max_batches=settings.grammar_events_retention_max_batches_per_startup,
        max_elapsed_sec=settings.grammar_events_retention_max_elapsed_sec,
    )


_EXTRA_RETENTION_TABLES = (
    "grammar_edges",
    "grammar_atoms",
    "substrate_organ_emissions",
    "grammar_traces",
    "substrate_proposal_frames",
)


# Tables whose retention window could not be read from Settings on the last snapshot.
# Surfaced as a degraded reason rather than an exception -- see the comment inside.
_retention_window_config_missing: set[str] = set()


def _other_retention_truth_blocks(settings) -> dict[str, dict[str, Any]]:
    # Derived from _EXTRA_RETENTION_TABLES rather than hand-listed. The hand-written dict
    # this replaces raised KeyError and took the ENTIRE /health snapshot down the moment a
    # table was added to _EXTRA_RETENTION_TABLES without also being added here -- caught by
    # tests when grammar_traces was added, but it is a trap that would fire again.
    # getattr with a hard failure, not a silent default: a 0 here reads as "retention
    # disabled for this table", which is exactly the wrong thing to invent from a typo.
    # DEGRADE, DO NOT RAISE. build_grammar_truth_snapshot() is called from /health with no
    # try/except (main.py), so raising here returns 500 for the WHOLE endpoint -- the exact
    # blast radius of the KeyError this replaced, just with a different exception name. A
    # missing window is reported as a degraded reason on the table it belongs to instead.
    # The real gate against this drift is a test that binds _EXTRA_RETENTION_TABLES to the
    # live Settings class (tests/test_grammar_retention_periodic.py); this branch is the
    # runtime backstop for when that test is wrong, not the primary defence.
    days_by_table: dict[str, int] = {}
    missing: list[str] = []
    for table in _EXTRA_RETENTION_TABLES:
        attr = f"{table}_retention_days"
        value = getattr(settings, attr, None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            missing.append(table)
            logger.error(
                "%s is in _EXTRA_RETENTION_TABLES but Settings has no usable %s -- "
                "reporting it as unconfigured; retention for it is NOT running",
                table,
                attr,
            )
            days_by_table[table] = 0
            continue
        days_by_table[table] = int(value)
    _retention_window_config_missing.clear()
    _retention_window_config_missing.update(missing)
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
        if table in _retention_window_config_missing:
            degraded_reasons.append(f"{table}_retention_window_unconfigured")

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
            "retention_skips_derived_grammar_tables": (
                "grammar_events/grammar_edges/grammar_atoms/grammar_traces/"
                "substrate_organ_emissions all have bounded periodic retention. "
                "grammar_temporal_hops/grammar_compactions/grammar_projections still do "
                "not -- deliberately: all three are effectively unwritten (16-24 kB each "
                "live 2026-08-20, versus 143 MB - 18 GB for the managed tables), so a "
                "pruner for them would be ceremony, not a fix. Revisit if they ever grow."
            ),
            "retention_debt_requires_restarts": (
                "Retention runs every GRAMMAR_RETENTION_INTERVAL_SEC seconds (app/grammar_retention_loop.py); remaining_debt is measured against the retention window, not the possibly-clamped cutoff. While a table still carries debt, the Grammar Atlas can list traces older than the window whose events/atoms were pruned first -- 42% of traces were already hollow when grammar_traces retention was added."
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


# Retention runs on a timer, not only at boot.
#
# WHY THIS EXISTS. Measured live 2026-08-19/20, from the service's own logs and
# pg_stat_user_tables, before this patch:
#
#   arrival      1,117,440 rows/day across the four tables (2.42 GB/day)
#   deletion       365,000 rows per PROCESS START, then exactly 0 until the next one
#   debt         5,352,878 rows already past the cutoff, and climbing
#
# Every startup ran to its cap and stopped: grammar_events and grammar_atoms hit the
# 100-batch cap, grammar_edges hit the 120s wall at batch 65, substrate_organ_emissions
# hit both the cap and a statement timeout on its own debt query. The instrumentation was
# never the problem -- each run logged `remaining_debt` correctly. Nothing ever ran again
# to act on it, because the only trigger was process start.
#
# A startup-triggered job cannot converge against continuous arrival, no matter how its
# caps are tuned: it would need roughly three restarts a day just to break even, and about
# fifteen more on top of that to clear the standing debt. The fix is not a bigger cap.
#
# ORDER IS DELIBERATE but SMALL: grammar_traces runs LAST, after every table that hangs
# off it. Nothing in the database enforces this -- there are no live FK constraints on
# grammar_* -- so it is an argument this list has to make for itself.
#
# Do not overstate it. Intra-cycle ordering buys microseconds. What actually governs how
# long a child row can point at a deleted parent is per-table debt versus drain rate, and
# on deploy that is DAYS. Measured 2026-08-20, with the periodic caps (3 batches x 1000
# rows x 1440 cycles/day = 4.32M rows/day of capacity per table):
#
#   table                      debt past cutoff   arrival/hr   converges in
#   grammar_traces                      417,966        1,714   ~2.4 hours
#   grammar_atoms                     1,768,239       15,337   ~0.45 days
#   grammar_edges                     2,021,469       15,282   ~0.5 days
#   grammar_events                    6,227,497       34,135   ~1.8 days
#
# grammar_traces reaches its cutoff ~18x faster than grammar_events, so for roughly the
# first two days after deploy there are millions of events whose parent trace is already
# gone. That is inert -- reducers key on grammar_events and never join grammar_traces, and
# the Atlas keys on grammar_traces -- but it is the real shape of the window, not the
# microsecond one this ordering controls.
GRAMMAR_RETENTION_TABLES: tuple[tuple[str, Any], ...] = (
    ("grammar_events", apply_grammar_events_retention),
    ("grammar_edges", apply_grammar_edges_retention),
    ("grammar_atoms", apply_grammar_atoms_retention),
    ("substrate_organ_emissions", apply_substrate_organ_emissions_retention),
    ("grammar_traces", apply_grammar_traces_retention),
    # Independent of the grammar tables above -- different engine, different floor, no
    # ordering relationship with them. Placed last only because it is newest.
    ("substrate_proposal_frames", apply_substrate_proposal_frames_retention),
)


def run_one_retention_cycle(
    *,
    days_for: dict[str, int],
    max_batches: int,
    max_elapsed_sec: float,
    stop: Any = None,
) -> dict[str, GrammarRetentionState]:
    """One bounded pass over every retention-managed table. Synchronous by design.

    Callers on an event loop MUST run this in a worker thread -- it uses blocking
    SQLAlchemy connections, and sql-writer's event loop is also draining the bus.
    """
    out: dict[str, GrammarRetentionState] = {}
    for table, fn in GRAMMAR_RETENTION_TABLES:
        # Cooperative stop. asyncio.to_thread is NOT cancellable once the callable has
        # started, so without this the event loop reports a clean shutdown while this
        # thread keeps deleting -- and Python joins the executor thread at interpreter
        # exit, blocking the process for up to 4 tables x (elapsed cap + one in-flight
        # batch). Docker's default 10s stop grace would then SIGKILL every restart.
        if stop is not None and stop.is_set():
            logger.info("grammar_retention_cycle_stopping before table=%s", table)
            break
        days = int(days_for.get(table, 0) or 0)
        if days <= 0:
            continue
        try:
            out[table] = fn(days, max_batches=max_batches, max_elapsed_sec=max_elapsed_sec)
        except Exception:
            # One table failing must not stop the others -- they are independent tables
            # with independent debt, and the whole point of the loop is that it keeps
            # running. _apply_bounded_table_retention already logs the detail.
            logger.exception("%s_retention_cycle_failed", table)
    return out
