from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.attention.field_attention.candidate_precision_weighted import (
    PrecisionEwmaBaseline,
    advance_precision_baseline,
)
from orion.attention.field_attention.goal_provenance import DominanceStreak
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from app.dominance_runs import advance_run
from orion.schemas.prediction_error_definitions import (
    UNSTAMPED_DEFINITION_VERSION,
    prediction_error_definition_version,
)

logger = logging.getLogger("orion.attention.runtime.store")

# Singleton row id for the one real node-target dominance streak this service
# tracks (see load_node_dominance_streak/save_node_dominance_streak below).
_NODE_DOMINANCE_STREAK_ID = "node_target_dominance_streak"

# How long a "definition_version column is missing" answer is trusted before
# re-checking (the column arrives via a manual migration that may land while
# this service is running).
_DEFINITION_VERSION_COLUMN_RECHECK_SEC = 300.0

# Batched, guard-railed prune: never deletes the newest frame (by generated_at,
# matching load_latest_attention_frame's ordering).
PRUNE_ATTENTION_FRAMES_SQL = """
DELETE FROM substrate_attention_frames
WHERE ctid IN (
    SELECT ctid
    FROM substrate_attention_frames
    WHERE created_at < :cutoff
      AND frame_id <> (
          SELECT frame_id FROM substrate_attention_frames
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""


_UPSERT_BASELINE_SQL = """
INSERT INTO substrate_node_prediction_error_baseline (
    target_id, reducer_key, ewma, variance,
    observation_count, last_value, last_receipt_created_at,
    updated_at
) VALUES (
    :target_id, :reducer_key, :ewma, :variance,
    :observation_count, :last_value, :last_receipt_created_at,
    :updated_at
)
ON CONFLICT (target_id) DO UPDATE SET
    reducer_key = EXCLUDED.reducer_key,
    ewma = EXCLUDED.ewma,
    variance = EXCLUDED.variance,
    observation_count = EXCLUDED.observation_count,
    last_value = EXCLUDED.last_value,
    last_receipt_created_at = EXCLUDED.last_receipt_created_at,
    updated_at = EXCLUDED.updated_at
"""

_UPSERT_BASELINE_VERSIONED_SQL = """
INSERT INTO substrate_node_prediction_error_baseline (
    target_id, reducer_key, ewma, variance,
    observation_count, last_value, last_receipt_created_at,
    updated_at, definition_version
) VALUES (
    :target_id, :reducer_key, :ewma, :variance,
    :observation_count, :last_value, :last_receipt_created_at,
    :updated_at, :definition_version
)
ON CONFLICT (target_id) DO UPDATE SET
    reducer_key = EXCLUDED.reducer_key,
    ewma = EXCLUDED.ewma,
    variance = EXCLUDED.variance,
    observation_count = EXCLUDED.observation_count,
    last_value = EXCLUDED.last_value,
    last_receipt_created_at = EXCLUDED.last_receipt_created_at,
    updated_at = EXCLUDED.updated_at,
    definition_version = EXCLUDED.definition_version
"""


def _parse_definition_version(raw) -> int:
    """A stamped version, or ``UNSTAMPED_DEFINITION_VERSION`` for a missing/garbled one
    (receipts and rows written before versions existed carry none)."""
    if raw is None:
        return UNSTAMPED_DEFINITION_VERSION
    try:
        return int(raw)
    except (TypeError, ValueError):
        return UNSTAMPED_DEFINITION_VERSION


def _observed_at_from_row(row) -> datetime | None:
    """When the reading behind a persisted ``last_value`` was written: the
    ``last_value_observed_at`` column when present and set, else the receipt cursor
    (rows written before the column existed, or before it is migrated in). The
    cursor fallback can read fresher than the truth when skipped receipts followed
    the last folded one."""
    raw = row.get("last_value_observed_at")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return row.get("last_receipt_created_at")


class AttentionRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )
        self._definition_version_column: bool | None = None
        self._definition_version_checked_at: datetime | None = None
        self._observed_at_column: bool | None = None
        self._observed_at_checked_at: datetime | None = None

    def _has_observed_at_column(self, conn) -> bool:
        """Whether ``substrate_node_prediction_error_baseline.last_value_observed_at``
        exists (services/orion-sql-db/manual_migration_node_prediction_error_
        baseline_v3_last_value_observed_at.sql). Same caching as the
        definition_version probe."""
        cached = getattr(self, "_observed_at_column", None)
        checked_at = getattr(self, "_observed_at_checked_at", None)
        now = datetime.now(timezone.utc)
        if cached is True:
            return True
        if (
            cached is False
            and checked_at is not None
            and (now - checked_at).total_seconds() < _DEFINITION_VERSION_COLUMN_RECHECK_SEC
        ):
            return False
        present = bool(
            conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'substrate_node_prediction_error_baseline'
                          AND column_name = 'last_value_observed_at'
                    ) AS present
                    """
                )
            ).scalar()
        )
        if not present:
            logger.warning(
                "node_prediction_error_baseline_observed_at_column_missing "
                "apply services/orion-sql-db/manual_migration_node_prediction_error_"
                "baseline_v3_last_value_observed_at.sql; the staleness fade ages "
                "readings from the receipt cursor until it is applied, and the "
                "cursor keeps moving over skipped receipts"
            )
        self._observed_at_column = present
        self._observed_at_checked_at = now
        return present

    def _has_definition_version_column(self, conn) -> bool:
        """Whether ``substrate_node_prediction_error_baseline.definition_version``
        exists (services/orion-sql-db/manual_migration_node_prediction_error_
        baseline_v2_definition_version.sql). Cached once found; a missing answer is
        re-checked every ``_DEFINITION_VERSION_COLUMN_RECHECK_SEC``."""
        cached = getattr(self, "_definition_version_column", None)
        checked_at = getattr(self, "_definition_version_checked_at", None)
        now = datetime.now(timezone.utc)
        if cached is True:
            return True
        if (
            cached is False
            and checked_at is not None
            and (now - checked_at).total_seconds() < _DEFINITION_VERSION_COLUMN_RECHECK_SEC
        ):
            return False
        present = bool(
            conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'substrate_node_prediction_error_baseline'
                          AND column_name = 'definition_version'
                    ) AS present
                    """
                )
            ).scalar()
        )
        if not present:
            logger.warning(
                "node_prediction_error_baseline_definition_version_column_missing "
                "apply services/orion-sql-db/manual_migration_node_prediction_error_"
                "baseline_v2_definition_version.sql; baselines are NOT reset on a "
                "prediction-error definition change until it is applied"
            )
        self._definition_version_column = present
        self._definition_version_checked_at = now
        return present

    def load_latest_field(self) -> FieldStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldStateV1.model_validate(payload)

    def load_field_for_tick(self, tick_id: str) -> FieldStateV1 | None:
        """One stored field tick by id (primary key). Used for the previous
        attention frame's own field tick (#2534 decision 2: novelty must not
        count a channel going dark as a change)."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT field_json FROM substrate_field_state WHERE tick_id = :tick_id"),
                    {"tick_id": tick_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldStateV1.model_validate(payload)

    def load_prediction_error_history(self, *, reducer_key: str, limit: int) -> list[float]:
        """Real, ASC-by-time prediction-error history for one reducer (oldest
        first, most recent/"current" last), for Candidate A
        (`orion/attention/field_attention/candidate_precision_weighted.py::
        precision_weighted_salience`) to compute real precision (1/variance)
        from.

        **No longer called by the live tick as of 2026-07-30** (Sentience Striving
        Program officer review -- see `orion/sentience_striving_program/README.md`
        §12): this method's own ~30-minute rolling-window recompute was the root
        cause of a live incident (a target with as few as 2 real samples surviving
        the window could win a fully-confident-looking `salience_score=1.0`). The
        live tick now calls `advance_node_prediction_error_baseline` (below)
        instead, which persists a cumulative baseline immune to this window. This
        method is kept, unchanged, for offline/replay analysis over a full
        historical export where a persisted incremental baseline doesn't apply
        (`scripts/analysis/measure_precision_weighted_salience_probe.py`,
        `measure_candidate_a_vs_b_head_to_head.py`) -- do not wire it back into the
        live worker tick.

        2026-07-30 fix (caught while reviewing a sibling script's own
        divergence from this method): the query fetches the NEWEST `limit`
        rows (`ORDER BY created_at DESC LIMIT`), then reverses to ASC order
        in Python. The original version queried `ORDER BY created_at ASC
        LIMIT` directly -- fetching the OLDEST `limit` rows within the
        retention window instead. Harmless only as long as a reducer never
        produces more than `limit` real rows within its own retention
        window; the moment it does, that version would silently return
        stale data and never see the actual latest tick at all --
        `precision_weighted_salience()`'s `current_error` is defined as the
        list's last element, so a real "current" reading would go
        permanently stale for any reducer with a tick rate high enough to
        exceed `limit` within `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES`
        (30 min live default). Not yet observed live at the row counts
        checked during this session (well under 200), but a real, latent
        bug, not just a hypothetical -- fixed here rather than left for a
        future high-volume reducer to trip over silently.

        `substrate_reduction_receipts` retains success receipts for only
        `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` -- this is always a
        rolling recent window, not full history, a structural property of
        the source table, not a bug here. Degrades to `[]` on any error
        (missing table, bad row) -- a history-fetch failure must never
        crash the attention tick; `precision_weighted_salience([])` already
        handles the empty case honestly (zero salience, n_samples=0).
        """
        try:
            with self._engine.connect() as conn:
                rows = (
                    conn.execute(
                        text(
                            """
                            SELECT
                                receipt_json -> 'state_deltas' -> 0 -> 'after'
                                    -> 'pressure_hints' ->> 'prediction_error' AS error
                            FROM substrate_reduction_receipts
                            WHERE reducer_name = :reducer_id
                            ORDER BY created_at DESC
                            LIMIT :limit
                            """
                        ),
                        {"reducer_id": f"substrate.{reducer_key}", "limit": limit},
                    )
                    .mappings()
                    .all()
                )
                rows = list(reversed(rows))  # newest-first fetch -> oldest-first (ASC) for the caller
        except Exception:
            return []

        out: list[float] = []
        for row in rows:
            value = row.get("error")
            if value is None:
                continue
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    def advance_node_prediction_error_baseline(
        self,
        *,
        target_id: str,
        reducer_key: str,
        alpha: float,
        min_variance: float,
        fetch_limit: int,
    ) -> PrecisionEwmaBaseline:
        """Real EWMA-baseline sibling of `load_prediction_error_history` (above)
        for Candidate A's live tick (2026-07-30 fix; see that method's own
        docstring and `orion/sentience_striving_program/README.md` §12 for the
        incident this replaces).

        Reads this target's persisted baseline row from
        `substrate_node_prediction_error_baseline` (or a cold-start zero baseline
        if none exists yet), fetches only real `substrate_reduction_receipts` rows
        strictly NEWER than the persisted cursor (`last_receipt_created_at`; no
        cursor yet -> all real rows up to `fetch_limit`, oldest-first), folds them
        into the baseline one at a time via
        `candidate_precision_weighted.advance_precision_baseline`, persists the
        advanced baseline plus the new cursor, and returns it.

        A tick with no new real receipts is a true no-op: the persisted row is
        read but never rewritten, and the unchanged baseline is returned as-is --
        "nothing new landed at this exact 2-second poll instant" must never look
        like "this target has no real history," which is exactly what the old
        every-tick rolling-window recompute conflated. `observation_count` on the
        returned baseline is therefore a real cumulative count across this
        target's entire real receipt history, immune to
        `substrate_reduction_receipts`' ~30-minute retention prune (the prune only
        ever deletes a raw receipt row after this method has already folded its
        value into the persisted baseline).

        A row whose `prediction_error` value fails to parse (missing/malformed)
        is skipped for the fold but its `created_at` still advances the cursor --
        otherwise a single permanently-malformed row would wedge this target on
        the same cursor position forever, re-fetching (and re-skipping) it every
        tick. Degrades to a cold-start zero baseline (`observation_count=0`,
        excluded by `select_node_targets` the same as any target with no real
        data) on any DB error -- never touches the persisted row in that case, so
        a transient failure cannot corrupt real accumulated state, only delay this
        one tick's read of it; the very next successful tick reads the real
        persisted baseline again. A baseline-advance failure must never crash the
        attention tick, same contract as `load_prediction_error_history`'s `[]`
        degrade.

        **Definition-version reset (2026-09-25).** A baseline summarises one
        formula's numbers. ``orion.schemas.prediction_error_definitions`` names the
        live formula version per ``reducer_key``, and the substrate runtime stamps
        it on every receipt (``after.definition_version``; unstamped = 1). If the
        persisted row was built on a different version, it restarts cold (the
        cursor is kept), and only receipts stamped with the live version are
        folded -- so a receipt the old producer wrote before a deploy cannot seed
        the new baseline, whichever service restarts first. Until the
        ``definition_version`` column exists this falls back to the pre-2026-09-25
        behaviour (fold everything, never reset) and logs a warning.
        """
        live_version = prediction_error_definition_version(reducer_key)
        try:
            with self._engine.begin() as conn:
                versioned = self._has_definition_version_column(conn)
                existing = (
                    conn.execute(
                        text(
                            """
                            SELECT ewma, variance, observation_count, last_value,
                                   last_receipt_created_at,
                                   to_jsonb(b) ->> 'definition_version' AS definition_version,
                                   to_jsonb(b) ->> 'last_value_observed_at'
                                       AS last_value_observed_at
                            FROM substrate_node_prediction_error_baseline b
                            WHERE target_id = :target_id
                            """
                        ),
                        {"target_id": target_id},
                    )
                    .mappings()
                    .first()
                )
                if existing is None:
                    baseline = PrecisionEwmaBaseline()
                    cursor = None
                else:
                    baseline = PrecisionEwmaBaseline(
                        ewma=float(existing["ewma"]),
                        variance=float(existing["variance"]),
                        observation_count=int(existing["observation_count"]),
                        last_value=(
                            float(existing["last_value"])
                            if existing["last_value"] is not None
                            else None
                        ),
                        last_observed_at=_observed_at_from_row(existing),
                    )
                    cursor = existing["last_receipt_created_at"]

                reset = False
                if versioned and existing is not None:
                    stored_version = _parse_definition_version(existing.get("definition_version"))
                    if stored_version != live_version:
                        logger.info(
                            "node_prediction_error_baseline_definition_reset target_id=%s "
                            "reducer_key=%s stored_version=%s live_version=%s "
                            "discarded_observation_count=%s",
                            target_id,
                            reducer_key,
                            stored_version,
                            live_version,
                            baseline.observation_count,
                        )
                        baseline = PrecisionEwmaBaseline()
                        reset = True

                # Strict `>` cursor comparison (code review, 2026-07-30): if two real
                # receipts for the same reducer ever land with an exactly identical
                # `created_at` (same microsecond) and only one is captured before
                # `fetch_limit` truncates a batch, the other would never be fetched
                # again once the cursor advances past its own timestamp. Not
                # observed live (each receipt is its own INSERT via a real reducer
                # tick) and not worth a compound (created_at, receipt_id) cursor for
                # a theoretical tie -- disclosed here rather than silently assumed
                # impossible.
                new_rows = (
                    conn.execute(
                        text(
                            """
                            SELECT
                                receipt_json -> 'state_deltas' -> 0 -> 'after'
                                    -> 'pressure_hints' ->> 'prediction_error' AS error,
                                receipt_json -> 'state_deltas' -> 0 -> 'after'
                                    ->> 'definition_version' AS definition_version,
                                created_at
                            FROM substrate_reduction_receipts
                            WHERE reducer_name = :reducer_id
                              AND (:cursor IS NULL OR created_at > :cursor)
                            ORDER BY created_at ASC
                            LIMIT :limit
                            """
                        ),
                        {
                            "reducer_id": f"substrate.{reducer_key}",
                            "cursor": cursor,
                            "limit": fetch_limit,
                        },
                    )
                    .mappings()
                    .all()
                )

                if not new_rows and not reset:
                    return baseline

                new_values: list[float] = []
                newest_created_at = cursor
                last_folded_at = None
                version_skipped = 0
                for row in new_rows:
                    newest_created_at = row["created_at"]
                    if versioned and (
                        _parse_definition_version(row.get("definition_version")) != live_version
                    ):
                        version_skipped += 1
                        continue
                    raw = row.get("error")
                    if raw is None:
                        continue
                    try:
                        new_values.append(float(raw))
                    except (TypeError, ValueError):
                        continue
                    last_folded_at = row["created_at"]

                if version_skipped:
                    # Receipts from a producer running a different formula version
                    # (deploy skew or a rolled-back substrate runtime). Skipped so
                    # they cannot seed this baseline -- logged so a target stuck at
                    # observation_count=0 has a visible cause.
                    logger.warning(
                        "node_prediction_error_baseline_version_skipped target_id=%s "
                        "reducer_key=%s skipped=%s live_version=%s",
                        target_id,
                        reducer_key,
                        version_skipped,
                        live_version,
                    )

                advanced = advance_precision_baseline(
                    baseline, new_values, alpha=alpha, min_variance=min_variance
                )
                if last_folded_at is not None:
                    # When the receipt behind `last_value` was written -- the
                    # staleness fade's clock (read-side only, never enters the
                    # EWMA). Persisted in its own column below, NOT re-derived from
                    # the cursor: the cursor also advances over skipped receipts
                    # (malformed / other definition version), so a substrate-only
                    # rollback emitting old-version receipts every tick would keep
                    # an old reading looking seconds old forever.
                    advanced = replace(advanced, last_observed_at=last_folded_at)

                conn.execute(
                    text(_UPSERT_BASELINE_VERSIONED_SQL if versioned else _UPSERT_BASELINE_SQL),
                    {
                        "definition_version": live_version,
                        "target_id": target_id,
                        "reducer_key": reducer_key,
                        "ewma": advanced.ewma,
                        "variance": advanced.variance,
                        "observation_count": advanced.observation_count,
                        "last_value": advanced.last_value,
                        "last_receipt_created_at": newest_created_at,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                if last_folded_at is not None and self._has_observed_at_column(conn):
                    conn.execute(
                        text(
                            """
                            UPDATE substrate_node_prediction_error_baseline
                            SET last_value_observed_at = :observed_at
                            WHERE target_id = :target_id
                            """
                        ),
                        {"observed_at": last_folded_at, "target_id": target_id},
                    )
                return advanced
        except Exception:
            # Re-probe the definition_version column next time: a dropped column
            # (migration rollback) must not wedge every advance until restart.
            self._definition_version_column = None
            self._observed_at_column = None
            logger.exception(
                "node_prediction_error_baseline_advance_failed target_id=%s", target_id
            )
            return PrecisionEwmaBaseline()

    def load_node_dominance_streak(self) -> DominanceStreak:
        """Real, persisted node-target goal-provenance dominance streak
        (`orion.attention.field_attention.goal_provenance.DominanceStreak`),
        surviving a worker restart.

        Previously this streak lived only in `AttentionRuntimeWorker.
        _node_streak`, reset to `DominanceStreak()` (count=0) on every
        process restart -- a disclosed, accepted gap when it only gated an
        internal emit-debounce (PR #1529, worst case a brief warm-up delay).
        That calculus changed once PR #1543's design doc (2026-07-31,
        docs-only, not yet implemented) proposed surfacing this exact count
        directly into `FieldGoalProvenanceV1.goal_text`, which reaches the
        real LLM-facing chat-stance prompt: a restart-truncated streak would
        then read as a small, plausible-looking-but-wrong number in that
        prompt, not just an internal delay -- so this now needs to survive
        a restart honestly.

        Degrades to a cold `DominanceStreak()` on any DB error or missing
        row (fresh deploy, or an intentionally-reset row) -- same
        never-crash-the-tick contract as this file's other loaders.
        """
        try:
            with self._engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                            """
                            SELECT target_id, count FROM substrate_goal_provenance_streak
                            WHERE streak_id = :streak_id
                            """
                        ),
                        {"streak_id": _NODE_DOMINANCE_STREAK_ID},
                    )
                    .mappings()
                    .first()
                )
        except Exception:
            logger.exception("node_dominance_streak_load_failed")
            return DominanceStreak()
        if row is None:
            return DominanceStreak()
        return DominanceStreak(target_id=row["target_id"], count=int(row["count"]))

    def save_node_dominance_streak(self, streak: DominanceStreak) -> None:
        """Persist the decision debounce, retaining its existing best-effort policy."""
        try:
            with self._engine.begin() as conn:
                self._write_node_dominance_streak(conn, streak)
        except Exception:
            logger.exception("node_dominance_streak_save_failed")

    def _write_node_dominance_streak(self, conn, streak: DominanceStreak) -> None:
        conn.execute(text("""
            INSERT INTO substrate_goal_provenance_streak (streak_id, target_id, count, updated_at)
            VALUES (:streak_id, :target_id, :count, :updated_at)
            ON CONFLICT (streak_id) DO UPDATE SET target_id = EXCLUDED.target_id,
                count = EXCLUDED.count, updated_at = EXCLUDED.updated_at
        """), dict(streak_id=_NODE_DOMINANCE_STREAK_ID, target_id=streak.target_id,
                   count=streak.count, updated_at=datetime.now(timezone.utc)))

    def load_latest_attention_frame(self) -> FieldAttentionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def load_competing_loop_refs(self, *, max_age_sec: float) -> set[str] | None:
        """Node ids the substrate's workspace competition currently holds as open
        loops -- the ``node:`` entries of every loop's ``source_refs`` in the latest
        ``substrate_attention_broadcast_projection`` (singleton row, written by
        orion-substrate-runtime every ~30s).

        Extracted in SQL: the projection row is 3.5-8.7KB and this runs on every
        ~2s goal tick, while the answer is ~55 bytes. The age bound is a WHERE
        clause on the DB clock (no container/DB skew), so a stale singleton
        returns no row and no payload.

        ``None`` when there is no fresh projection OR the row's shape is not
        what this reader expects (no ``frame.open_loops`` array): an unknown
        competition must read as unknown, never as empty -- ``set()`` means
        "fresh projection, nothing competing". The caller falls back to its
        pre-bridge behaviour on None.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT
                            jsonb_typeof(projection_json->'frame'->'open_loops') AS loops_type,
                            (
                                SELECT array_agg(DISTINCT r)
                                FROM jsonb_array_elements(
                                    CASE WHEN jsonb_typeof(projection_json->'frame'->'open_loops') = 'array'
                                         THEN projection_json->'frame'->'open_loops'
                                         ELSE '[]'::jsonb END
                                ) AS l,
                                jsonb_array_elements_text(
                                    CASE WHEN jsonb_typeof(l->'source_refs') = 'array'
                                         THEN l->'source_refs' ELSE '[]'::jsonb END
                                ) AS r
                                WHERE r LIKE 'node:%'
                            ) AS refs
                        FROM substrate_attention_broadcast_projection
                        WHERE generated_at > now() - make_interval(secs => :max_age)
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"max_age": float(max_age_sec)},
                )
                .mappings()
                .first()
            )
        if not row or row["loops_type"] != "array":
            return None
        return {str(r) for r in (row["refs"] or [])}

    def load_attention_frame_for_field_tick(self, tick_id: str) -> FieldAttentionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        WHERE source_field_tick_id = :tick_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"tick_id": tick_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def save_attention_frame(
        self, frame: FieldAttentionFrameV1, *,
        streak: DominanceStreak | None = None,
        target_kind: str | None = None,
        min_streak: int = 1,
    ) -> bool:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO substrate_attention_frames (
                        frame_id,
                        source_field_tick_id,
                        source_field_generated_at,
                        generated_at,
                        policy_id,
                        frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_field_tick_id,
                        :source_field_generated_at,
                        :generated_at,
                        :policy_id,
                        :frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO NOTHING
                    RETURNING frame_id
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "source_field_generated_at": frame.source_field_generated_at,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.attention_policy_id,
                    "frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
            if result.scalar() is None:
                return False
            if streak is not None:
                # Recording must never stop a frame or goal. The savepoint keeps
                # a recorder SQL error from poisoning the outer frame transaction.
                try:
                    with conn.begin_nested():
                        self._save_dominance_tick(conn, frame, streak, target_kind, min_streak)
                except Exception:
                    logger.exception("attention_focus_record_failed frame_id=%s tick_id=%s",
                                     frame.frame_id, frame.source_field_tick_id)
                # Preserve the old best-effort debounce persistence contract too.
                try:
                    with conn.begin_nested():
                        self._write_node_dominance_streak(conn, streak)
                except Exception:
                    logger.exception("node_dominance_streak_save_failed")
        return True

    def _save_dominance_tick(self, conn, frame, streak, target_kind, min_streak) -> None:
        # The row lock serializes checkpoint updates. Frame PK dedupe above
        # and this transaction make retrying a failed/duplicate tick harmless.
        conn.execute(text("""
            INSERT INTO substrate_goal_provenance_streak (streak_id, count)
            VALUES (:id, 0) ON CONFLICT (streak_id) DO NOTHING
        """), {"id": _NODE_DOMINANCE_STREAK_ID})
        row = conn.execute(text("""
            SELECT target_id, count, run_state,
                (SELECT source_field_tick_id FROM substrate_attention_frames
                 WHERE frame_id <> :frame_id ORDER BY generated_at DESC LIMIT 1)
                    AS previous_field_tick_id
            FROM substrate_goal_provenance_streak
            WHERE streak_id = :id FOR UPDATE
        """), {"id": _NODE_DOMINANCE_STREAK_ID, "frame_id": frame.frame_id}).mappings().one()
        prior_state = row["run_state"]
        if prior_state and prior_state["last_field_tick_id"] != row["previous_field_tick_id"]:
            # A saved frame without a recorder advance means a failure/disabled
            # period. Its winner is unknown: discard the incomplete open run,
            # rather than silently claiming uninterrupted focus across the gap.
            logger.warning("attention_focus_observation_gap frame_id=%s checkpoint_tick_id=%s",
                           frame.frame_id, prior_state["last_field_tick_id"])
            prior_state = None
        state, completed = advance_run(
            prior_state, target_id=streak.target_id, target_kind=target_kind,
            observed_at=frame.generated_at, field_tick_id=frame.source_field_tick_id,
            frame_id=frame.frame_id, min_streak=min_streak,
            left_censored=(prior_state is None and row["target_id"] == streak.target_id
                           and row["count"] > 0),
        )
        if completed is not None:
            conn.execute(text("""
                INSERT INTO field_dominance_run (
                    run_id, target_id, target_kind, started_at, ended_at, tick_count,
                    min_streak_at_run, first_source_attention_frame_id,
                    last_source_attention_frame_id, left_censored
                ) VALUES (
                    :run_id, :target_id, :target_kind, :started_at, :ended_at, :tick_count,
                    :min_streak_at_run, :first_source_attention_frame_id,
                    :last_source_attention_frame_id, :left_censored
                ) ON CONFLICT (run_id) DO NOTHING
            """), completed.model_dump())
        conn.execute(text("""
            UPDATE substrate_goal_provenance_streak
            SET run_state = :run_state WHERE streak_id = :id
        """), dict(id=_NODE_DOMINANCE_STREAK_ID, run_state=Json(state)))

    def prune_attention_frames(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        while True:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(PRUNE_ATTENTION_FRAMES_SQL),
                    {"cutoff": cutoff, "batch_size": batch_size},
                )
            deleted = result.rowcount or 0
            total_deleted += deleted
            if deleted < batch_size:
                break
        return total_deleted

    def attention_frame_oldest_age_hours(self) -> float | None:
        # Keys on created_at -- the same column PRUNE_ATTENTION_FRAMES_SQL's cutoff
        # filters on -- so staleness detection can never disagree with the pruner
        # about which column defines "age".
        with self._engine.connect() as conn:
            row = (
                conn.execute(text("SELECT min(created_at) AS oldest FROM substrate_attention_frames"))
                .mappings()
                .first()
            )
        oldest = row["oldest"] if row else None
        if oldest is None:
            return None
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - oldest).total_seconds() / 3600.0
