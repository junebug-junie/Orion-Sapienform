from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta, timezone
# NOT `import time`: this module already imports datetime.time above, and the plain module
# import loses that race silently -- AttributeError: type object 'datetime.time' has no
# attribute 'monotonic', at runtime, on the reconciler path only.
from time import monotonic as _monotonic

from psycopg2.extras import Json
from pydantic import ValidationError
from orion.autonomy.contrast import TreatedCellKey
from orion.autonomy.prediction import EffectPosterior
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.substrate.bus_synaptic_surprise import (
    latest_bus_synaptic_prediction_error as _shared_latest_bus_synaptic_prediction_error,
)

logger = logging.getLogger("orion.execution_dispatch.runtime.store")


def _coerce_starvation_counts(raw: object) -> dict[str, int]:
    """Persisted starvation counters -> a usable dict, never an exception.

    Rows written before this field existed have no key at all, and this value
    only ever feeds an admission-ordering bonus -- a malformed one should cost
    the aging bonus for a tick, not stall the dispatch runtime. Returns {}
    rather than None: absent and empty genuinely mean the same thing here
    (nothing is starving), unlike the EWMA baselines beside it where
    "never seeded" is a distinct state that has to be told apart.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


class ExecutionDispatchRuntimeStore:
    def __init__(self, postgres_uri: str, *, reconcile_interval_sec: float = 900.0) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )
        # Seeded to NOW, not None: otherwise the expensive full-table anti-join in
        # reconcile_dispatch_pending runs on the first tick of every process start, and a crash
        # loop would re-run it per restart -- defeating the rate limit that makes it safe to
        # call every tick.
        self._reconcile_interval_sec = float(reconcile_interval_sec)
        self._last_reconcile_mono: float | None = _monotonic()

    def _validate_policy_frame_row(
        self, payload: object, *, log_label: str
    ) -> PolicyDecisionFrameV1 | None:
        """Shared validate-or-retire logic for a single raw policy_decision_
        frame_json payload. Returns None (and retires the row via a stub
        dispatch frame) on schema-validation failure -- see
        _retire_incompatible_policy_frame's own docstring for why this must
        never just re-degrade to None without retiring: a naive skip would
        re-select the exact same incompatible row forever."""
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return PolicyDecisionFrameV1.model_validate(payload)
        except ValidationError:
            raw_frame_id = payload.get("frame_id") if isinstance(payload, dict) else None
            raw_proposal_frame_id = (
                payload.get("source_proposal_frame_id") if isinstance(payload, dict) else None
            )
            logger.warning(
                "policy_decision_frame_incompatible_schema %s frame_id=%s",
                log_label,
                raw_frame_id,
                exc_info=True,
            )
            if raw_frame_id:
                self._retire_incompatible_policy_frame(raw_frame_id, raw_proposal_frame_id)
            return None

    def load_oldest_policy_frames_without_dispatch(
        self, limit: int
    ) -> list[PolicyDecisionFrameV1]:
        """Up to `limit` oldest unprocessed policy frames in ONE query
        (2026-07-30, docs/superpowers/specs/2026-07-30-execution-dispatch-
        staleness-discard-design.md's "Part 1c" -- a real, EXPLAIN-ANALYZE-
        verified performance fix, not a guess). This replaces what used to
        be a `LIMIT 1` query called up to MAX_STALE_DISCARDS_PER_TICK (200)
        times per tick from a while loop in _drain_stale_policy_frames.

        Real numbers (live, 2026-07-30): this anti-join (`LEFT JOIN ... WHERE
        d.frame_id IS NULL`) costs ~280-300ms per call regardless of LIMIT
        size (Postgres builds the full hash-joined result before sorting for
        LIMIT either way -- confirmed via EXPLAIN ANALYZE, LIMIT 1 costs
        ~280ms, LIMIT 200 costs ~327ms, not 200x more). Calling it once per
        tick with LIMIT=200 instead of 200 times with LIMIT=1 cuts real
        per-tick SELECT cost from ~56s to ~0.3s -- this was the actual
        dominant cost behind the ~75s/tick cadence observed live after the
        staleness-discard + fresh-priority-fallback patches shipped (traced
        via EXPLAIN ANALYZE, adversarially re-tested against stale-planner-
        statistics as an alternative explanation -- ruled out, a fresh
        ANALYZE on both tables left the plan and cost unchanged).

        SUPERSEDED 2026-08-19 (ROADMAP D2): this is no longer a join at all. It reads
        `p.dispatch_pending` off a PARTIAL index containing only unprocessed rows, so the
        "huge prefix of already-processed ancient history" the analysis below is about does not
        exist to be walked. That analysis was correct and is kept because it explains why the
        obvious `NOT EXISTS` rewrite was NOT the answer -- the marker is. Batching the LIMIT,
        the actual fix from that patch, is unchanged and still right for reading a backlog in
        chunks; only the cost of FINDING each chunk changed (829 MB hash join -> index scan).
        See services/orion-sql-db/manual_migration_policy_dispatch_pending_marker.sql.

        Historical note, still accurate for the join shape it describes:
        Deliberately still the `LEFT JOIN` shape here, not `NOT EXISTS`: a
        `NOT EXISTS` rewrite is dramatically cheaper for the DESC "freshest"
        direction (load_freshest_policy_frame_without_dispatch below,
        ~1500x) precisely because almost nothing near "now" has been
        processed yet, so a nested-loop anti-join terminates on the first
        probe -- but for THIS ascending direction, a huge prefix of already-
        processed ancient history (predating the backlog that motivated this
        whole feature) sits before the real backlog start, and a nested loop
        would have to walk hundreds of thousands of already-matched rows
        before finding the first true miss (confirmed live: 6+ seconds,
        *slower* than the Hash Join it would replace). Batching the LIMIT is
        the real fix for this direction; the query shape itself is already
        the cheaper available plan for oldest-first access.

        Each row is validated individually -- a schema-incompatible row
        (see _validate_policy_frame_row) is retired and simply excluded from
        the returned list rather than aborting the whole batch, so one bad
        historical row can't block every valid one fetched alongside it.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT p.policy_decision_frame_json, p.frame_id
                        FROM substrate_policy_decision_frames p
                        WHERE p.dispatch_pending
                        ORDER BY p.generated_at ASC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        frames: list[PolicyDecisionFrameV1] = []
        for row in rows:
            frame = self._validate_policy_frame_row(
                row["policy_decision_frame_json"], log_label="oldest_batch_lookup"
            )
            if frame is None:
                continue
            if self._already_dispatched(frame.frame_id):
                continue
            frames.append(frame)
        return frames

    def load_freshest_policy_frame_without_dispatch(self) -> PolicyDecisionFrameV1 | None:
        """The single newest unprocessed policy frame, regardless of backlog
        depth (2026-07-30, docs/superpowers/specs/2026-07-30-execution-
        dispatch-staleness-discard-design.md's own follow-up finding: a real
        deep backlog fully starved real-time dispatch under the FIFO-only
        design, since _drain_stale_policy_frames' oldest-first walk can spend
        its entire per-tick discard budget on old backlog without ever
        reaching "now"). _tick() checks this as a fallback whenever the FIFO
        drain doesn't surface a candidate to process, so a genuinely current
        proposal is never gated behind however deep the old backlog is.

        `NOT EXISTS`, not `LEFT JOIN ... WHERE d.frame_id IS NULL` (2026-07-30
        "Part 1c" perf fix, same design doc as load_oldest_policy_frames_
        without_dispatch above -- see that method's docstring for the full
        account of why the two directions need DIFFERENT query shapes).
        Confirmed live via EXPLAIN ANALYZE: this exact rewrite took the LEFT
        JOIN version's ~294ms down to ~0.19ms for this specific (DESC)
        direction -- almost nothing near "now" has been processed yet, so
        Postgres's nested-loop anti-join plan terminates on the very first
        probe instead of building a full hash join over both tables.

        Same schema-validation-failure handling as load_oldest_policy_frames_
        without_dispatch above (retire an incompatible row via a stub
        dispatch frame rather than re-selecting it forever) -- this query can
        hit the exact same incompatible-row case, just approached from the
        newest end instead of the oldest.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT p.policy_decision_frame_json, p.frame_id
                        FROM substrate_policy_decision_frames p
                        WHERE p.dispatch_pending
                        ORDER BY p.generated_at DESC
                        LIMIT 1
                        """
                    ),
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        frame = self._validate_policy_frame_row(
            row["policy_decision_frame_json"], log_label="freshest_lookup"
        )
        if frame is not None and self._already_dispatched(frame.frame_id):
            return None
        return frame

    def _already_dispatched(self, policy_frame_id: str) -> bool:
        """Restores, at the read boundary, the guarantee the anti-join used to give for free.

        `WHERE d.frame_id IS NULL` could not return an already-dispatched frame -- it looked. A
        marker can be stale-true (every pre-migration row is, until backfilled; so is anything
        restored from a backup taken mid-migration), and re-dispatching is NOT a harmless
        repeat: it would run a real cortex action a second time against a policy decision that
        has already been acted on. Checked here rather than in the two workers' call sites, so
        both read directions are covered by one guard and neither can be updated without it.

        A stale marker found this way is cleared, in a batch, so the FIFO advances instead of
        re-selecting the same rows every tick.
        """
        if self.load_dispatch_frame_for_policy_frame(policy_frame_id) is None:
            return False
        logger.info("dispatch_pending_stale_marker policy_frame_id=%s", policy_frame_id)
        self.clear_dispatch_pending(policy_frame_id)
        return True

    def clear_dispatch_pending(self, policy_frame_id: str, *, drain_batch: int = 5000) -> int:
        """Clear this row's marker, plus a batch of others that already have a dispatch frame.

        The marker defaults to TRUE, so every pre-migration row starts pending. Clearing one at
        a time would put 423k stale rows ahead of real work under `ORDER BY generated_at ASC`.
        The batch is the migration's own backfill statement and rides the partial index.
        """
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE substrate_policy_decision_frames
                       SET dispatch_pending = false
                     WHERE frame_id = :frame_id
                """),
                {"frame_id": policy_frame_id},
            )
            result = conn.execute(
                text("""
                    UPDATE substrate_policy_decision_frames u
                       SET dispatch_pending = false
                     WHERE u.frame_id IN (
                           SELECT x.frame_id
                             FROM substrate_policy_decision_frames x
                             JOIN substrate_execution_dispatch_frames d
                               ON d.source_policy_frame_id = x.frame_id
                            WHERE x.dispatch_pending
                            LIMIT :batch
                     )
                """),
                {"batch": max(0, int(drain_batch))},
            )
        drained = 1 + int(result.rowcount or 0)
        if drained > 1:
            logger.info("dispatch_pending_bulk_drained cleared=%s", drained)
        return drained

    def reconcile_dispatch_pending(self, *, force: bool = False) -> int:
        """Re-queue any policy frame whose marker was cleared without a dispatch frame existing.

        Only ever sets the marker TRUE -- it can add work, never remove it, so a bug here costs
        duplicated effort rather than lost effort. It IS the expensive anti-join the marker
        exists to avoid, hence rate-limited rather than run on every tick.
        """
        now = _monotonic()
        if not force and self._last_reconcile_mono is not None:
            if (now - self._last_reconcile_mono) < self._reconcile_interval_sec:
                return 0
        self._last_reconcile_mono = now
        with self._engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE substrate_policy_decision_frames p
                       SET dispatch_pending = true
                     WHERE NOT p.dispatch_pending
                       AND NOT EXISTS (
                             SELECT 1 FROM substrate_execution_dispatch_frames d
                              WHERE d.source_policy_frame_id = p.frame_id
                       )
                """)
            )
        requeued = int(result.rowcount or 0)
        if requeued:
            logger.warning(
                "dispatch_pending_reconciled requeued=%s -- policy frames had their pending "
                "marker cleared with no dispatch frame present. Work would have been lost.",
                requeued,
            )
        return requeued

    def _retire_incompatible_policy_frame(
        self, raw_frame_id: str, raw_proposal_frame_id: str | None
    ) -> None:
        """Insert a stub, unattempted execution_dispatch_frame for a policy
        frame that failed schema validation, so load_oldest_policy_frames_
        without_dispatch's/load_freshest_policy_frame_without_dispatch's
        lookups don't re-select this exact row forever."""
        stub = ExecutionDispatchFrameV1(
            frame_id=f"execution.dispatch.frame:{raw_frame_id}:schema_incompatible",
            generated_at=datetime.now(timezone.utc),
            source_policy_frame_id=raw_frame_id,
            source_proposal_frame_id=raw_proposal_frame_id or "unknown",
            dispatch_attempted=False,
            warnings=["source_policy_frame_schema_incompatible"],
        )
        self.save_dispatch_frame(stub)

    def load_proposal_frame(self, frame_id: str) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json FROM substrate_proposal_frames
                        WHERE frame_id = :frame_id
                        LIMIT 1
                        """
                    ),
                    {"frame_id": frame_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["proposal_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ProposalFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning(
                "proposal_frame_incompatible_schema frame_id=%s", frame_id, exc_info=True
            )
            return None

    def load_dispatch_frame_for_policy_frame(
        self, policy_frame_id: str
    ) -> ExecutionDispatchFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT dispatch_frame_json
                        FROM substrate_execution_dispatch_frames
                        WHERE source_policy_frame_id = :policy_frame_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"policy_frame_id": policy_frame_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["dispatch_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ExecutionDispatchFrameV1.model_validate(payload)
        except ValidationError:
            # Looked up by a fixed policy_frame_id, so a naive raise would
            # permanently block this caller on a schema-incompatible
            # historical row.
            logger.warning(
                "dispatch_frame_incompatible_schema policy_frame_id=%s", policy_frame_id, exc_info=True
            )
            return None

    def load_latest_dispatch_frame(self) -> ExecutionDispatchFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT dispatch_frame_json
                        FROM substrate_execution_dispatch_frames
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
        payload = row["dispatch_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ExecutionDispatchFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning("dispatch_frame_incompatible_schema latest_lookup", exc_info=True)
            return None

    def save_dispatch_frame(self, frame: ExecutionDispatchFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_execution_dispatch_frames (
                        frame_id,
                        source_policy_frame_id,
                        source_proposal_frame_id,
                        source_field_tick_id,
                        generated_at,
                        policy_id,
                        dispatch_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_policy_frame_id,
                        :source_proposal_frame_id,
                        :source_field_tick_id,
                        :generated_at,
                        :policy_id,
                        :dispatch_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_policy_frame_id = EXCLUDED.source_policy_frame_id,
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        dispatch_frame_json = EXCLUDED.dispatch_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_policy_frame_id": frame.source_policy_frame_id,
                    "source_proposal_frame_id": frame.source_proposal_frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.execution_dispatch_policy_id,
                    "dispatch_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
            # ROADMAP D2, 2026-08-19. SAME TRANSACTION as the insert: a crash between the two
            # would otherwise either lose the work (marker cleared, no dispatch frame) or
            # reprocess it, and only one of those is recoverable.
            conn.execute(
                text("""
                    UPDATE substrate_policy_decision_frames
                       SET dispatch_pending = false
                     WHERE frame_id = :frame_id
                """),
                {"frame_id": frame.source_policy_frame_id},
            )

    def save_dispatch_result(
        self,
        *,
        result_id: str,
        dispatch_id: str,
        frame_id: str,
        status: str,
        result_json: dict,
        raw_len: int,
        latency_ms: float | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_dispatch_results (
                        result_id, dispatch_id, frame_id, status, result_json, raw_len,
                        latency_ms, created_at
                    ) VALUES (
                        :result_id, :dispatch_id, :frame_id, :status, :result_json, :raw_len,
                        :latency_ms, :created_at
                    )
                    ON CONFLICT (result_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result_json = EXCLUDED.result_json,
                        raw_len = EXCLUDED.raw_len
                    """
                ),
                {
                    "result_id": result_id,
                    "dispatch_id": dispatch_id,
                    "frame_id": frame_id,
                    "status": status,
                    "result_json": Json(result_json),
                    "raw_len": raw_len,
                    "created_at": now,
                },
            )

    def load_dispatch_result_by_dispatch_id(self, dispatch_id: str) -> dict | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT result_id, status, result_json, raw_len
                        FROM substrate_dispatch_results
                        WHERE dispatch_id = :dispatch_id
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"dispatch_id": dispatch_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        result_json = row["result_json"]
        if isinstance(result_json, str):
            result_json = json.loads(result_json)
        return {
            "result_id": row["result_id"],
            "status": row["status"],
            "result_json": result_json,
            "raw_len": row["raw_len"],
        }

    def sum_risk_dispatched_today(self) -> float:
        """Real cumulative risk_score spent today, not a blind action count.

        Replaces count_dispatches_today() (2026-07-26) -- a flat count
        couldn't distinguish five trivial risk_score~0.05 inspects from five
        genuinely higher-risk candidates. Reads `dispatched_candidates` off
        `substrate_execution_dispatch_frames` directly -- each candidate
        already carries its own real, already-computed `risk_score`
        (ExecutionDispatchCandidateV1.risk_score); no schema migration or
        new column needed, this is purely a smarter read of data already
        being written.
        """
        # Explicit UTC bound computed in Python, not date_trunc('day', now())
        # -- matches this file's own datetime.now(timezone.utc) convention
        # elsewhere and doesn't depend on the Postgres session's configured
        # timezone (confirmed Etc/UTC live, but not worth relying on).
        today_start_utc = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT COALESCE(SUM((cand.value ->> 'risk_score')::float), 0.0) AS total_risk
                    FROM substrate_execution_dispatch_frames f
                    CROSS JOIN LATERAL jsonb_array_elements(
                        f.dispatch_frame_json -> 'dispatched_candidates'
                    ) AS cand(value)
                    WHERE f.created_at >= :today_start
                      -- jsonb_array_elements() errors on a literal JSON null
                      -- (as opposed to a missing key or a real [] array) --
                      -- not reachable today (Pydantic's default_factory=list
                      -- means normal serialization can't produce one, and
                      -- 0 of today's real rows have this shape, confirmed
                      -- live), but this guard means a future/legacy row of
                      -- that shape degrades that row to zero contribution
                      -- instead of throwing this whole query.
                      AND jsonb_typeof(f.dispatch_frame_json -> 'dispatched_candidates') = 'array'
                    """
                ),
                {"today_start": today_start_utc},
            ).mappings().first()
        return float(row["total_risk"]) if row else 0.0

    def sum_uncapped_risk_for_day(self, day_start_utc: datetime, day_end_utc: datetime) -> float:
        """Real, *uncapped* risk demand for one UTC calendar day -- the
        self-calibrating daily risk ceiling's own feedstock, NOT the same
        thing as sum_risk_dispatched_today() (which stays as-is, still the
        right read for "what did we actually spend against the cap this
        tick").

        Why this has to be a separate method, not a reuse of
        sum_risk_dispatched_today(): once the derived cap enforces a real
        ceiling (this patch), actual dispatched risk is right-censored at
        whatever that ceiling was that day -- it can never exceed it, so it
        structurally cannot report true demand back into the thing that sets
        the ceiling. Feeding a capped value into next day's baseline would
        recreate the exact "clamped value masks true magnitude" disease this
        whole fix exists to kill, one layer down (the fixed
        ORION_DISPATCH_MAX_RISK_PER_DAY=10.0 constant clamping real demand at
        exactly 10.00/day on 2026-07-26 and 2026-07-27, only visible once
        advisory-only briefly lifted the clamp on 2026-07-28 and real demand
        turned out to be 817.65 -- 80x the enforced number).

        Sums risk_score across every candidate that existed that day
        regardless of whether it actually got sent: `dispatch_status ==
        'prepared_for_dispatch'` entries in `candidates` (a candidate that
        was ready to send but didn't fit that tick's remaining budget --
        build_execution_dispatch_frame/worker.py never retries these, so
        this is the only place their risk_score is ever counted) PLUS every
        entry already in `dispatched_candidates` (already real spend, and
        every entry there already carries `dispatch_status == 'dispatched'`
        by application-level construction -- every append site in
        worker.py's `_send_one` sets it explicitly; nothing at the Pydantic
        schema level itself forbids a different status from landing there).
        Together this is real demand for the day,
        uninfluenced by whatever cap happened to be enforced that day.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT COALESCE(SUM(total_risk), 0.0) AS total_risk
                        FROM (
                            SELECT (cand.value ->> 'risk_score')::float AS total_risk
                            FROM substrate_execution_dispatch_frames f
                            CROSS JOIN LATERAL jsonb_array_elements(
                                f.dispatch_frame_json -> 'candidates'
                            ) AS cand(value)
                            WHERE f.created_at >= :day_start
                              AND f.created_at < :day_end
                              AND jsonb_typeof(f.dispatch_frame_json -> 'candidates') = 'array'
                              AND cand.value ->> 'dispatch_status' = 'prepared_for_dispatch'

                            UNION ALL

                            SELECT (cand.value ->> 'risk_score')::float AS total_risk
                            FROM substrate_execution_dispatch_frames f
                            CROSS JOIN LATERAL jsonb_array_elements(
                                f.dispatch_frame_json -> 'dispatched_candidates'
                            ) AS cand(value)
                            WHERE f.created_at >= :day_start
                              AND f.created_at < :day_end
                              AND jsonb_typeof(f.dispatch_frame_json -> 'dispatched_candidates') = 'array'
                        ) AS combined
                        """
                    ),
                    {"day_start": day_start_utc, "day_end": day_end_utc},
                )
                .mappings()
                .first()
            )
        return float(row["total_risk"]) if row and row["total_risk"] is not None else 0.0

    def load_latest_daily_risk_baseline(self) -> dict | None:
        """Latest persisted daily-risk EWMA baseline state, read off whichever
        dispatch frame was saved most recently (every saved frame carries the
        current baseline state forward, per ExecutionDispatchFrameV1's own
        docstring, so this is correct regardless of which tick's frame is
        newest). Returns None only when there are no dispatch frame rows at
        all (a truly first-ever tick); a real row with these fields absent
        (a pre-migration row from before this patch) degrades to the
        cold-start defaults (ewma/var=0.0, n=0, last_day=None), not None --
        the caller can't tell "no history" from "history predates this
        field" any other way, and both should behave the same: fall through
        to the cold-start seed path.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT
                            dispatch_frame_json ->> 'daily_risk_baseline_ewma' AS ewma,
                            dispatch_frame_json ->> 'daily_risk_baseline_ewma_var' AS var,
                            dispatch_frame_json ->> 'daily_risk_baseline_ewma_n' AS n,
                            dispatch_frame_json ->> 'daily_risk_baseline_last_day' AS last_day
                        FROM substrate_execution_dispatch_frames
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        return {
            "daily_risk_baseline_ewma": float(row["ewma"]) if row["ewma"] is not None else 0.0,
            "daily_risk_baseline_ewma_var": float(row["var"]) if row["var"] is not None else 0.0,
            "daily_risk_baseline_ewma_n": int(row["n"]) if row["n"] is not None else 0,
            "daily_risk_baseline_last_day": row["last_day"],
        }

    def load_latest_staleness_discard_baseline(self) -> dict | None:
        """Latest persisted staleness-discard-count EWMA baseline state, same
        carried-forward-on-every-frame read pattern as
        load_latest_daily_risk_baseline() above -- see that method's own
        docstring for why None only means "no dispatch frame rows exist at
        all" and a pre-migration row degrades to cold-start defaults instead.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT
                            dispatch_frame_json ->> 'staleness_discard_count_ewma' AS ewma,
                            dispatch_frame_json ->> 'staleness_discard_count_ewma_var' AS var,
                            dispatch_frame_json ->> 'staleness_discard_count_ewma_n' AS n,
                            dispatch_frame_json -> 'starvation_counts' AS starvation_counts
                        FROM substrate_execution_dispatch_frames
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        return {
            "staleness_discard_count_ewma": float(row["ewma"]) if row["ewma"] is not None else 0.0,
            "staleness_discard_count_ewma_var": float(row["var"]) if row["var"] is not None else 0.0,
            "staleness_discard_count_ewma_n": int(row["n"]) if row["n"] is not None else 0,
            # 2026-08-12: starvation counters ride along on the SAME row this
            # already reads rather than in a second query. Not just cheaper --
            # a separate `ORDER BY created_at DESC LIMIT 1` could resolve to a
            # different row under concurrent inserts, and the two states would
            # then be carried forward from different ticks.
            #
            # This carry has to be real for aging to work at all: stale-discard
            # frames are saved rows too (worker._drain_stale_policy_frames
            # stamps them with this whole dict), so without it one discard tick
            # would zero every counter -- silently, in exactly the direction
            # that keeps the starved thing starved.
            "starvation_counts": _coerce_starvation_counts(row["starvation_counts"]),
        }

    def most_recent_closed_day_with_data(
        self, before_day_start_utc: datetime
    ) -> tuple[str, float] | None:
        """One-time cold-start seed: the most recent UTC calendar day
        strictly before `before_day_start_utc` that has any dispatch frames
        with real candidate data, and that day's real uncapped risk total
        (via sum_uncapped_risk_for_day). Used only when the baseline has
        never been seeded (load_latest_daily_risk_baseline() reports n==0,
        last_day is None) -- seeds sample #1 from real closed-day history
        (2026-07-28's 817.65) instead of inventing a hardcoded starting
        constant. Returns None if no historical day has any real candidate
        data (should never actually trigger against this repo's real
        history; the caller falls back to the static
        ORION_DISPATCH_MAX_RISK_PER_DAY setting in that case).

        `created_at AT TIME ZONE 'UTC'` (not `date_trunc`) so this is robust
        to whatever the connecting session's own timezone is configured as
        (confirmed Etc/UTC live, but not worth relying on) -- converts the
        stored timestamptz to a UTC wall-clock timestamp before truncating
        to a date, matching this file's own explicit-UTC-bounds convention
        elsewhere.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT (created_at AT TIME ZONE 'UTC')::date AS day
                        FROM substrate_execution_dispatch_frames
                        WHERE created_at < :before_day_start
                          AND (
                            jsonb_array_length(
                                COALESCE(dispatch_frame_json -> 'candidates', '[]'::jsonb)
                            ) > 0
                            OR jsonb_array_length(
                                COALESCE(dispatch_frame_json -> 'dispatched_candidates', '[]'::jsonb)
                            ) > 0
                          )
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"before_day_start": before_day_start_utc},
                )
                .mappings()
                .first()
            )
        if not row or row["day"] is None:
            return None
        day: date = row["day"]
        day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        total = self.sum_uncapped_risk_for_day(day_start, day_end)
        return (day.isoformat(), total)

    def latest_bus_synaptic_prediction_error(self) -> float | None:
        """Real, live surprise signal for `ActionOutcomeEmitV1.surprise` -- replaces the
        hardcoded `0.0` this service previously emitted (a disclosed, honest placeholder
        per docs/superpowers/specs/2026-07-13-autonomy-experience-loop-p2-design.md, not a
        bug, but never a real signal either).

        Delegates to `orion.substrate.bus_synaptic_surprise.
        latest_bus_synaptic_prediction_error()` (see that function's docstring for the
        full provenance/staleness rationale) -- extracted there 2026-07-28 so
        `orion-spark-concept-induction`'s emitters reuse the exact same query and
        staleness logic instead of a second, divergence-prone copy.
        """
        return _shared_latest_bus_synaptic_prediction_error(self._engine)

    def recent_dispatch_result_statuses(self, limit: int = 10) -> list[str]:
        """Kept, not removed: no longer called by worker.py's theater tripwire
        as of 2026-07-25 (that check moved to an in-process deque so a
        restart gives it a genuinely clean slate -- querying this table let
        stale pre-restart rows defeat "restart to re-arm" in a real
        incident). Still real, still tested (test_execution_dispatch_
        runtime_store.py), left in place as a real building block for a
        possible future debug/inspection endpoint over actual historical
        Postgres data, not the live tripwire's own decision.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT status
                    FROM substrate_dispatch_results
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        return [str(row["status"]) for row in rows]

    def load_effect_posteriors(self) -> dict[TreatedCellKey, EffectPosterior]:
        """Current per-(kind, target, signal, baseline_bin) belief.

        Full read of a tiny, primary-keyed table (tens of rows). Explicitly
        NOT the newest-row-of-a-big-table pattern used by
        load_latest_daily_risk_baseline: that pattern, applied to the 2 GB
        dispatch-frame table, is 49.8% of this database's entire buffer
        traffic as of 2026-08-20, and repeating it for a value read on every
        single tick would make the same mistake twice.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT dispatch_kind, target_id, signal_id, baseline_bin,
                               posterior_mean, posterior_variance, posterior_n
                          FROM substrate_action_effect_posterior
                        """
                    )
                )
                .mappings()
                .all()
            )
        return {
            (
                r["dispatch_kind"],
                r["target_id"],
                r["signal_id"],
                int(r["baseline_bin"]),
            ): EffectPosterior(
                mean=float(r["posterior_mean"]),
                variance=float(r["posterior_variance"]),
                n=int(r["posterior_n"]),
            )
            for r in rows
        }
