from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from orion.autonomy.contrast import ControlCell, ControlCellKey, TreatedCellKey
from orion.autonomy.prediction import EffectPosterior
from orion.schemas.action_prediction import ActionOutcomeRecordV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1

logger = logging.getLogger("orion.feedback_runtime.store")


def _field_from_json(payload) -> FieldStateV1 | None:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return None
    return FieldStateV1.model_validate(payload)


class FeedbackRuntimeStore:
    def __init__(
        self,
        postgres_uri: str,
        *,
        reconcile_interval_sec: float = 900.0,
    ) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )
        # The safety net for the `feedback_pending` marker. See reconcile_feedback_pending.
        self._reconcile_interval_sec = float(reconcile_interval_sec)
        # Seeded to NOW, not None: otherwise the expensive full-table anti-join runs on the
        # first tick of every process start, and a crash loop (this service has documented
        # schema-incompat stalls) would re-run it on each restart -- defeating the rate limit
        # that is the entire reason the reconciler is safe to call every tick.
        self._last_reconcile_mono: float | None = time.monotonic()

    # ROADMAP D2 follow-through, 2026-08-19. This lookup WAS athena's I/O ceiling.
    #
    # It used to ask "which dispatch frame has no feedback frame yet" as an unbounded anti-join
    # over both full tables: 106,052 blocks read (829 MB) plus 465 MB spilled to temp, PER
    # EXECUTION, every FEEDBACK_POLL_INTERVAL_SEC.
    #
    # A time bound was tried first and REVERTED -- it strands the backlog. This pipeline
    # legitimately runs hours to days behind (2026-08-14: 29,264 frames processed at ~34h of
    # age), and any "recent rows only" window silently abandons everything older once fresh work
    # keeps the fast path busy. See the migration
    # services/orion-sql-db/manual_migration_substrate_pending_markers.sql.
    #
    # The marker makes the question O(pending) instead of O(history), which is correct at ANY
    # backlog depth. `feedback_pending` defaults to TRUE, so anything new -- or anything this
    # code has never seen -- is work, never silently skipped. It is cleared inside the same
    # transaction as the feedback insert (see save_feedback_frame), and a periodic reconciler
    # (reconcile_feedback_pending) re-sets it for any row that lost it without a feedback frame
    # actually existing. The reconciler can only ADD work back, never remove it.
    _PENDING_SQL = text("""
        SELECT d.dispatch_frame_json, d.generated_at
        FROM substrate_execution_dispatch_frames d
        WHERE d.feedback_pending
        ORDER BY d.generated_at ASC
        LIMIT 1
    """)

    def load_latest_dispatch_frame_without_feedback(self) -> ExecutionDispatchFrameV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(self._PENDING_SQL).mappings().first()
        if not row:
            return None
        payload = row["dispatch_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ExecutionDispatchFrameV1.model_validate(payload)
        except ValidationError:
            # Live incident (2026-07-22, immediately after the SelfStateV1
            # burn deploy): this is the FIFO lookup (oldest dispatch frame
            # without feedback yet) -- a bare None-degrade re-selects this
            # exact row every tick forever (confirmed live: feedback-runtime
            # sat silently stuck on one legacy dispatch frame for 15+
            # minutes, producing nothing, while proposal/policy/execution-
            # dispatch-runtime all correctly drained their own backlogs via
            # the sibling retirement fix in
            # orion-policy-runtime/orion-execution-dispatch-runtime's
            # store.py). Retire the bad row with a stub "unevaluable"
            # feedback frame so the FIFO actually advances.
            raw_frame_id = payload.get("frame_id") if isinstance(payload, dict) else None
            logger.warning(
                "dispatch_frame_incompatible_schema fifo_lookup frame_id=%s",
                raw_frame_id,
                exc_info=True,
            )
            if raw_frame_id:
                self._retire_incompatible_dispatch_frame(raw_frame_id)
            return None

    def _retire_incompatible_dispatch_frame(self, raw_frame_id: str) -> None:
        """Insert a stub 'unevaluable' feedback_frame for a dispatch frame
        that failed schema validation, so
        load_latest_dispatch_frame_without_feedback's FIFO lookup doesn't
        re-select this exact row forever."""
        stub = FeedbackFrameV1(
            frame_id=f"feedback.frame:{raw_frame_id}:schema_incompatible",
            generated_at=datetime.now(timezone.utc),
            source_execution_dispatch_frame_id=raw_frame_id,
            outcome_status="unknown",
            outcome_score=0.0,
            confidence_score=0.0,
            warnings=["source_dispatch_frame_schema_incompatible"],
        )
        self.save_feedback_frame(stub)

    def clear_feedback_pending(self, dispatch_frame_id: str, *, drain_batch: int = 5000) -> int:
        """Clear this row's marker, and up to `drain_batch` others that are already done.

        The marker defaults to TRUE, so every pre-migration row starts pending. Clearing ONE per
        poll would drain 423k rows at 1 per FEEDBACK_POLL_INTERVAL_SECs -- about 9.8 days during which the stage
        produces nothing, with no error and no log line to show it, because
        `ORDER BY generated_at ASC` puts every stale row ahead of genuinely new work. The batch
        is the same statement the migration's backfill uses and rides the partial index, so it
        turns that 9.8 days into a few minutes.

        Returns the number of markers cleared.
        """
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE substrate_execution_dispatch_frames
                       SET feedback_pending = false
                     WHERE frame_id = :frame_id
                """),
                {"frame_id": dispatch_frame_id},
            )
            result = conn.execute(
                text("""
                    UPDATE substrate_execution_dispatch_frames u
                       SET feedback_pending = false
                     WHERE u.frame_id IN (
                           SELECT x.frame_id
                             FROM substrate_execution_dispatch_frames x
                             JOIN substrate_feedback_frames y ON y.source_execution_dispatch_frame_id = x.frame_id
                            WHERE x.feedback_pending
                            LIMIT :batch
                     )
                """),
                {"batch": max(0, int(drain_batch))},
            )
        drained = 1 + int(result.rowcount or 0)
        if drained > 1:
            logger.info("feedback_pending_bulk_drained cleared=%s", drained)
        return drained

    def reconcile_feedback_pending(self, *, force: bool = False) -> int:
        """Re-queue any dispatch frame whose marker was cleared without a feedback frame.

        The marker is cleared transactionally, so this should find nothing -- but "should" is
        not a guarantee across manual SQL, restores, or a future bug, and the failure it guards
        against is silent work loss. It only ever sets the marker back to TRUE: it can add work,
        never remove it, so a bug here costs duplicated effort rather than lost effort.

        This is the expensive anti-join the marker exists to avoid, which is why it is
        rate-limited to once per `reconcile_interval_sec` (default 900s) instead of running on
        the 2s poll. Returns the number of rows re-queued.
        """
        now = time.monotonic()
        if not force and self._last_reconcile_mono is not None:
            if (now - self._last_reconcile_mono) < self._reconcile_interval_sec:
                return 0
        self._last_reconcile_mono = now
        with self._engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE substrate_execution_dispatch_frames d
                       SET feedback_pending = true
                     WHERE NOT d.feedback_pending
                       AND NOT EXISTS (
                             SELECT 1 FROM substrate_feedback_frames f
                              WHERE f.source_execution_dispatch_frame_id = d.frame_id
                       )
                """)
            )
        requeued = int(result.rowcount or 0)
        if requeued:
            logger.warning(
                "feedback_pending_reconciled requeued=%s -- dispatch frames had their pending "
                "marker cleared with no feedback frame present. Work would have been lost.",
                requeued,
            )
        return requeued

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

    def load_policy_frame(self, frame_id: str) -> PolicyDecisionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT policy_decision_frame_json
                        FROM substrate_policy_decision_frames
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
        payload = row["policy_decision_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return PolicyDecisionFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning(
                "policy_decision_frame_incompatible_schema frame_id=%s", frame_id, exc_info=True
            )
            return None

    def load_proposal_frame(self, frame_id: str) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json
                        FROM substrate_proposal_frames
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
            logger.warning("proposal_frame_incompatible_schema frame_id=%s", frame_id, exc_info=True)
            return None

    def load_field_for_tick(self, tick_id: str) -> FieldStateV1 | None:
        """2026-07-22 (SelfStateV1 burn): replaces load_self_state. Looks up
        the exact field tick a dispatch frame was built against
        (source_field_tick_id), mirroring
        orion-self-state-runtime/app/store.py's load_field_for_tick before
        that service is retired."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                        WHERE tick_id = :tick_id
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
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return FieldStateV1.model_validate(payload)
        except ValidationError:
            # build_feedback_frame already accepts field_before=None (an
            # intentionally optional input), so degrading here doesn't stall
            # this service's own FIFO queue the way policy/execution-
            # dispatch-runtime's fixed-id lookups could.
            logger.warning("field_state_incompatible_schema tick_id=%s", tick_id, exc_info=True)
            return None

    def load_action_scoring_window(
        self,
        dispatch_generated_at: datetime,
        *,
        settle_sec: float,
    ) -> tuple[FieldStateV1 | None, FieldStateV1 | None]:
        """The field window that actually CONTAINS the action.

        The feedback frame's own `field_before`/`field_after` do not, and the
        action ledger was scoring against them. Measured live 2026-08-22 over
        2,564 frames: `field_before` is the tick the POLICY frame was built
        against -- p50 **204.7 seconds** before the dispatch -- and
        `field_after` is the FIRST tick after it, p50 **1.12 seconds** later.

        So the window was [t-205s, t+1.1s]. The action starts ~0.33s after t
        and takes 1.2-5.4s, so the "after" snapshot was taken while the action
        was still running, usually before it had returned, and always before
        the digester could observe any consequence and fold it into a
        pressure. Roughly 70 other dispatches sat inside the same window.

        The estimator was therefore unbiased for a quantity that is null by
        construction -- "the effect of this action on the field one second
        before it finished". It would converge on ~0 with shrinking error bars
        no matter how well the action worked, which is precisely the
        confident-plausible-wrong failure orion/autonomy/contrast.py exists to
        prevent, and it invalidated the interpretation of every contrast
        measured before this fix.

        This window is:
          before = the newest tick at or before the dispatch
          after  = the first tick at least `settle_sec` after the dispatch

        `settle_sec` must cover send offset + action latency + the digester's
        own fold. It is no longer a constant: worker.py::_scoring_settle_sec
        adds the frame's worst MEASURED latency to `action_settle_sec`, because
        a fixed 15s was sized against the 1.2-5.4s action population above and
        could not follow a ~50s `express` run -- whose "after" sample therefore
        landed 35s before the action finished, reproducing this exact defect
        (three live outcomes 2026-08-31 with baseline == observed_after to 4dp).

        The original note here said "waiting costs nothing: the feedback runtime
        processes a dispatch frame minutes after it is written, so the later
        tick already exists". Measured over 10,261 frames (6h, 2026-08-31) that
        is true at p50 (94.5s) and p95 (172.5s) but FALSE at the minimum
        (0.1s) -- some frames are scored almost immediately, and for those this
        returns (None, None) while the caller still clears `feedback_pending`,
        losing the measurement permanently. worker.py::_tick now DEFERS a frame
        whose window has not closed instead of consuming it, which is what
        makes the claim true rather than merely usually-true.

        Returns (None, None) rather than a partial window -- a half-window is
        not a weaker measurement, it is a different one.
        """
        with self._engine.connect() as conn:
            before = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                         WHERE generated_at <= :at
                         ORDER BY generated_at DESC
                         LIMIT 1
                        """
                    ),
                    {"at": dispatch_generated_at},
                )
                .mappings()
                .first()
            )
            after = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                         WHERE generated_at >= :at + make_interval(secs => :settle)
                         ORDER BY generated_at ASC
                         LIMIT 1
                        """
                    ),
                    {"at": dispatch_generated_at, "settle": float(settle_sec)},
                )
                .mappings()
                .first()
            )
        if not before or not after:
            return None, None
        return _field_from_json(before["field_json"]), _field_from_json(after["field_json"])

    def load_latest_field_after(
        self,
        generated_at: datetime,
        *,
        window_sec: int = 30,
    ) -> FieldStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                        WHERE generated_at > :generated_at
                          AND generated_at <= :generated_at + make_interval(secs => :window_sec)
                        ORDER BY generated_at ASC
                        LIMIT 1
                        """
                    ),
                    {"generated_at": generated_at, "window_sec": float(window_sec)},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return FieldStateV1.model_validate(payload)
        except ValidationError:
            logger.warning("field_state_after_incompatible_schema", exc_info=True)
            return None

    def load_latest_feedback_frame(self) -> FeedbackFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT feedback_frame_json
                        FROM substrate_feedback_frames
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
        payload = row["feedback_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return FeedbackFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning("feedback_frame_incompatible_schema latest_lookup", exc_info=True)
            return None

    def load_feedback_frame_for_dispatch(self, dispatch_frame_id: str) -> FeedbackFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT feedback_frame_json
                        FROM substrate_feedback_frames
                        WHERE source_execution_dispatch_frame_id = :dispatch_frame_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"dispatch_frame_id": dispatch_frame_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["feedback_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return FeedbackFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning(
                "feedback_frame_incompatible_schema dispatch_frame_id=%s",
                dispatch_frame_id,
                exc_info=True,
            )
            return None

    def load_cortex_result_evidence(
        self, dispatch_frame: ExecutionDispatchFrameV1
    ) -> list[dict[str, object]]:
        dispatch_ids = [
            c.dispatch_id
            for c in (
                list(dispatch_frame.candidates)
                + list(dispatch_frame.blocked_candidates)
                + list(dispatch_frame.dispatched_candidates)
            )
        ]
        if not dispatch_ids:
            return []

        stmt = text(
            """
            SELECT result_id, dispatch_id, status, result_json, latency_ms
            FROM substrate_dispatch_results
            WHERE dispatch_id IN :dispatch_ids
            ORDER BY created_at DESC
            """
        ).bindparams(bindparam("dispatch_ids", expanding=True))

        with self._engine.connect() as conn:
            rows = conn.execute(stmt, {"dispatch_ids": dispatch_ids}).mappings().all()
        if not rows:
            return []

        evidence: list[dict[str, object]] = []
        seen_dispatch_ids: set[str] = set()
        for row in rows:
            dispatch_id = row["dispatch_id"]
            if dispatch_id in seen_dispatch_ids:
                # Most-recent-first ordering means the first occurrence per
                # dispatch_id is the latest result; later duplicates are stale.
                continue
            try:
                payload = row["result_json"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                evidence_refs = list(payload.get("evidence_refs") or []) if isinstance(payload, dict) else []
                entry: dict[str, object] = {
                    "result_id": row["result_id"],
                    "dispatch_id": dispatch_id,
                    "status": row["status"],
                    "evidence_refs": evidence_refs,
                }
                # 2026-08-21: this dict used to be exactly the four keys above,
                # which made worker.py::_latencies() UNREACHABLE -- it scans
                # these entries for latency_ms/duration_ms/elapsed_ms and they
                # were filtered out one layer earlier. Combined with nothing
                # writing the value in the first place, ActionOutcomeRecordV1.
                # latency_ms was populated on 0 of 5,739 rows over 6 hours: a
                # schema field, a column and a reader, none of which could ever
                # carry anything. Absent stays absent -- never coerced to 0.0,
                # which would read as "this action was free".
                # .get(), not ["..."]: a mapping without the key must read as
                # "no cost recorded", not crash. A KeyError here would abort
                # the whole evidence load and take feedback scoring down with
                # it, for the sake of one absent optional measurement. (If the
                # COLUMN were missing the SELECT above would fail first, so
                # this is not hiding a schema problem.)
                latency = row.get("latency_ms")
                if latency is not None:
                    entry["latency_ms"] = float(latency)
                evidence.append(entry)
                seen_dispatch_ids.add(dispatch_id)
            except (TypeError, ValueError, json.JSONDecodeError):
                # Malformed result_json on one row shouldn't sink the whole
                # query -- skip this row and keep the rest, mirroring this
                # file's degrade-gracefully-on-bad-data convention elsewhere.
                logger.warning(
                    "dispatch_result_incompatible_payload dispatch_id=%s", dispatch_id, exc_info=True
                )
                continue
        return evidence

    def load_effect_posteriors(self) -> dict[TreatedCellKey, EffectPosterior]:
        """Current belief about what each action does to each signal.

        A full read of a tiny table (one row per real (kind, target, signal),
        tens of rows) rather than a newest-row-wins scan over the ledger.
        That distinction is not cosmetic here: re-deriving state by scanning
        a large frame table on every check is exactly the pattern that made
        the daily risk baseline 49.8% of this database's buffer traffic.
        """
        rows = []
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

    def load_control_posteriors(self) -> dict[ControlCellKey, ControlCell]:
        """The untreated arm: what each signal does on ticks where nothing ran.

        Same tiny-table, full-read shape as load_effect_posteriors -- one row
        per (signal, arm, baseline bin), so at most
        len(PredictableSignal) * arms * 10 rows.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT signal_id, arm, baseline_bin,
                               posterior_mean, posterior_variance, posterior_n,
                               moved_n, move_rate
                          FROM substrate_signal_control_cells
                        """
                    )
                )
                .mappings()
                .all()
            )
        return {
            (r["signal_id"], r["arm"], int(r["baseline_bin"])): ControlCell(
                posterior=EffectPosterior(
                    mean=float(r["posterior_mean"]),
                    variance=float(r["posterior_variance"]),
                    n=int(r["posterior_n"]),
                ),
                moved_n=int(r["moved_n"]),
                move_rate=float(r["move_rate"]),
            )
            for r in rows
        }

    def save_feedback_frame(
        self,
        frame: FeedbackFrameV1,
        outcome_records: list[ActionOutcomeRecordV1] | None = None,
        control_cells: dict[ControlCellKey, ControlCell] | None = None,
        control_frame_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_feedback_frames (
                        frame_id,
                        source_execution_dispatch_frame_id,
                        source_policy_frame_id,
                        source_proposal_frame_id,
                        source_field_tick_id,
                        generated_at,
                        policy_id,
                        feedback_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_execution_dispatch_frame_id,
                        :source_policy_frame_id,
                        :source_proposal_frame_id,
                        :source_field_tick_id,
                        :generated_at,
                        :policy_id,
                        :feedback_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_execution_dispatch_frame_id = EXCLUDED.source_execution_dispatch_frame_id,
                        source_policy_frame_id = EXCLUDED.source_policy_frame_id,
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        feedback_frame_json = EXCLUDED.feedback_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_execution_dispatch_frame_id": frame.source_execution_dispatch_frame_id,
                    "source_policy_frame_id": frame.source_policy_frame_id,
                    "source_proposal_frame_id": frame.source_proposal_frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.feedback_policy_id,
                    "feedback_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
            # SAME TRANSACTION as the insert above, deliberately. Clearing the marker in a
            # separate transaction would mean a crash between the two either loses the work
            # (marker cleared, no frame) or reprocesses it. Inside one transaction, neither can
            # happen. Reprocessing would be harmless anyway -- the insert is ON CONFLICT DO
            # UPDATE -- but losing work would not, so this is the direction to be strict in.
            conn.execute(
                text("""
                    UPDATE substrate_execution_dispatch_frames
                       SET feedback_pending = false
                     WHERE frame_id = :frame_id
                """),
                {"frame_id": frame.source_execution_dispatch_frame_id},
            )

            # SAME TRANSACTION again, for the same reason and one more: the
            # marker clear above is what makes this dispatch frame stop being
            # work. If the ledger write landed in its own transaction and the
            # process died between the two, the observation would be gone
            # permanently -- nothing ever revisits a frame whose marker is
            # already clear.
            # SAVEPOINT, not a bare call. Finding 4 (review, 2026-08-21):
            # this shares a transaction with the frame insert and the
            # `feedback_pending = false` clear ABOVE, so an exception here
            # used to roll back all three. The next tick's
            # load_latest_dispatch_frame_without_feedback() is a FIFO on the
            # oldest unfed frame, so it would re-select the identical row and
            # fail identically -- a permanent head-of-line stall, which this
            # service has already suffered once (see that method's own
            # comment). A nested transaction lets the ledger write fail alone:
            # the frame and the marker still commit, the pipeline keeps
            # moving, and the lost observation is one row rather than the
            # whole loop.
            try:
                # TWO savepoints, not one (review finding 12). The arms are
                # independent data from different populations; a constraint
                # violation on one scored action must not also discard that
                # tick's untreated observations, which is what a shared
                # savepoint did.
                with conn.begin_nested():
                    self._write_action_outcomes(conn, outcome_records or [])
            except Exception:
                logger.exception(
                    "action_outcome_ledger_write_failed frame_id=%s records=%d",
                    frame.frame_id,
                    len(outcome_records or []),
                )
            try:
                with conn.begin_nested():
                    self._write_control_cells(
                        conn, control_cells or {}, dispatch_frame_id=control_frame_id
                    )
            except Exception:
                logger.exception(
                    "action_control_cell_write_failed frame_id=%s cells=%d",
                    frame.frame_id,
                    len(control_cells or {}),
                )

    @staticmethod
    def _write_action_outcomes(conn, records: list[ActionOutcomeRecordV1]) -> None:
        """Append scored outcomes and advance the posteriors they produced.

        Two guards against double-counting, because an observation absorbed
        twice corrupts the belief permanently and silently:

        1. The ledger insert is ON CONFLICT DO NOTHING on
           (dispatch_id, signal_id) and RETURNS the rows that actually
           landed. A reprocessed frame inserts nothing.
        2. The posterior is advanced ONLY for records whose ledger row was
           genuinely new, and only when it moves `posterior_n` forward. An
           out-of-order or replayed write cannot walk the belief backwards.
        """
        if not records:
            return
        for record in records:
            inserted = conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_outcomes (
                        dispatch_id, dispatch_frame_id, feedback_frame_id,
                        dispatch_kind, target_id, signal_id, direction,
                        arm, baseline_bin, frame_dispatch_count,
                        observed_at, baseline, observed_after, observed_delta,
                        predicted_delta, prediction_error, surprise_nats,
                        posterior_mean, posterior_variance, posterior_n,
                        co_predictors, latency_ms, claim_upheld
                    ) VALUES (
                        :dispatch_id, :dispatch_frame_id, :feedback_frame_id,
                        :dispatch_kind, :target_id, :signal_id, :direction,
                        :arm, :baseline_bin, :frame_dispatch_count,
                        :observed_at, :baseline, :observed_after, :observed_delta,
                        :predicted_delta, :prediction_error, :surprise_nats,
                        :posterior_mean, :posterior_variance, :posterior_n,
                        :co_predictors, :latency_ms, :claim_upheld
                    )
                    ON CONFLICT (dispatch_id, signal_id, dispatch_frame_id) DO NOTHING
                    RETURNING id
                    """
                ),
                # Explicit param map, not model_dump(): the model carries a
                # `schema_version` field with no matching bind parameter, and
                # relying on the driver to quietly drop unknown keys is the
                # kind of implicit behaviour that breaks on a library bump.
                {
                    "dispatch_id": record.dispatch_id,
                    "dispatch_frame_id": record.dispatch_frame_id,
                    "feedback_frame_id": record.feedback_frame_id,
                    "dispatch_kind": record.dispatch_kind,
                    "target_id": record.target_id,
                    "signal_id": record.signal_id,
                    "direction": record.direction,
                    "arm": record.arm,
                    "baseline_bin": record.baseline_bin,
                    "frame_dispatch_count": record.frame_dispatch_count,
                    "observed_at": record.observed_at,
                    "baseline": record.baseline,
                    "observed_after": record.observed_after,
                    "observed_delta": record.observed_delta,
                    "predicted_delta": record.predicted_delta,
                    "prediction_error": record.prediction_error,
                    "surprise_nats": record.surprise_nats,
                    "posterior_mean": record.posterior_mean,
                    "posterior_variance": record.posterior_variance,
                    "posterior_n": record.posterior_n,
                    "co_predictors": record.co_predictors,
                    "latency_ms": record.latency_ms,
                    "claim_upheld": record.claim_upheld,
                },
            ).first()
            if inserted is None:
                continue
            if record.arm != "dispatched":
                # A capacity-blocked candidate never ran. Its row is kept --
                # "the action that lost the race saw the signal do X" is real
                # evidence -- but folding it into the ACTION's own belief
                # would record the weather as the action's effect, which is
                # the exact defect the control arm exists to expose. The
                # posterior_n guard below would mask this as a silent no-op
                # rather than a decision, so it is stated here instead.
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_effect_posterior (
                        dispatch_kind, target_id, signal_id, baseline_bin,
                        posterior_mean, posterior_variance, posterior_n, updated_at
                    ) VALUES (
                        :dispatch_kind, :target_id, :signal_id, :baseline_bin,
                        :posterior_mean, :posterior_variance, :posterior_n, now()
                    )
                    ON CONFLICT (dispatch_kind, target_id, signal_id, baseline_bin)
                    DO UPDATE SET
                        posterior_mean = EXCLUDED.posterior_mean,
                        posterior_variance = EXCLUDED.posterior_variance,
                        posterior_n = EXCLUDED.posterior_n,
                        updated_at = now()
                     WHERE substrate_action_effect_posterior.posterior_n
                           < EXCLUDED.posterior_n
                    """
                ),
                {
                    "dispatch_kind": record.dispatch_kind,
                    "target_id": record.target_id,
                    "signal_id": record.signal_id,
                    "baseline_bin": record.baseline_bin,
                    "posterior_mean": record.posterior_mean,
                    "posterior_variance": record.posterior_variance,
                    "posterior_n": record.posterior_n,
                },
            )

    @staticmethod
    def _write_control_cells(
        conn,
        cells: dict[ControlCellKey, ControlCell],
        *,
        dispatch_frame_id: str | None = None,
    ) -> None:
        """Advance the untreated arm, once per dispatch frame.

        TWO guards, because the first one alone is not what it was documented
        to be (review finding 2). The monotone `posterior_n <` comparison
        stops the belief moving BACKWARDS; it does nothing against
        double-counting, since a replayed tick reads n=N+k, recomputes
        n=N+2k, and lands again quite happily. `last_dispatch_frame_id` is
        the actual dedup: the same dispatch frame folded twice is refused.

        Not triggerable today -- nothing prunes `substrate_feedback_frames`,
        so `reconcile_feedback_pending` never finds an aged frame to re-queue
        -- but that reconciler is a live, deliberate replay mechanism, and
        adding retention to that table would otherwise recount the entire
        backlog into ONE arm of the contrast and not the other. A replay that
        corrupts only the control side is worse than one that corrupts both.

        `dispatch_frame_id=None` disables the dedup rather than silently
        matching everything: a caller with no frame identity should still be
        able to advance a cell.

        That required an explicit NULL branch, which the first version did not
        have and got exactly backwards. `IS DISTINCT FROM` is **FALSE** when
        both sides are NULL (`SELECT NULL::text IS DISTINCT FROM NULL::text`
        -> `f`), so with the `control_frame_id=None` default a cell took its
        first INSERT and then had every subsequent update refused forever --
        the control arm silently frozen at one observation, with no error. The
        live worker always passes `dispatch.frame_id`, so this never fired in
        production, but the parameter default was a trap and the reasoning
        recorded beside it was wrong.
        """
        for (signal_id, arm, bin_index), cell in cells.items():
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_signal_control_cells (
                        signal_id, arm, baseline_bin,
                        posterior_mean, posterior_variance, posterior_n,
                        moved_n, move_rate, last_dispatch_frame_id, updated_at
                    ) VALUES (
                        :signal_id, :arm, :baseline_bin,
                        :posterior_mean, :posterior_variance, :posterior_n,
                        :moved_n, :move_rate, :dispatch_frame_id, now()
                    )
                    ON CONFLICT (signal_id, arm, baseline_bin) DO UPDATE SET
                        posterior_mean = EXCLUDED.posterior_mean,
                        posterior_variance = EXCLUDED.posterior_variance,
                        posterior_n = EXCLUDED.posterior_n,
                        moved_n = EXCLUDED.moved_n,
                        move_rate = EXCLUDED.move_rate,
                        last_dispatch_frame_id = EXCLUDED.last_dispatch_frame_id,
                        updated_at = now()
                     WHERE substrate_signal_control_cells.posterior_n
                           < EXCLUDED.posterior_n
                       AND (
                             EXCLUDED.last_dispatch_frame_id IS NULL
                             OR substrate_signal_control_cells.last_dispatch_frame_id
                                IS DISTINCT FROM EXCLUDED.last_dispatch_frame_id
                           )
                    """
                ),
                {
                    "signal_id": signal_id,
                    "arm": arm,
                    "baseline_bin": bin_index,
                    "posterior_mean": cell.posterior.mean,
                    "posterior_variance": cell.posterior.variance,
                    "posterior_n": cell.posterior.n,
                    "moved_n": cell.moved_n,
                    "move_rate": cell.move_rate,
                    "dispatch_frame_id": dispatch_frame_id,
                },
            )
