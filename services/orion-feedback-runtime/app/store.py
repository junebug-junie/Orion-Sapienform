from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from orion.autonomy.prediction import EffectPosterior
from orion.schemas.action_prediction import ActionOutcomeRecordV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1

logger = logging.getLogger("orion.feedback_runtime.store")


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
            SELECT result_id, dispatch_id, status, result_json
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
                evidence.append(
                    {
                        "result_id": row["result_id"],
                        "dispatch_id": dispatch_id,
                        "status": row["status"],
                        "evidence_refs": evidence_refs,
                    }
                )
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

    def load_effect_posteriors(self) -> dict[tuple[str, str, str], EffectPosterior]:
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
                        SELECT dispatch_kind, target_id, signal_id,
                               posterior_mean, posterior_variance, posterior_n
                          FROM substrate_action_effect_posterior
                        """
                    )
                )
                .mappings()
                .all()
            )
        return {
            (r["dispatch_kind"], r["target_id"], r["signal_id"]): EffectPosterior(
                mean=float(r["posterior_mean"]),
                variance=float(r["posterior_variance"]),
                n=int(r["posterior_n"]),
            )
            for r in rows
        }

    def save_feedback_frame(
        self,
        frame: FeedbackFrameV1,
        outcome_records: list[ActionOutcomeRecordV1] | None = None,
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
            self._write_action_outcomes(conn, outcome_records or [])

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
                        observed_at, baseline, observed_after, observed_delta,
                        predicted_delta, prediction_error, surprise_nats,
                        posterior_mean, posterior_variance, posterior_n,
                        co_predictors, latency_ms
                    ) VALUES (
                        :dispatch_id, :dispatch_frame_id, :feedback_frame_id,
                        :dispatch_kind, :target_id, :signal_id, :direction,
                        :observed_at, :baseline, :observed_after, :observed_delta,
                        :predicted_delta, :prediction_error, :surprise_nats,
                        :posterior_mean, :posterior_variance, :posterior_n,
                        :co_predictors, :latency_ms
                    )
                    ON CONFLICT (dispatch_id, signal_id) DO NOTHING
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
                },
            ).first()
            if inserted is None:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_effect_posterior (
                        dispatch_kind, target_id, signal_id,
                        posterior_mean, posterior_variance, posterior_n, updated_at
                    ) VALUES (
                        :dispatch_kind, :target_id, :signal_id,
                        :posterior_mean, :posterior_variance, :posterior_n, now()
                    )
                    ON CONFLICT (dispatch_kind, target_id, signal_id) DO UPDATE SET
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
                    "posterior_mean": record.posterior_mean,
                    "posterior_variance": record.posterior_variance,
                    "posterior_n": record.posterior_n,
                },
            )
