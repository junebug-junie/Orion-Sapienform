from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.substrate.pressure import PressureConfig

logger = logging.getLogger("orion.execution_dispatch.runtime.store")

# Reuses the same horizon `orion/substrate/endogenous_curiosity.py`'s
# `_prediction_error_staleness_decay()` applies to this identical undecayed raw
# field, rather than inventing a fresh number -- see
# `latest_bus_synaptic_prediction_error()`'s docstring for why this method treats
# a stale row as absent (None) instead of partially decaying it.
_BUS_SYNAPTIC_STALENESS_HORIZON_SEC = PressureConfig().prediction_error_decay_horizon_seconds


class ExecutionDispatchRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_policy_frame_without_dispatch(self) -> PolicyDecisionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT p.policy_decision_frame_json
                        FROM substrate_policy_decision_frames p
                        LEFT JOIN substrate_execution_dispatch_frames d
                          ON d.source_policy_frame_id = p.frame_id
                        WHERE d.frame_id IS NULL
                        ORDER BY p.generated_at ASC
                        LIMIT 1
                        """
                    ),
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
            # This is the FIFO "oldest undispatched policy frame" lookup --
            # a naive None-degrade would re-select this exact row forever
            # (it can never validate), permanently blocking every policy
            # frame queued behind it. A schema migration (e.g. 2026-07-22's
            # SelfStateV1 burn) can leave historical rows like this
            # incompatible with the currently-running PolicyDecisionFrameV1.
            # Retire it with a stub, unattempted dispatch frame so the FIFO
            # advances past it.
            raw_frame_id = payload.get("frame_id") if isinstance(payload, dict) else None
            raw_proposal_frame_id = (
                payload.get("source_proposal_frame_id") if isinstance(payload, dict) else None
            )
            logger.warning(
                "policy_decision_frame_incompatible_schema fifo_lookup frame_id=%s",
                raw_frame_id,
                exc_info=True,
            )
            if raw_frame_id:
                self._retire_incompatible_policy_frame(raw_frame_id, raw_proposal_frame_id)
            return None

    def _retire_incompatible_policy_frame(
        self, raw_frame_id: str, raw_proposal_frame_id: str | None
    ) -> None:
        """Insert a stub, unattempted execution_dispatch_frame for a policy
        frame that failed schema validation, so
        load_latest_policy_frame_without_dispatch's FIFO lookup doesn't
        re-select this exact row forever."""
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

    def save_dispatch_result(
        self,
        *,
        result_id: str,
        dispatch_id: str,
        frame_id: str,
        status: str,
        result_json: dict,
        raw_len: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_dispatch_results (
                        result_id, dispatch_id, frame_id, status, result_json, raw_len, created_at
                    ) VALUES (
                        :result_id, :dispatch_id, :frame_id, :status, :result_json, :raw_len, :created_at
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

    def latest_bus_synaptic_prediction_error(self) -> float | None:
        """Real, live surprise signal for `ActionOutcomeEmitV1.surprise` -- replaces the
        hardcoded `0.0` this service previously emitted (a disclosed, honest placeholder
        per docs/superpowers/specs/2026-07-13-autonomy-experience-loop-p2-design.md, not a
        bug, but never a real signal either).

        Reads `orion/substrate/prediction_error.py::bus_synaptic_prediction_error()`'s
        already-written output off `substrate_field_state` (`node:substrate.bus_synaptic`'s
        `prediction_error` field) -- populated by `orion-field-digester`'s
        `state_deltas.py` (`target_kind == "prediction_signal"`, `mode="replace"`), not
        directly by `orion-substrate-runtime`, which only publishes the perturbation that
        triggers that write. Chosen over the other four domains (execution/biometrics/chat/
        route) because it is generic across the whole bus mesh rather than scoped to one
        reducer, and because it already went through a real live-data sanity check (PR #1391
        fixed a ~0.27 calm-floor bias found by recovering real numbers, not by eyeballing
        variance -- see that PR and
        docs/superpowers/specs/2026-07-26-transport-domain-retirement-bus-synaptic-successor-design.md).
        Edge-count cold-start filtering (`count < ~5` unreliable per
        `services/orion-bus-mirror/README.md`) already happens upstream, in
        `orion-substrate-runtime`'s `_bus_synaptic_tick`, before this value is ever written --
        not re-done here.

        **Staleness guard**: `prediction_error` was deliberately made an undecayed raw
        snapshot (removed from `NODE_DECAY_CHANNELS` 2026-07-26, per
        `services/orion-field-digester/app/digestion/decay.py`) -- it sits at whatever was
        last written until the next perturbation, and does not self-correct if the writing
        tick stalls or the producing service goes down. This module has already been burned
        twice by trusting an undecayed/mis-decayed field at face value (`node:substrate.route`
        decaying unopposed for 48h; the bus_synaptic calm-floor bug). `orion/substrate/
        endogenous_curiosity.py::_prediction_error_staleness_decay()` guards the same raw
        field for its own consumer by age-decaying it; this method instead treats a row
        older than `_BUS_SYNAPTIC_STALENESS_HORIZON_SEC` as unavailable (`None`) rather than
        partially decaying it -- a single scalar has no good "partially stale" representation,
        and honest absence is the same choice #1379's design doc already made for this field.

        Returns `None` (not `0.0`) when the node is absent, stale, or unparseable, so the
        caller can tell "no real signal available" apart from "genuinely calm" -- collapsing
        those would recreate exactly the degenerate-zero failure mode this patch exists to fix.
        """
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT field_json -> 'node_vectors' -> 'node:substrate.bus_synaptic'
                           ->> 'prediction_error' AS value,
                           generated_at
                    FROM substrate_field_state
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        if not row or row["value"] is None:
            return None
        generated_at = row["generated_at"]
        if generated_at is not None:
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - generated_at).total_seconds()
            if age_seconds > _BUS_SYNAPTIC_STALENESS_HORIZON_SEC:
                return None
        try:
            return float(row["value"])
        except (TypeError, ValueError):
            return None

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
