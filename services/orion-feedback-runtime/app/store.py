from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1

logger = logging.getLogger("orion.feedback_runtime.store")


class FeedbackRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_dispatch_frame_without_feedback(self) -> ExecutionDispatchFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT d.dispatch_frame_json
                        FROM substrate_execution_dispatch_frames d
                        LEFT JOIN substrate_feedback_frames f
                          ON f.source_execution_dispatch_frame_id = d.frame_id
                        WHERE f.frame_id IS NULL
                        ORDER BY d.generated_at ASC
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
            # Mirrors load_self_state's guard below: this is the FIFO lookup
            # (oldest dispatch frame without feedback yet) -- a naive raise
            # would re-select and fail on this exact row every tick forever,
            # blocking every dispatch frame queued behind it. Degrading to
            # None here means this tick does no work for a bad row instead of
            # crash-looping on it; the row itself isn't retired (no feedback
            # frame is written for it), which is the same acknowledged
            # limitation self_state's None-degrade already accepts elsewhere
            # in this file.
            logger.warning("dispatch_frame_incompatible_schema fifo_lookup", exc_info=True)
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
        return PolicyDecisionFrameV1.model_validate(payload)

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
        return ProposalFrameV1.model_validate(payload)

    def load_self_state(self, self_state_id: str) -> SelfStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
                        WHERE self_state_id = :self_state_id
                        LIMIT 1
                        """
                    ),
                    {"self_state_id": self_state_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["self_state_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return SelfStateV1.model_validate(payload)
        except ValidationError:
            # build_feedback_frame already accepts self_state_before=None
            # (an intentionally optional input), so degrading here doesn't
            # stall this service's own FIFO queue the way policy/execution-
            # dispatch-runtime's fixed-id lookups could.
            logger.warning(
                "self_state_incompatible_schema self_state_id=%s", self_state_id, exc_info=True
            )
            return None

    def load_latest_self_state_after(
        self,
        generated_at: datetime,
        *,
        window_sec: int = 30,
    ) -> SelfStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
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
        payload = row["self_state_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return SelfStateV1.model_validate(payload)
        except ValidationError:
            logger.warning("self_state_after_incompatible_schema", exc_info=True)
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
        return FeedbackFrameV1.model_validate(payload)

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
        return FeedbackFrameV1.model_validate(payload)

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

    def save_feedback_frame(self, frame: FeedbackFrameV1) -> None:
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
                        source_self_state_id,
                        generated_at,
                        policy_id,
                        feedback_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_execution_dispatch_frame_id,
                        :source_policy_frame_id,
                        :source_proposal_frame_id,
                        :source_self_state_id,
                        :generated_at,
                        :policy_id,
                        :feedback_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_execution_dispatch_frame_id = EXCLUDED.source_execution_dispatch_frame_id,
                        source_policy_frame_id = EXCLUDED.source_policy_frame_id,
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_self_state_id = EXCLUDED.source_self_state_id,
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
                    "source_self_state_id": frame.source_self_state_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.feedback_policy_id,
                    "feedback_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
