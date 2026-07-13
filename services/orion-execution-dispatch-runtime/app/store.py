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
from orion.schemas.self_state import SelfStateV1

logger = logging.getLogger("orion.execution_dispatch.runtime.store")


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
        return PolicyDecisionFrameV1.model_validate(payload)

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
            # See services/orion-policy-runtime/app/store.py's identical fix
            # for why: looked up by a fixed self_state_id, so a naive raise
            # would permanently block this (and every queued) frame.
            logger.warning(
                "self_state_incompatible_schema self_state_id=%s", self_state_id, exc_info=True
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
            # Same reasoning as load_self_state's guard above: looked up by a
            # fixed policy_frame_id, so a naive raise would permanently block
            # this caller on a schema-incompatible historical row.
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
                        source_self_state_id,
                        generated_at,
                        policy_id,
                        dispatch_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_policy_frame_id,
                        :source_proposal_frame_id,
                        :source_self_state_id,
                        :generated_at,
                        :policy_id,
                        :dispatch_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_policy_frame_id = EXCLUDED.source_policy_frame_id,
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_self_state_id = EXCLUDED.source_self_state_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        dispatch_frame_json = EXCLUDED.dispatch_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_policy_frame_id": frame.source_policy_frame_id,
                    "source_proposal_frame_id": frame.source_proposal_frame_id,
                    "source_self_state_id": frame.source_self_state_id,
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

    def count_dispatches_today(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT count(*) AS n
                    FROM substrate_dispatch_results
                    WHERE created_at >= date_trunc('day', now())
                    """
                ),
            ).mappings().first()
        return int(row["n"]) if row else 0

    def recent_dispatch_result_statuses(self, limit: int = 10) -> list[str]:
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
