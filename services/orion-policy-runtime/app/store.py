from __future__ import annotations

import json
from datetime import datetime, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


class PolicyRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_proposal_frame(self) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json FROM substrate_proposal_frames
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
        return SelfStateV1.model_validate(payload)

    def load_policy_frame_for_proposal(self, proposal_frame_id: str) -> PolicyDecisionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT policy_decision_frame_json
                        FROM substrate_policy_decision_frames
                        WHERE source_proposal_frame_id = :proposal_frame_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"proposal_frame_id": proposal_frame_id},
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

    def load_latest_policy_decision_frame(self) -> PolicyDecisionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT policy_decision_frame_json
                        FROM substrate_policy_decision_frames
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
        payload = row["policy_decision_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return PolicyDecisionFrameV1.model_validate(payload)

    def save_policy_decision_frame(self, frame: PolicyDecisionFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_policy_decision_frames (
                        frame_id,
                        source_proposal_frame_id,
                        source_self_state_id,
                        generated_at,
                        policy_id,
                        policy_decision_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_proposal_frame_id,
                        :source_self_state_id,
                        :generated_at,
                        :policy_id,
                        :policy_decision_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_self_state_id = EXCLUDED.source_self_state_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        policy_decision_frame_json = EXCLUDED.policy_decision_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_proposal_frame_id": frame.source_proposal_frame_id,
                    "source_self_state_id": frame.source_self_state_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.policy_id,
                    "policy_decision_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
