from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1

logger = logging.getLogger("orion.policy_runtime.store")


class PolicyRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_next_proposal_without_policy_frame(self) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT p.proposal_frame_json
                        FROM substrate_proposal_frames p
                        LEFT JOIN substrate_policy_decision_frames d
                          ON d.source_proposal_frame_id = p.frame_id
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
        payload = row["proposal_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ProposalFrameV1.model_validate(payload)
        except ValidationError:
            # This is the FIFO "oldest proposal without a policy frame"
            # lookup -- a naive None-degrade would re-select this exact row
            # forever (it can never validate), permanently blocking every
            # proposal queued behind it. A schema migration (e.g. 2026-07-22's
            # SelfStateV1 burn) can leave historical rows like this
            # incompatible with the currently-running ProposalFrameV1.
            # Retire it with a stub "unevaluable" decision frame so the FIFO
            # advances past it, mirroring the pre-burn
            # build_unevaluable_policy_decision_frame pattern.
            raw_frame_id = payload.get("frame_id") if isinstance(payload, dict) else None
            logger.warning(
                "proposal_frame_incompatible_schema fifo_lookup frame_id=%s",
                raw_frame_id,
                exc_info=True,
            )
            if raw_frame_id:
                self._retire_incompatible_proposal_frame(raw_frame_id)
            return None

    def _retire_incompatible_proposal_frame(self, raw_frame_id: str) -> None:
        """Insert a stub 'unevaluable' policy_decision_frame for a proposal
        frame that failed schema validation, so
        load_next_proposal_without_policy_frame's FIFO lookup doesn't
        re-select this exact row forever."""
        stub = PolicyDecisionFrameV1(
            frame_id=f"policy.frame:{raw_frame_id}:schema_incompatible",
            generated_at=datetime.now(timezone.utc),
            source_proposal_frame_id=raw_frame_id,
            decisions=[],
            overall_risk=0.0,
            operator_review_required=True,
            execution_allowed=False,
            warnings=["source_proposal_frame_schema_incompatible"],
        )
        self.save_policy_decision_frame(stub)

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
        try:
            return ProposalFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning("proposal_frame_incompatible_schema latest_lookup", exc_info=True)
            return None

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
        try:
            return PolicyDecisionFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning(
                "policy_decision_frame_incompatible_schema proposal_frame_id=%s",
                proposal_frame_id,
                exc_info=True,
            )
            return None

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
        try:
            return PolicyDecisionFrameV1.model_validate(payload)
        except ValidationError:
            logger.warning("policy_decision_frame_incompatible_schema latest_lookup", exc_info=True)
            return None

    def save_policy_decision_frame(self, frame: PolicyDecisionFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_policy_decision_frames (
                        frame_id,
                        source_proposal_frame_id,
                        source_field_tick_id,
                        generated_at,
                        policy_id,
                        policy_decision_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_proposal_frame_id,
                        :source_field_tick_id,
                        :generated_at,
                        :policy_id,
                        :policy_decision_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_proposal_frame_id = EXCLUDED.source_proposal_frame_id,
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        policy_decision_frame_json = EXCLUDED.policy_decision_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_proposal_frame_id": frame.source_proposal_frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.policy_id,
                    "policy_decision_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
