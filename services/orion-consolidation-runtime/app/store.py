from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, TypeVar

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import ConsolidationFrameV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1

T = TypeVar("T")


class ConsolidationRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_consolidation_frame_for_window(
        self, frame_id: str
    ) -> ConsolidationFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT consolidation_frame_json
                        FROM substrate_consolidation_frames
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
        return self._parse_json(row["consolidation_frame_json"], ConsolidationFrameV1)

    def load_latest_consolidation_frame(self) -> ConsolidationFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT consolidation_frame_json
                        FROM substrate_consolidation_frames
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
        return self._parse_json(row["consolidation_frame_json"], ConsolidationFrameV1)

    def load_window_data(
        self,
        window_start: datetime,
        window_end: datetime,
        max_per_source: int,
    ) -> ConsolidationWindowData:
        return ConsolidationWindowData(
            window_start=window_start,
            window_end=window_end,
            self_states=self._load_rows(
                """
                SELECT self_state_json
                FROM substrate_self_state
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                SelfStateV1,
                "self_state_json",
            ),
            attention_frames=self._load_rows(
                """
                SELECT frame_json
                FROM substrate_attention_frames
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                FieldAttentionFrameV1,
                "frame_json",
            ),
            proposal_frames=self._load_rows(
                """
                SELECT proposal_frame_json
                FROM substrate_proposal_frames
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                ProposalFrameV1,
                "proposal_frame_json",
            ),
            policy_frames=self._load_rows(
                """
                SELECT policy_decision_frame_json
                FROM substrate_policy_decision_frames
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                PolicyDecisionFrameV1,
                "policy_decision_frame_json",
            ),
            dispatch_frames=self._load_rows(
                """
                SELECT dispatch_frame_json
                FROM substrate_execution_dispatch_frames
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                ExecutionDispatchFrameV1,
                "dispatch_frame_json",
            ),
            feedback_frames=self._load_rows(
                """
                SELECT feedback_frame_json
                FROM substrate_feedback_frames
                WHERE generated_at >= :window_start
                  AND generated_at < :window_end
                ORDER BY generated_at DESC
                LIMIT :max_per_source
                """,
                window_start,
                window_end,
                max_per_source,
                FeedbackFrameV1,
                "feedback_frame_json",
            ),
        )

    def save_consolidation_frame(self, frame: ConsolidationFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_consolidation_frames (
                        frame_id,
                        window_start,
                        window_end,
                        generated_at,
                        policy_id,
                        consolidation_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :window_start,
                        :window_end,
                        :generated_at,
                        :policy_id,
                        :consolidation_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO NOTHING
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "window_start": frame.window_start,
                    "window_end": frame.window_end,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.consolidation_policy_id,
                    "consolidation_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def _load_rows(
        self,
        sql: str,
        window_start: datetime,
        window_end: datetime,
        max_per_source: int,
        model: Callable[[dict], T],
        json_column: str,
    ) -> list[T]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(sql),
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "max_per_source": max_per_source,
                },
            ).mappings()
            return [
                self._parse_json(row[json_column], model)
                for row in rows
                if row.get(json_column) is not None
            ]

    @staticmethod
    def _parse_json(payload: object, model: Callable[[dict], T]) -> T:
        if isinstance(payload, str):
            payload = json.loads(payload)
        return model.model_validate(payload)
