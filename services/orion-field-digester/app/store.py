from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_state import FieldStateV1
from orion.schemas.reduction_receipt import ReductionReceiptV1

from app.ingest.receipts import parse_receipt_json

FIELD_DIGEST_CURSOR_NAME = "field_digest_receipt_consumer"


@dataclass(frozen=True)
class FetchedReceipt:
    receipt: ReductionReceiptV1
    created_at: datetime


@dataclass(frozen=True)
class PendingDelta:
    delta_id: str
    receipt_id: str


class FieldDigesterStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def fetch_new_receipts(self, *, limit: int = 50) -> list[FetchedReceipt]:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT last_receipt_created_at, last_receipt_id
                        FROM substrate_field_digest_cursor
                        WHERE cursor_name = :cursor_name
                        """
                    ),
                    {"cursor_name": FIELD_DIGEST_CURSOR_NAME},
                )
                .mappings()
                .first()
            )

            params: dict[str, Any] = {"limit": limit}
            if row and row["last_receipt_created_at"]:
                params["cursor_ts"] = row["last_receipt_created_at"]
                params["cursor_id"] = row["last_receipt_id"] or ""
                query = """
                    SELECT receipt_id, receipt_json, created_at
                    FROM substrate_reduction_receipts
                    WHERE (
                        created_at > :cursor_ts
                        OR (created_at = :cursor_ts AND receipt_id > :cursor_id)
                    )
                    ORDER BY created_at ASC, receipt_id ASC
                    LIMIT :limit
                """
            else:
                query = """
                    SELECT receipt_id, receipt_json, created_at
                    FROM substrate_reduction_receipts
                    ORDER BY created_at ASC, receipt_id ASC
                    LIMIT :limit
                """

            rows = conn.execute(text(query), params).mappings().all()

        fetched: list[FetchedReceipt] = []
        for row in rows:
            created_at = row["created_at"]
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            fetched.append(
                FetchedReceipt(
                    receipt=parse_receipt_json(row["receipt_json"]),
                    created_at=created_at,
                )
            )
        return fetched

    def is_delta_applied(self, delta_id: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT 1 FROM substrate_field_applied_deltas
                    WHERE delta_id = :delta_id
                    """
                ),
                {"delta_id": delta_id},
            ).first()
        return row is not None

    def mark_delta_applied(self, delta_id: str, receipt_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_field_applied_deltas (
                        delta_id, receipt_id, applied_at
                    ) VALUES (
                        :delta_id, :receipt_id, :applied_at
                    )
                    ON CONFLICT (delta_id) DO NOTHING
                    """
                ),
                {
                    "delta_id": delta_id,
                    "receipt_id": receipt_id,
                    "applied_at": now,
                },
            )

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

    def save_field(self, state: FieldStateV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_field_state (
                        tick_id, generated_at, field_json, created_at
                    ) VALUES (
                        :tick_id, :generated_at, :field_json, :created_at
                    )
                    ON CONFLICT (tick_id) DO UPDATE SET
                        generated_at = EXCLUDED.generated_at,
                        field_json = EXCLUDED.field_json
                    """
                ),
                {
                    "tick_id": state.tick_id,
                    "generated_at": state.generated_at,
                    "field_json": state.model_dump(mode="json"),
                    "created_at": now,
                },
            )

    def advance_cursor(self, receipt_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_field_digest_cursor (
                        cursor_name, last_receipt_created_at, last_receipt_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :receipt_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_receipt_created_at = EXCLUDED.last_receipt_created_at,
                        last_receipt_id = EXCLUDED.last_receipt_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": FIELD_DIGEST_CURSOR_NAME,
                    "created_at": created_at,
                    "receipt_id": receipt_id,
                    "updated_at": now,
                },
            )

    def commit_digest_tick(
        self,
        *,
        state: FieldStateV1,
        pending_deltas: list[PendingDelta],
        cursor_receipt_id: str,
        cursor_created_at: datetime,
    ) -> None:
        """Atomically persist field state, applied deltas, and receipt cursor."""
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_field_state (
                        tick_id, generated_at, field_json, created_at
                    ) VALUES (
                        :tick_id, :generated_at, :field_json, :created_at
                    )
                    ON CONFLICT (tick_id) DO UPDATE SET
                        generated_at = EXCLUDED.generated_at,
                        field_json = EXCLUDED.field_json
                    """
                ),
                {
                    "tick_id": state.tick_id,
                    "generated_at": state.generated_at,
                    "field_json": state.model_dump(mode="json"),
                    "created_at": now,
                },
            )
            for pending in pending_deltas:
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_field_applied_deltas (
                            delta_id, receipt_id, applied_at
                        ) VALUES (
                            :delta_id, :receipt_id, :applied_at
                        )
                        ON CONFLICT (delta_id) DO NOTHING
                        """
                    ),
                    {
                        "delta_id": pending.delta_id,
                        "receipt_id": pending.receipt_id,
                        "applied_at": now,
                    },
                )
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_field_digest_cursor (
                        cursor_name, last_receipt_created_at, last_receipt_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :receipt_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_receipt_created_at = EXCLUDED.last_receipt_created_at,
                        last_receipt_id = EXCLUDED.last_receipt_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": FIELD_DIGEST_CURSOR_NAME,
                    "created_at": cursor_created_at,
                    "receipt_id": cursor_receipt_id,
                    "updated_at": now,
                },
            )
