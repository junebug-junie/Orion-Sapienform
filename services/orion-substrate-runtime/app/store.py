from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ReductionReceiptV1

from orion.substrate.biometrics_loop.constants import GRAMMAR_CURSOR_NAME
from orion.substrate.biometrics_loop.lineage import emission_touches_node, receipt_touches_node


class BiometricsSubstrateStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def fetch_biometrics_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT last_event_created_at, last_event_id
                    FROM substrate_reduction_cursor
                    WHERE cursor_name = :cursor_name
                    """
                ),
                {"cursor_name": GRAMMAR_CURSOR_NAME},
            ).mappings().first()

            params: dict[str, Any] = {"limit": limit}
            if row and row["last_event_created_at"]:
                params["cursor_ts"] = row["last_event_created_at"]
                params["cursor_id"] = row["last_event_id"] or ""
                query = """
                    SELECT event_id, event_json, created_at
                    FROM grammar_events
                    WHERE source_service = 'orion-biometrics'
                      AND trace_id LIKE 'biometrics.node:%'
                      AND (
                        created_at > :cursor_ts
                        OR (created_at = :cursor_ts AND event_id > :cursor_id)
                      )
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT :limit
                """
            else:
                query = """
                    SELECT event_id, event_json, created_at
                    FROM grammar_events
                    WHERE source_service = 'orion-biometrics'
                      AND trace_id LIKE 'biometrics.node:%'
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT :limit
                """

            rows = conn.execute(text(query), params).mappings().all()

        events: list[GrammarEventV1] = []
        for r in rows:
            payload = r["event_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            events.append(GrammarEventV1.model_validate(payload))
        return events

    def advance_cursor(self, *, event_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_cursor (
                        cursor_name, last_event_created_at, last_event_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :event_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_event_created_at = EXCLUDED.last_event_created_at,
                        last_event_id = EXCLUDED.last_event_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": GRAMMAR_CURSOR_NAME,
                    "created_at": created_at,
                    "event_id": event_id,
                    "updated_at": now,
                },
            )

    def _load_projection(self, table: str, projection_id: str, model: type) -> Any:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT projection_json FROM {table}
                    WHERE projection_id = :projection_id
                    """
                ),
                {"projection_id": projection_id},
            ).mappings().first()
        if not row:
            return None
        payload = row["projection_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return model.model_validate(payload)

    def _save_projection(self, table: str, projection: Any) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {table} (projection_id, generated_at, projection_json, created_at)
                    VALUES (:projection_id, :generated_at, :projection_json, :created_at)
                    ON CONFLICT (projection_id) DO UPDATE SET
                        generated_at = EXCLUDED.generated_at,
                        projection_json = EXCLUDED.projection_json
                    """
                ),
                {
                    "projection_id": projection.projection_id,
                    "generated_at": projection.generated_at,
                    "projection_json": Json(projection.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def load_node_biometrics(self, projection_id: str) -> NodeBiometricsProjectionV1 | None:
        return self._load_projection(
            "substrate_node_biometrics_projection",
            projection_id,
            NodeBiometricsProjectionV1,
        )

    def save_node_biometrics(self, projection: NodeBiometricsProjectionV1) -> None:
        self._save_projection("substrate_node_biometrics_projection", projection)

    def load_active_pressure(self, projection_id: str) -> ActiveNodePressureProjectionV1 | None:
        return self._load_projection(
            "substrate_active_node_pressure_projection",
            projection_id,
            ActiveNodePressureProjectionV1,
        )

    def save_active_pressure(self, projection: ActiveNodePressureProjectionV1) -> None:
        self._save_projection("substrate_active_node_pressure_projection", projection)

    def save_receipt(self, receipt: ReductionReceiptV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_receipts (
                        receipt_id, organ_id, emission_id, receipt_json, created_at
                    ) VALUES (
                        :receipt_id, :organ_id, :emission_id, :receipt_json, :created_at
                    )
                    ON CONFLICT (receipt_id) DO NOTHING
                    """
                ),
                {
                    "receipt_id": receipt.receipt_id,
                    "organ_id": receipt.organ_id,
                    "emission_id": receipt.emission_id,
                    "receipt_json": Json(receipt.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def save_emission(self, emission: OrganEmissionV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_organ_emissions (
                        emission_id, organ_id, invocation_id, emission_json, created_at
                    ) VALUES (
                        :emission_id, :organ_id, :invocation_id, :emission_json, :created_at
                    )
                    ON CONFLICT (emission_id) DO NOTHING
                    """
                ),
                {
                    "emission_id": emission.emission_id,
                    "organ_id": emission.organ_id,
                    "invocation_id": emission.invocation_id,
                    "emission_json": Json(emission.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def latest_emission_for_node(self, node_id: str) -> OrganEmissionV1 | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT emission_json FROM substrate_organ_emissions
                    WHERE organ_id = 'biometrics_pressure'
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                ),
            ).mappings().all()
        for row in rows:
            payload = row["emission_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            emission = OrganEmissionV1.model_validate(payload)
            if emission_touches_node(emission, node_id):
                return emission
        return None

    def latest_receipt_for_node(self, node_id: str) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                ),
            ).mappings().all()
        for row in rows:
            payload = row["receipt_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            receipt = ReductionReceiptV1.model_validate(payload)
            if receipt_touches_node(receipt, node_id):
                return receipt
        return None

    def load_receipt(self, receipt_id: str) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    WHERE receipt_id = :receipt_id
                    """
                ),
                {"receipt_id": receipt_id},
            ).mappings().first()
        if not row:
            return None
        payload = row["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ReductionReceiptV1.model_validate(payload)

    def grammar_event_created_at(self, event_id: str) -> datetime | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT created_at FROM grammar_events WHERE event_id = :event_id"),
                {"event_id": event_id},
            ).mappings().first()
        if not row:
            return None
        created_at = row["created_at"]
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=timezone.utc)
        return created_at

    def latest_receipt(self) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
            ).mappings().first()
        if not row:
            return None
        payload = row["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ReductionReceiptV1.model_validate(payload)
