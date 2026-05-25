"""Read-only Postgres accessors for consolidation substrate artifacts."""

from __future__ import annotations

import json
from typing import Callable, TypeVar

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.consolidation_frame import (
    ConsolidationFrameV1,
    ExpectationV1,
    MotifObservationV1,
    SchemaCandidateV1,
    SparseTensorSliceV1,
)

T = TypeVar("T")


def _engine_for(postgres_uri: str, engine: Engine | None = None) -> Engine:
    if engine is not None:
        return engine
    return create_engine(
        postgres_uri,
        pool_pre_ping=True,
        json_serializer=json.dumps,
        json_deserializer=json.loads,
    )


def _parse_json(payload: object, model: Callable[[dict], T]) -> T:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return model.model_validate(payload)


def _load_latest_consolidation_frame(
    engine: Engine,
) -> ConsolidationFrameV1 | None:
    with engine.connect() as conn:
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
    return _parse_json(row["consolidation_frame_json"], ConsolidationFrameV1)


def load_recent_motifs(
    postgres_uri: str,
    limit: int,
    *,
    engine: Engine | None = None,
) -> list[MotifObservationV1]:
    frame = _load_latest_consolidation_frame(_engine_for(postgres_uri, engine))
    if frame is None:
        return []
    return list(frame.motif_observations[: max(0, limit)])


def load_expectations_for_motif(
    postgres_uri: str,
    trigger_motif_id: str,
    *,
    engine: Engine | None = None,
) -> list[ExpectationV1]:
    with _engine_for(postgres_uri, engine).connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT expectation_json
                FROM substrate_expectations
                WHERE trigger_motif_id = :trigger_motif_id
                ORDER BY updated_at DESC
                """
            ),
            {"trigger_motif_id": trigger_motif_id},
        ).mappings()
        return [
            _parse_json(row["expectation_json"], ExpectationV1)
            for row in rows
            if row.get("expectation_json") is not None
        ]


def load_schema_candidates(
    postgres_uri: str,
    limit: int,
    *,
    engine: Engine | None = None,
) -> list[SchemaCandidateV1]:
    with _engine_for(postgres_uri, engine).connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT schema_candidate_json
                FROM substrate_schema_candidates
                ORDER BY updated_at DESC
                LIMIT :limit
                """
            ),
            {"limit": max(0, limit)},
        ).mappings()
        return [
            _parse_json(row["schema_candidate_json"], SchemaCandidateV1)
            for row in rows
            if row.get("schema_candidate_json") is not None
        ]


def load_latest_tensor_slices(
    postgres_uri: str,
    limit: int,
    *,
    engine: Engine | None = None,
) -> list[SparseTensorSliceV1]:
    with _engine_for(postgres_uri, engine).connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT tensor_json
                FROM substrate_tensor_slices
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": max(0, limit)},
        ).mappings()
        return [
            _parse_json(row["tensor_json"], SparseTensorSliceV1)
            for row in rows
            if row.get("tensor_json") is not None
        ]
