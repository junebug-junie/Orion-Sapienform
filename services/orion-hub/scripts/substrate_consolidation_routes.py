"""Read-only debug API for substrate consolidation frames."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from orion.consolidation.repository import (
    load_expectations_for_motif,
    load_latest_tensor_slices,
    load_recent_motifs,
    load_schema_candidates,
)
from orion.schemas.consolidation_frame import ConsolidationFrameV1

router = APIRouter(prefix="/api/substrate/consolidation", tags=["substrate-consolidation"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _postgres_uri() -> str:
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return uri


def _load_latest_consolidation_frame() -> ConsolidationFrameV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT consolidation_frame_json
                FROM substrate_consolidation_frames
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["consolidation_frame_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ConsolidationFrameV1.model_validate(payload)


@router.get("/latest")
async def consolidation_latest() -> dict[str, Any]:
    frame = _load_latest_consolidation_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")


@router.get("/motifs")
async def consolidation_motifs(
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    motifs = load_recent_motifs(_postgres_uri(), limit)
    return [motif.model_dump(mode="json") for motif in motifs]


@router.get("/expectations")
async def consolidation_expectations(
    trigger_motif_id: str = Query(..., min_length=1),
) -> list[dict[str, Any]]:
    expectations = load_expectations_for_motif(_postgres_uri(), trigger_motif_id)
    return [expectation.model_dump(mode="json") for expectation in expectations]


@router.get("/schema-candidates")
async def consolidation_schema_candidates(
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    candidates = load_schema_candidates(_postgres_uri(), limit)
    return [candidate.model_dump(mode="json") for candidate in candidates]


@router.get("/tensor-slices/latest")
async def consolidation_tensor_slices_latest(
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    slices = load_latest_tensor_slices(_postgres_uri(), limit)
    return [tensor_slice.model_dump(mode="json") for tensor_slice in slices]
