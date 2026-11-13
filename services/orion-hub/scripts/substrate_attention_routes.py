"""Read-only debug API for substrate field attention frames."""

from __future__ import annotations

import asyncio

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from orion.schemas.field_attention_frame import FieldAttentionFrameV1

from .pg_engine import module_engine

router = APIRouter(prefix="/api/substrate/attention", tags=["substrate-attention"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return module_engine(__name__, uri)


def _load_latest_attention_frame() -> FieldAttentionFrameV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT frame_json FROM substrate_attention_frames
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["frame_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return FieldAttentionFrameV1.model_validate(payload)


@router.get("/latest")
async def attention_latest() -> dict[str, Any]:
    frame = await asyncio.to_thread(_load_latest_attention_frame)
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
