"""Read-only debug API for substrate execution dispatch frames."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1

router = APIRouter(
    prefix="/api/substrate/execution-dispatch",
    tags=["substrate-execution-dispatch"],
)


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _load_latest_dispatch_frame() -> ExecutionDispatchFrameV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT dispatch_frame_json
                FROM substrate_execution_dispatch_frames
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["dispatch_frame_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ExecutionDispatchFrameV1.model_validate(payload)


@router.get("/latest")
async def execution_dispatch_latest() -> dict[str, Any]:
    frame = _load_latest_dispatch_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
