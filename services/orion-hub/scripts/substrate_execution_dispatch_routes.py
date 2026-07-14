"""Read-only debug API for substrate execution dispatch frames."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
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
    try:
        return ExecutionDispatchFrameV1.model_validate(payload)
    except ValidationError:
        return None


def _dispatch_status_summary(frame: ExecutionDispatchFrameV1) -> dict[str, int]:
    """Operator-visible breakdown of what actually happened in this frame --
    P3's "dispatched vs dry_run/prepared_for_dispatch counts" ask. Derived
    entirely from fields the frame already carries; no new schema fields.
    theater_tripwire_active is deliberately NOT included here -- that flag
    is in-process state on orion-execution-dispatch-runtime's own worker,
    not persisted anywhere this Postgres-only route can reach without a
    cross-service call (named as a non-goal in the P3 design spec).
    """
    dry_run_count = sum(1 for c in frame.candidates if c.dispatch_status == "dry_run")
    prepared_for_dispatch_count = sum(
        1 for c in frame.candidates if c.dispatch_status == "prepared_for_dispatch"
    )
    return {
        "dispatched_count": len(frame.dispatched_candidates),
        "prepared_for_dispatch_count": prepared_for_dispatch_count,
        "dry_run_count": dry_run_count,
    }


@router.get("/latest")
async def execution_dispatch_latest() -> dict[str, Any]:
    frame = _load_latest_dispatch_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    payload = frame.model_dump(mode="json")
    payload["status_summary"] = _dispatch_status_summary(frame)
    return payload
