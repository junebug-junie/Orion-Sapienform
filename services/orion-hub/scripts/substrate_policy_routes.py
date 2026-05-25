"""Read-only debug API for substrate policy decision frames."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1

router = APIRouter(prefix="/api/substrate/policy", tags=["substrate-policy"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _load_latest_policy_frame() -> PolicyDecisionFrameV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT policy_decision_frame_json
                FROM substrate_policy_decision_frames
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["policy_decision_frame_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return PolicyDecisionFrameV1.model_validate(payload)


@router.get("/latest")
async def policy_latest() -> dict[str, Any]:
    frame = _load_latest_policy_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
