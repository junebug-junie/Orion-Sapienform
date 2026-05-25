"""Read-only debug API for substrate self-state snapshots."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

from orion.schemas.self_state import SelfStateV1

router = APIRouter(prefix="/api/substrate/self-state", tags=["substrate-self-state"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _load_latest_self_state() -> SelfStateV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT self_state_json FROM substrate_self_state
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["self_state_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return SelfStateV1.model_validate(payload)


@router.get("/latest")
async def self_state_latest() -> dict[str, Any]:
    state = _load_latest_self_state()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return state.model_dump(mode="json")
