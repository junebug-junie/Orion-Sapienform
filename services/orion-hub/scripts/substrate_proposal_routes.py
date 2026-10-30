"""Read-only debug API for substrate proposal frames."""

from __future__ import annotations

import asyncio

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from sqlalchemy import text

from orion.schemas.proposal_frame import ProposalFrameV1

from .pg_engine import module_engine

router = APIRouter(prefix="/api/substrate/proposals", tags=["substrate-proposals"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return module_engine(__name__, uri)


def _load_latest_proposal_frame() -> ProposalFrameV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT proposal_frame_json FROM substrate_proposal_frames
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["proposal_frame_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    try:
        return ProposalFrameV1.model_validate(payload)
    except ValidationError:
        # A pre-migration row (e.g. 2026-07-22's SelfStateV1 burn) can be
        # incompatible with the currently-running ProposalFrameV1 -- degrade
        # to "not found" instead of 500ing this debug endpoint.
        return None


@router.get("/latest")
async def proposals_latest() -> dict[str, Any]:
    frame = await asyncio.to_thread(_load_latest_proposal_frame)
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
