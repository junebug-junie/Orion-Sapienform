"""Action outcome persistence and read API for the autonomy feedback loop."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from psycopg2.extras import Json
from sqlalchemy import create_engine, text

from orion.autonomy.models import ActionOutcomeRefV1

router = APIRouter(prefix="/api/substrate/autonomy", tags=["substrate-autonomy"])

_MAX_RECENT = 20


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def save_action_outcome(outcome: ActionOutcomeRefV1) -> None:
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return
    engine = create_engine(uri, pool_pre_ping=True)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO autonomy_action_outcomes (
                    outcome_id, action_id, kind, success, surprise,
                    observed_at, outcome_json, created_at
                ) VALUES (
                    :outcome_id, :action_id, :kind, :success, :surprise,
                    :observed_at, :outcome_json, :created_at
                )
                ON CONFLICT (outcome_id) DO NOTHING
                """
            ),
            {
                "outcome_id": f"outcome:{outcome.action_id}:{outcome.observed_at}",
                "action_id": outcome.action_id,
                "kind": outcome.kind,
                "success": outcome.success,
                "surprise": outcome.surprise,
                "observed_at": outcome.observed_at or now,
                "outcome_json": Json(outcome.model_dump(mode="json")),
                "created_at": now,
            },
        )


@router.get("/outcomes/recent")
def get_recent_action_outcomes(limit: int = 10) -> list[dict[str, Any]]:
    safe_limit = min(limit, _MAX_RECENT)
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT outcome_json FROM autonomy_action_outcomes
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": safe_limit},
        ).mappings().fetchall()
    out = []
    for row in rows:
        payload = row["outcome_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        out.append(payload)
    return out
