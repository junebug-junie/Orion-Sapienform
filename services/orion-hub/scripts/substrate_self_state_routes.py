"""Read-only debug API for substrate self-state snapshots."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateV1

router = APIRouter(prefix="/api/substrate/self-state", tags=["substrate-self-state"])

logger = logging.getLogger(__name__)


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
    try:
        return SelfStateV1.model_validate(payload)
    except ValidationError:
        # A schema change (e.g. a removed dimension_id enum value) can leave
        # the last-saved row incompatible with the current schema. Treat as
        # "no snapshot available" -- the same graceful-degradation path every
        # other SelfStateV1 consumer takes (see the 2026-07-12 schema-drift
        # incident PR) -- instead of 500ing this debug endpoint forever on a
        # row that will never validate under the current code.
        logger.warning("self_state_latest_incompatible_schema", exc_info=True)
        return None


def _load_self_state_by_id(self_state_id: str) -> SelfStateV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT self_state_json FROM substrate_self_state
                WHERE self_state_id = :self_state_id
                LIMIT 1
                """
            ),
            {"self_state_id": self_state_id},
        ).mappings().first()
    if not row:
        return None
    payload = row["self_state_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    try:
        return SelfStateV1.model_validate(payload)
    except ValidationError:
        logger.warning("self_state_by_id_incompatible_schema", exc_info=True)
        return None


def _load_field_state_for_tick(tick_id: str) -> FieldStateV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT field_json FROM substrate_field_state
                WHERE tick_id = :tick_id
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
            {"tick_id": tick_id},
        ).mappings().first()
    if not row:
        return None
    payload = row["field_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    try:
        return FieldStateV1.model_validate(payload)
    except ValidationError:
        # Same schema-drift tolerance as the self-state loaders above --
        # a field-state row could theoretically go stale under the same
        # class of enum/shape change.
        logger.warning("field_state_for_tick_incompatible_schema", exc_info=True)
        return None


def _build_evidence_trail(state: SelfStateV1) -> dict[str, Any]:
    field_state = _load_field_state_for_tick(state.source_field_tick_id)
    return {
        "self_state_id": state.self_state_id,
        "source_field_tick_id": state.source_field_tick_id,
        "self_state": state.model_dump(mode="json"),
        "field_state_available": field_state is not None,
        "field_state": (
            {
                "tick_id": field_state.tick_id,
                "generated_at": field_state.generated_at.isoformat(),
                "node_vectors": field_state.node_vectors,
                "capability_vectors": field_state.capability_vectors,
                "capability_provenance": field_state.capability_provenance,
            }
            if field_state is not None
            else None
        ),
    }


@router.get("/latest")
async def self_state_latest() -> dict[str, Any]:
    state = _load_latest_self_state()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return state.model_dump(mode="json")


@router.get("/latest/evidence-trail")
async def self_state_latest_evidence_trail() -> dict[str, Any]:
    state = _load_latest_self_state()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return _build_evidence_trail(state)


@router.get("/{self_state_id}/evidence-trail")
async def self_state_evidence_trail(self_state_id: str) -> dict[str, Any]:
    state = _load_self_state_by_id(self_state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return _build_evidence_trail(state)
