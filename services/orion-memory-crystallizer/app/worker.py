from __future__ import annotations

import logging
from typing import Any, Optional

import asyncpg

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.bus_service_chassis import Hunter
from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.repository import get_crystallization, insert_crystallization, insert_history
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
)

from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)


async def handle_memory_card_envelope(env: BaseEnvelope, pool: asyncpg.Pool, bus_hunter: Hunter) -> None:
    """Propose crystallization from high-salience memory card events (pre-governed proposals only)."""
    payload: dict[str, Any]
    if hasattr(env.payload, "model_dump"):
        payload = env.payload.model_dump()
    elif isinstance(env.payload, dict):
        payload = env.payload
    else:
        return

    card_id = str(payload.get("card_id") or env.id or "")
    summary = str(payload.get("summary") or payload.get("title") or "").strip()
    if not summary or not card_id:
        return

    priority = str(payload.get("priority") or "")
    if priority not in ("high_recall", "always_inject"):
        return

    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic",
        subject=str(payload.get("title") or "Card consolidation"),
        summary=summary,
        scope=list(payload.get("visibility_scope") or ["project:orion"]),
        source_card_ids=[card_id],
        evidence=[CrystallizationEvidenceRefV1(source_kind="memory_card", source_id=card_id, excerpt=summary[:200])],
        proposed_by=settings.SERVICE_NAME,
    )
    crystallization = propose(req)
    stored_id = await insert_crystallization(pool, crystallization)
    await insert_history(pool, crystallization_id=stored_id, op="propose", actor=settings.SERVICE_NAME, before=None, after={"status": crystallization.status})
    row = await get_crystallization(pool, stored_id)
    if row and bus_hunter.bus is not None:
        await emit_crystallization_lifecycle(
            bus_hunter.bus,
            lifecycle="proposed",
            crystallization=row,
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            channel=settings.CRYSTALLIZER_CHANNEL_PROPOSED,
        )
        logger.info("crystallizer_proposed_from_card card_id=%s crystallization_id=%s", card_id, stored_id)
