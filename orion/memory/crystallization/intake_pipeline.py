from __future__ import annotations

from typing import Any

import asyncpg

from orion.core.bus.async_service import OrionBusAsync
from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
from orion.memory.crystallization.detection import detect_duplicates
from orion.memory.crystallization.dynamics import reinforce
from orion.memory.crystallization.formation_executor import GovernorPathRequired, auto_activate
from orion.memory.crystallization.formation_policy import FormationPolicy, resolve_formation_policy
from orion.memory.crystallization.projector import ProjectionConfig, project_crystallization
from orion.memory.crystallization.repository import (
    insert_crystallization,
    list_crystallizations,
    update_crystallization,
)
from orion.memory.crystallization.salience import apply_salience
from orion.memory.crystallization.schemas import MemoryCrystallizationV1, _utc_now


def _emit_settings(settings: Any) -> dict[str, str]:
    return {
        "service_name": getattr(settings, "SERVICE_NAME", "orion-memory-consolidation"),
        "service_version": getattr(settings, "SERVICE_VERSION", "0.1.0"),
        "node_name": getattr(settings, "NODE_NAME", "consolidation"),
    }


def _append_new_evidence(
    target: MemoryCrystallizationV1,
    candidate: MemoryCrystallizationV1,
) -> MemoryCrystallizationV1:
    updated = target.model_copy(deep=True)
    seen = {(ev.source_kind, ev.source_id) for ev in updated.evidence}
    for ev in candidate.evidence:
        key = (ev.source_kind, ev.source_id)
        if key in seen:
            continue
        updated.evidence.append(ev)
        seen.add(key)
    return updated


async def process_consolidation_crystallization(
    pool: asyncpg.Pool,
    bus: OrionBusAsync | None,
    *,
    crystallization: MemoryCrystallizationV1,
    settings: Any,
    project_config: ProjectionConfig | None,
) -> tuple[str, MemoryCrystallizationV1, str]:
    """Returns (crystallization_id, final_row, outcome).

    Outcome is one of: auto_activated | proposed | reinforced.
    """
    emit_kw = _emit_settings(settings)

    existing = await list_crystallizations(pool, status=None, limit=200)
    detection = detect_duplicates(crystallization, existing)
    duplicate_id = detection.duplicates[0] if detection.duplicates else None
    policy, _ = resolve_formation_policy(crystallization, duplicate_id=duplicate_id)

    if policy == FormationPolicy.REINFORCE_EXISTING and duplicate_id:
        match = next(c for c in existing if c.crystallization_id == duplicate_id)
        merged = _append_new_evidence(match, crystallization)
        updated = reinforce(merged, now=_utc_now())
        await update_crystallization(pool, updated)
        await emit_crystallization_lifecycle(
            bus,
            lifecycle="reinforced",
            crystallization=updated,
            **emit_kw,
        )
        return duplicate_id, updated, "reinforced"

    if not settings.MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED:
        row = apply_salience(crystallization)
        cid = await insert_crystallization(pool, row)
        await emit_crystallization_lifecycle(
            bus,
            lifecycle="proposed",
            crystallization=row,
            **emit_kw,
        )
        return cid, row, "proposed"

    try:
        activated, _ = auto_activate(
            apply_salience(crystallization),
            encode_ratio=settings.MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO,
        )
        cid = await insert_crystallization(pool, activated)
        activated, _proj = await project_crystallization(
            pool,
            bus,
            activated,
            actor="system:formation_policy",
            config=project_config,
            project_graphiti=False,
        )
        await update_crystallization(pool, activated)
        await emit_crystallization_lifecycle(
            bus,
            lifecycle="auto_activated",
            crystallization=activated,
            **emit_kw,
        )
        return cid, activated, "auto_activated"
    except GovernorPathRequired:
        row = apply_salience(crystallization)
        cid = await insert_crystallization(pool, row)
        await emit_crystallization_lifecycle(
            bus,
            lifecycle="proposed",
            crystallization=row,
            **emit_kw,
        )
        return cid, row, "proposed"
