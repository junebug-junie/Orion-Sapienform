from __future__ import annotations

import logging
from typing import Any

import asyncpg

from orion.core.bus.async_service import OrionBusAsync
from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
from orion.memory.crystallization.concept_relation import (
    maybe_resolve_concept_relation,
    merge_new_evidence,
)
from orion.memory.crystallization.detection import detect_duplicates
from orion.memory.crystallization.dynamics import reinforce
from orion.memory.crystallization.formation_executor import GovernorPathRequired, auto_activate
from orion.memory.crystallization.formation_policy import (
    DEFAULT_DISCARD_PLATFORMS,
    FormationPolicy,
    resolve_formation_policy,
    source_platform_of,
)
from orion.memory.crystallization.projector import ProjectionConfig, project_crystallization
from orion.memory.crystallization.repository import (
    insert_crystallization,
    list_crystallizations,
    update_crystallization,
)
from orion.memory.crystallization.salience import apply_salience
from orion.memory.crystallization.schemas import MemoryCrystallizationV1, _utc_now

logger = logging.getLogger(__name__)


def _discard_platforms(settings: Any) -> frozenset[str]:
    """Platform discard-set from the caller's settings, falling back to the module
    default. Tolerates settings objects (tests, other services) that predate the
    `discard_platforms` property rather than requiring every caller to grow it --
    getattr, not hasattr-then-access, so a property that raises is also handled
    the same way the rest of this module treats optional settings.
    """
    value = getattr(settings, "discard_platforms", None)
    if value is None:
        return DEFAULT_DISCARD_PLATFORMS
    return frozenset(value)


def _emit_settings(settings: Any) -> dict[str, str]:
    return {
        "service_name": getattr(settings, "SERVICE_NAME", "orion-memory-consolidation"),
        "service_version": getattr(settings, "SERVICE_VERSION", "0.1.0"),
        "node_name": getattr(settings, "NODE_NAME", "consolidation"),
    }


async def process_consolidation_crystallization(
    pool: asyncpg.Pool,
    bus: OrionBusAsync | None,
    *,
    crystallization: MemoryCrystallizationV1,
    settings: Any,
    project_config: ProjectionConfig | None,
) -> tuple[str | None, MemoryCrystallizationV1, str]:
    """Returns (crystallization_id, final_row, outcome).

    Outcome is one of:
    auto_activated | proposed | reinforced | reinforced_by_relation | discarded_external_platform.
    crystallization_id is None only for discarded_external_platform -- nothing was
    inserted, so there is no id to return. Callers must not pass a None id into
    anything that expects an existing crystallizations row (e.g. window bookkeeping
    that records a crystallization_id -- use the outcome string to branch instead).
    """
    emit_kw = _emit_settings(settings)

    # Checked before duplicate detection (which would otherwise run a real query)
    # and before any DB write: an allowlisted platform's window never becomes a
    # crystallization at all, of any kind -- not proposed, not auto-activated, not
    # reinforced into an existing memory. See formation_policy.DEFAULT_DISCARD_PLATFORMS
    # for why. The raw turns are unaffected; sql-writer already persisted them to
    # chat_history_log independently of this pipeline.
    discard_platforms = _discard_platforms(settings)
    platform = source_platform_of(crystallization)
    if platform is not None and platform in discard_platforms:
        logger.info(
            "crystallization_discarded_external_platform platform=%s kind=%s window=%s",
            platform,
            crystallization.kind,
            (crystallization.provenance or {}).get("memory_window_id"),
        )
        return None, crystallization, "discarded_external_platform"

    existing = await list_crystallizations(pool, status=None, limit=200)
    detection = detect_duplicates(crystallization, existing)
    duplicate_id = detection.duplicates[0] if detection.duplicates else None

    # Mutually exclusive with the REINFORCE_EXISTING branch below by construction: this
    # only fires when no same-window duplicate was found, so the already-working
    # scope-gated Jaccard dedup path (guarded on a truthy duplicate_id) always takes
    # priority whenever it applies, regardless of program order.
    if duplicate_id is None and getattr(settings, "CONCEPT_RELATION_RESOLUTION_ENABLED", False):
        relation_result = await maybe_resolve_concept_relation(
            pool, bus, candidate=crystallization, settings=settings, emit_kw=emit_kw,
        )
        if relation_result is not None:
            return relation_result

    policy, _ = resolve_formation_policy(
        crystallization, duplicate_id=duplicate_id, discard_platforms=discard_platforms
    )

    if policy == FormationPolicy.REINFORCE_EXISTING and duplicate_id:
        match = next(c for c in existing if c.crystallization_id == duplicate_id)
        merged = merge_new_evidence(match, crystallization)
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
            discard_platforms=discard_platforms,
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
    except GovernorPathRequired as exc:
        row = apply_salience(crystallization)
        row.provenance = {
            **(row.provenance or {}),
            "formation_policy_downgrade": str(exc),
            "formation_policy": "governor_queue",
        }
        cid = await insert_crystallization(pool, row)
        await emit_crystallization_lifecycle(
            bus,
            lifecycle="proposed",
            crystallization=row,
            **emit_kw,
        )
        return cid, row, "proposed"
