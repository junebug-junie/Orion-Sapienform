from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import asyncpg

from orion.core.bus.async_service import OrionBusAsync
from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
from orion.memory.crystallization.chroma_publish import publish_crystallization_to_chroma
from orion.memory.crystallization.projection_cards import build_memory_card_projection
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.schemas import MemoryCrystallizationV1

logger = logging.getLogger(__name__)


@dataclass
class ProjectionConfig:
    collection: str = "orion_memory_crystallizations"
    vector_channel: str = "orion:memory:vector:upsert"
    project_channel: str = "orion:memory:crystallization:project"
    embed_host_url: str = ""
    embed_mode: str = "http"
    embed_request_channel: str = "orion:embedding:generate"
    embed_result_channel: str = "orion:embedding:result"
    embed_timeout_ms: int = 8000
    graphiti_enabled: bool = False
    graphiti_url: str = ""
    falkordb_uri: str = ""
    service_name: str = "orion-hub"
    service_version: str = "0.1.0"
    node_name: str = "hub"


@dataclass
class ProjectionResult:
    card_id: str | None = None
    chroma: dict[str, Any] = field(default_factory=dict)
    graphiti: dict[str, Any] = field(default_factory=dict)
    bus_project_emitted: bool = False
    errors: list[str] = field(default_factory=list)


async def project_crystallization(
    pool: asyncpg.Pool,
    bus: Optional[OrionBusAsync],
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str,
    config: ProjectionConfig | None = None,
    project_card: bool = True,
    project_chroma: bool = True,
    project_graphiti: bool = True,
) -> tuple[MemoryCrystallizationV1, ProjectionResult]:
    """Project active crystallization to derived stores and emit bus project event."""
    cfg = config or ProjectionConfig()
    result = ProjectionResult()
    updated = crystallization.model_copy(deep=True)

    if updated.status != "active":
        result.errors.append(f"projection_requires_active status={updated.status}")
        return updated, result

    if project_card:
        card_create = build_memory_card_projection(updated)
        if card_create is not None:
            try:
                card_id = await mc_dal.insert_card(pool, card_create, actor=actor, op="crystallization_project")
                result.card_id = str(card_id)
                if str(card_id) not in updated.projection_refs.memory_card_ids:
                    updated.projection_refs.memory_card_ids = list(updated.projection_refs.memory_card_ids) + [str(card_id)]
            except Exception as exc:
                logger.warning("card_projection_failed id=%s error=%s", updated.crystallization_id, exc)
                result.errors.append(f"card_projection_failed:{exc}")

    if project_chroma:
        try:
            updated, chroma_result = await publish_crystallization_to_chroma(
                updated,
                bus,
                collection=cfg.collection,
                vector_channel=cfg.vector_channel,
                embed_host_url=cfg.embed_host_url,
                embed_mode=cfg.embed_mode,
                embed_request_channel=cfg.embed_request_channel,
                embed_result_channel=cfg.embed_result_channel,
                embed_timeout_ms=cfg.embed_timeout_ms,
                service_name=cfg.service_name,
            )
            result.chroma = chroma_result
        except Exception as exc:
            logger.warning("chroma_projection_failed id=%s error=%s", updated.crystallization_id, exc)
            result.errors.append(f"chroma_projection_failed:{exc}")

    if updated.governance.sensitivity == "intimate":
        result.errors.append("graphiti_projection_skipped:intimate_sensitivity")
    elif project_graphiti and cfg.graphiti_enabled:
        try:
            adapter = GraphitiAdapter(enabled=True, url=cfg.graphiti_url, falkordb_uri=cfg.falkordb_uri)
            gresult = adapter.sync_crystallization(updated)
            updated = adapter.apply_projection_refs(updated, gresult)
            result.graphiti = {
                "episode_ids": gresult.episode_ids,
                "entity_ids": gresult.entity_ids,
                "edge_ids": gresult.edge_ids,
                "canonical_mutated": gresult.canonical_mutated,
            }
        except Exception as exc:
            result.errors.append(f"graphiti_projection_failed:{exc}")

    result.bus_project_emitted = await emit_crystallization_lifecycle(
        bus,
        lifecycle="project",
        crystallization=updated,
        service_name=cfg.service_name,
        service_version=cfg.service_version,
        node_name=cfg.node_name,
        channel=cfg.project_channel,
    )
    return updated, result
