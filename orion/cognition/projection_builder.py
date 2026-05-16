"""Shared CognitiveUnificationLayer → CognitiveProjection builder.

Phase-3 seam: move the substrate/projection spine out of the chat stance
service module so both Exec and Orch can build the same pre-LLM cognitive
projection before Mind is allowed to claim synthesis authority.

This module owns no chat prompt text and no final-response behavior. It only:

1. wires the known substrate producer registry,
2. runs ``CognitiveUnificationLayer`` for a bounded anchor set,
3. optionally publishes cold-path tier telemetry, and
4. projects unified beliefs into ``CognitiveProjectionV1``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from orion.cognition.projection import CognitiveProjectionV1, project_unified_beliefs_for_mind
from orion.substrate import build_substrate_store_from_env
from orion.substrate.relational import (
    CONCEPT_INDUCED,
    GRAPHDB_DURABLE,
    OPERATOR_STATIC,
    SNAPSHOT_EPHEMERAL,
    CognitiveUnificationLayer,
    ProducerEntryV1,
    ProducerRegistryV1,
    UnifiedRelationalBeliefSetV1,
    map_autonomy_ctx_to_substrate,
    map_concept_induction_ctx_to_substrate,
    map_identity_yaml_to_substrate,
    map_orionmem_to_substrate,
    map_recall_bundle_to_substrate,
    map_self_study_to_substrate,
    map_social_ctx_to_substrate,
)
from orion.cognition.projection_context import summarize_projection_inputs
from orion.substrate.relational.layer import _lightweight_belief_set, _skip_unified_beliefs_ctx
from orion.substrate.relational.adapters.spark_ctx import map_spark_ctx_to_substrate

logger = logging.getLogger("orion.cognition.projection_builder")

DEFAULT_PROJECTION_ANCHORS: tuple[str, ...] = ("orion", "relationship", "juniper")

_UNIFICATION_LAYER: CognitiveUnificationLayer | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def build_projection_unification_registry() -> ProducerRegistryV1:
    """Construct the shared producer registry for cognitive projection reads."""
    return ProducerRegistryV1(
        producers=[
            ProducerEntryV1(
                producer_id="identity_yaml",
                trust_tier=OPERATOR_STATIC,
                anchor_scopes=("orion",),
                freshness_ttl_sec=86400,
                pull_on_cold=True,
                adapter_fn=map_identity_yaml_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="self_study",
                trust_tier=GRAPHDB_DURABLE,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_self_study_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="autonomy",
                trust_tier=GRAPHDB_DURABLE,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_autonomy_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="concept_induction",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_concept_induction_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="spark",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_spark_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="orionmem",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion", "relationship"),
                freshness_ttl_sec=120,
                pull_on_cold=True,
                adapter_fn=map_orionmem_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="recall",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=0,
                pull_on_cold=False,
                adapter_fn=map_recall_bundle_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="social",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("relationship",),
                freshness_ttl_sec=0,
                pull_on_cold=False,
                adapter_fn=map_social_ctx_to_substrate,
            ),
        ]
    )


def get_projection_unification_layer() -> CognitiveUnificationLayer:
    """Return the process-level CognitiveUnificationLayer used by projection builders."""
    global _UNIFICATION_LAYER
    if _UNIFICATION_LAYER is None:
        registry = build_projection_unification_registry()
        store = build_substrate_store_from_env()
        _UNIFICATION_LAYER = CognitiveUnificationLayer(registry=registry, store=store)
    return _UNIFICATION_LAYER


def _publish_tier_outcomes_if_needed(*, beliefs: UnifiedRelationalBeliefSetV1, ctx: dict[str, Any]) -> None:
    if not beliefs.cold_anchors:
        return
    try:
        from orion.substrate.tier_outcomes_bus import publish_substrate_tier_outcomes_sync

        tier_map = {a: s.tier_outcomes for a, s in beliefs.anchors.items() if s.tier_outcomes}
        publish_substrate_tier_outcomes_sync(
            generated_at=beliefs.generated_at,
            cold_anchors=list(beliefs.cold_anchors),
            tier_outcomes=tier_map,
            degraded_producers=list(beliefs.degraded_producers),
            ctx=ctx,
        )
        logger.debug(
            "substrate_tier_outcomes_bus cold_anchors=%s degraded=%s",
            beliefs.cold_anchors,
            beliefs.degraded_producers,
        )
    except Exception as exc:
        logger.warning("substrate_tier_outcomes_publish_failed error=%s", exc)


def unified_beliefs_for_context(
    ctx: dict[str, Any] | None,
    *,
    anchors: Sequence[str] = DEFAULT_PROJECTION_ANCHORS,
    timeout_sec: float | None = None,
    publish_tier_outcomes: bool = False,
) -> UnifiedRelationalBeliefSetV1 | None:
    """Build unified beliefs for a request context."""
    ctx = ctx if isinstance(ctx, dict) else {}
    anchors_resolved = tuple(anchors) if anchors else DEFAULT_PROJECTION_ANCHORS
    try:
        if _skip_unified_beliefs_ctx(ctx):
            logger.info(
                "cognitive_projection_builder_skip reason=spark_or_unified_beliefs_disabled verb=%s correlation_id=%s",
                ctx.get("verb"),
                ctx.get("correlation_id") or ctx.get("trace_id"),
            )
            return _lightweight_belief_set(anchors_resolved)

        layer = get_projection_unification_layer()
        beliefs = layer.beliefs_for_stance(
            anchors=anchors_resolved,
            ctx=ctx,
            timeout_sec=float(timeout_sec if timeout_sec is not None else _env_float("UNIFIED_BELIEFS_TIMEOUT_SEC", 5.0)),
        )
    except Exception as exc:
        logger.warning("unified_beliefs_for_context_failed error=%s", exc)
        return None

    if beliefs is not None and publish_tier_outcomes:
        _publish_tier_outcomes_if_needed(beliefs=beliefs, ctx=ctx)
    return beliefs


def unified_beliefs_for_chat_stance(
    ctx: dict[str, Any] | None,
    *,
    timeout_sec: float | None = None,
) -> UnifiedRelationalBeliefSetV1 | None:
    """Compatibility seam for Exec chat stance unified-belief reads.

    This is the Phase-4 convergence target for
    ``services/orion-cortex-exec/app/chat_stance.py::_unified_beliefs_for_stance``.
    It preserves chat stance's historical anchors and cold-path telemetry behavior
    while ensuring the registry/store/build logic is owned here rather than in the
    service module.
    """
    return unified_beliefs_for_context(
        ctx,
        anchors=DEFAULT_PROJECTION_ANCHORS,
        timeout_sec=timeout_sec,
        publish_tier_outcomes=True,
    )


def summarize_projection_build(
    ctx: dict[str, Any] | None,
    *,
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    projection: CognitiveProjectionV1 | None,
    build_path: str = "orion.cognition.projection_builder",
) -> dict[str, Any]:
    """Compact diagnostics when a projection is empty or degraded at build time."""
    ctx = ctx if isinstance(ctx, dict) else {}
    registry = build_projection_unification_registry()
    requested = [producer.producer_id for producer in registry.producers]
    degraded = list(getattr(beliefs, "degraded_producers", []) or []) if beliefs is not None else []
    returned = [producer_id for producer_id in requested if producer_id not in degraded]
    source_counts: dict[str, int] = {}
    if projection is not None:
        for anchor_name, anchor_slice in (projection.anchors or {}).items():
            source_counts[str(anchor_name)] = len(anchor_slice.items or [])
    dropped_counts_by_reason: dict[str, int] = {}
    for note in list(getattr(projection, "notes", []) or []):
        dropped_counts_by_reason[str(note)] = dropped_counts_by_reason.get(str(note), 0) + 1
    if beliefs is None:
        dropped_counts_by_reason["beliefs_absent"] = dropped_counts_by_reason.get("beliefs_absent", 0) + 1
    if _skip_unified_beliefs_ctx(ctx):
        dropped_counts_by_reason["short_circuit_unified_beliefs"] = (
            dropped_counts_by_reason.get("short_circuit_unified_beliefs", 0) + 1
        )
    item_count = int(getattr(projection, "item_count", 0) or 0) if projection is not None else 0
    phase = "orch_mind_preflight" if "orch.mind_runtime" in build_path else "exec_chat_stance"
    if "exec" in build_path or "chat_stance" in build_path:
        phase = "exec_chat_stance"
    return {
        "build_path": build_path,
        "input_summary": summarize_projection_inputs(ctx, phase=phase),
        "projection_sources_requested": requested,
        "projection_sources_returned": returned,
        "source_counts": source_counts,
        "dropped_counts_by_reason": dropped_counts_by_reason,
        "producer_errors": degraded,
        "short_circuit_policy_active": _skip_unified_beliefs_ctx(ctx),
        "cold_anchors": list(getattr(beliefs, "cold_anchors", []) or []) if beliefs is not None else [],
        "lineage": list(getattr(beliefs, "lineage", []) or []) if beliefs is not None else [],
        "item_count": item_count,
        "projection_id": getattr(projection, "projection_id", None) if projection is not None else None,
    }


def build_cognitive_projection_for_context(
    ctx: dict[str, Any] | None,
    *,
    anchors: Sequence[str] = DEFAULT_PROJECTION_ANCHORS,
    timeout_sec: float | None = None,
    max_items_per_bucket: int = 6,
    max_total_items: int = 48,
    publish_tier_outcomes: bool = False,
) -> CognitiveProjectionV1 | None:
    """Build a bounded ``CognitiveProjectionV1`` for Mind/LLM pre-context."""
    beliefs = unified_beliefs_for_context(
        ctx,
        anchors=anchors,
        timeout_sec=timeout_sec,
        publish_tier_outcomes=publish_tier_outcomes,
    )
    return project_unified_beliefs_for_mind(
        beliefs,
        max_items_per_bucket=max_items_per_bucket,
        max_total_items=max_total_items,
    )


def build_cognitive_projection_for_mind_with_diagnostics(
    ctx: dict[str, Any] | None,
    *,
    anchors: Sequence[str] = DEFAULT_PROJECTION_ANCHORS,
    timeout_sec: float | None = None,
    max_items_per_bucket: int = 6,
    max_total_items: int = 48,
    publish_tier_outcomes: bool = True,
    build_path: str = "orion.cognition.projection_builder.build_cognitive_projection_for_mind_with_diagnostics",
) -> tuple[CognitiveProjectionV1 | None, dict[str, Any]]:
    """Build projection plus starvation diagnostics for Mind convergence."""
    beliefs = unified_beliefs_for_context(
        ctx,
        anchors=anchors,
        timeout_sec=timeout_sec,
        publish_tier_outcomes=publish_tier_outcomes,
    )
    projection = project_unified_beliefs_for_mind(
        beliefs,
        max_items_per_bucket=max_items_per_bucket,
        max_total_items=max_total_items,
    )
    diagnostics = summarize_projection_build(
        ctx,
        beliefs=beliefs,
        projection=projection,
        build_path=build_path,
    )
    return projection, diagnostics
