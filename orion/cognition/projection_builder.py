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
        # Projection should remain available even if telemetry publication is down.
        logger.warning("substrate_tier_outcomes_publish_failed error=%s", exc)


def unified_beliefs_for_context(
    ctx: dict[str, Any] | None,
    *,
    anchors: Sequence[str] = DEFAULT_PROJECTION_ANCHORS,
    timeout_sec: float | None = None,
    publish_tier_outcomes: bool = False,
) -> UnifiedRelationalBeliefSetV1 | None:
    """Build unified beliefs for a request context.

    Returns a lightweight belief set for explicit skip lanes and ``None`` only on
    total initialization/runtime failure. Individual producer failures remain
    encoded in ``degraded_producers`` and anchor ``degraded`` flags.
    """
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
