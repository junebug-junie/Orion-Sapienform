"""Autonomy context adapter — graphdb_durable tier.

Wraps ``build_autonomy_repository`` and maps its result to substrate nodes
directly via ``_map_autonomy_state_to_nodes`` below (the separate
``orion.substrate.adapters.autonomy.map_autonomy_artifacts_to_substrate``
this docstring used to reference was removed 2026-07-30,
chore/delete-orion-drives Wave 2c -- it had zero production callers once
services/orion-substrate-runtime's own drive_state materializer was deleted
in the same sprint; this adapter never actually called it) so that the
autonomy producer lane can be registered in ProducerRegistryV1.

When ``AUTONOMY_GRAPH_BACKEND=graphdb`` or SPARQL/Fuseki endpoints resolve, the adapter resolves SPARQL endpoint
from env, applies quick-lane bounds for fast chat verbs, and maps each available
``AutonomyStateV1`` into substrate nodes. When the gate is off (V1 default),
returns ``None`` without calling the graph endpoint.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from orion.autonomy.fanout_policy import autonomy_subject_fanout_from_runtime_ctx
from orion.core.schemas.cognitive_substrate import (
    GoalNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.autonomy_ctx")

_TIER_RANK = 2  # graphdb_durable


def _make_prov(*, subject: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="autonomy.state",
        source_channel="sparql.graph",
        producer="autonomy_ctx_adapter",
        tier_rank=_TIER_RANK,
        evidence_refs=[f"autonomy:subject:{subject}"],
    )


def _map_autonomy_state_to_nodes(state: Any, *, anchor: str) -> list[Any]:
    """Map an AutonomyStateV1 into substrate nodes.

    Only GoalNodeV1 remains as of 2026-07-30 (chore/delete-orion-drives Wave
    2a follow-up): the StateSnapshotNodeV1/DriveNodeV1/TensionNodeV1
    branches this used to build from drive_pressures/active_drives/
    tension_kinds/dominant_drive were permanently dead code -- those fields
    no longer exist on AutonomyStateV1 (removed in Wave 2a), so every
    getattr(state, ..., default) here always returned its default and these
    branches could never fire. Removed rather than left as unreachable code
    per CLAUDE.md's no-empty-shell-cognition rule. DriveNodeV1/TensionNodeV1
    themselves are not retired -- orion/substrate/relational/adapters/
    recall.py and orion/substrate/adapters/spark.py still produce them from
    real, non-drive sources.
    """
    nodes: list[Any] = []
    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov(subject=anchor)

    goal_headlines = list(getattr(state, "goal_headlines", []) or [])

    # GoalNodeV1 for each goal headline
    for gh in goal_headlines[:5]:
        headline = str(getattr(gh, "headline", gh) or "").strip()[:200]
        if not headline:
            continue
        nodes.append(
            GoalNodeV1(
                anchor_scope=anchor,
                goal_text=headline,
                priority=float(getattr(gh, "priority", 0.5) or 0.5),
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.75, salience=0.5),
                metadata={"proposal_signature": headline[:64].lower().replace(" ", "_")},
            )
        )

    return nodes


def map_autonomy_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Fetch autonomy state for all subjects and map to substrate nodes (graphdb_durable)."""
    verb = str(ctx.get("verb") or ctx.get("requested_verb") or "").strip().lower()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    lane = str(
        ctx.get("execution_lane")
        or ctx.get("llm_lane")
        or opts.get("execution_lane")
        or opts.get("llm_lane")
        or ""
    ).strip().lower()
    if (
        verb == "introspect_spark"
        or lane == "spark"
        or bool(ctx.get("skip_unified_beliefs"))
        or bool(ctx.get("skip_autonomy_context"))
        or bool(opts.get("skip_unified_beliefs"))
        or bool(opts.get("skip_autonomy_context"))
    ):
        logger.info(
            "autonomy_ctx_adapter_skip reason=spark_or_unified_beliefs_disabled verb=%s lane=%s correlation_id=%s",
            verb,
            lane,
            ctx.get("correlation_id") or ctx.get("trace_id"),
        )
        return None

    try:
        from orion.autonomy.graph_gate import (
            is_quick_autonomy_graph_lane,
            log_autonomy_graph_backend_decision,
            resolve_autonomy_graph_read_plan,
        )
        from orion.autonomy.repository import build_autonomy_repository  # noqa: PLC0415 — lazy to avoid spacy at import time
    except ImportError as exc:
        logger.debug("autonomy_ctx_adapter_import_failed error=%s", exc)
        return None

    mode = str(ctx.get("mode") or "").strip().lower()
    plan = resolve_autonomy_graph_read_plan(ctx)
    log_autonomy_graph_backend_decision(plan=plan, consumer="autonomy_ctx_adapter", verb=verb, mode=mode)

    if plan.mode not in ("graphdb", "sparql") or not plan.endpoint:
        reason = plan.skipped_reason or "backend_disabled"
        if plan.mode == "graphdb_degraded":
            logger.info(
                "autonomy_graph_backend_degraded consumer=autonomy_ctx_adapter verb=%s explicit=true reason=%s fallback=skip_adapter",
                verb,
                reason,
            )
        elif plan.mode == "sparql_degraded":
            logger.info(
                "autonomy_graph_backend_degraded consumer=autonomy_ctx_adapter verb=%s reason=%s fallback=skip_adapter",
                verb,
                reason,
            )
        else:
            logger.info(
                "autonomy_graph_backend_blocked consumer=autonomy_ctx_adapter verb=%s reason=%s fallback=skip_adapter",
                verb,
                reason,
            )
        return None

    try:
        subjects = list(plan.subjects)
        # Real backend is always LocalAutonomyRepository now -- the graph/shadow
        # backends were deleted 2026-07-30 (confirmed dead: no Fuseki, no GraphDB
        # container anywhere; this branch was already unreachable in production
        # since plan.mode not in ("graphdb", "sparql") returns above it). See
        # orion/autonomy/repository.py's comment for the full rationale.
        repository = build_autonomy_repository()
    except Exception as exc:
        logger.debug("autonomy_ctx_adapter_init_failed error=%s", exc)
        return None

    correlation_id = str(ctx.get("correlation_id") or ctx.get("trace_id") or "")
    session_id = str(ctx.get("session_id") or "")
    observer = {
        "consumer": "autonomy_ctx_adapter",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "autonomy_subject_fanout": autonomy_subject_fanout_from_runtime_ctx(ctx),
    }

    try:
        lookups = repository.list_latest(subjects, observer=observer)
    except Exception as exc:
        logger.debug("autonomy_ctx_adapter_fetch_failed error=%s", exc)
        return None

    all_nodes: list[Any] = []
    for lookup in lookups:
        if lookup.availability != "available" or lookup.state is None:
            continue
        anchor = lookup.subject
        if anchor not in ("orion", "relationship", "juniper"):
            continue
        nodes = _map_autonomy_state_to_nodes(lookup.state, anchor=anchor)
        all_nodes.extend(nodes)

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=all_nodes) if all_nodes else None
