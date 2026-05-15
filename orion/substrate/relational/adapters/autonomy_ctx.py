"""Autonomy context adapter — graphdb_durable tier.

Wraps ``build_autonomy_repository`` + ``map_autonomy_artifacts_to_substrate``
so that the autonomy producer lane can be registered in ProducerRegistryV1.

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
    DriveNodeV1,
    GoalNodeV1,
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    TensionNodeV1,
)
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.autonomy_ctx")

_TIER_RANK = 2  # graphdb_durable


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


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
    """Map an AutonomyStateV1 into substrate nodes."""
    nodes: list[Any] = []
    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov(subject=anchor)

    drive_pressures: dict[str, float] = dict(getattr(state, "drive_pressures", {}) or {})
    active_drives: list[str] = list(getattr(state, "active_drives", []) or [])
    tension_kinds: list[str] = list(getattr(state, "tension_kinds", []) or [])
    goal_headlines = list(getattr(state, "goal_headlines", []) or [])
    dominant_drive: str | None = getattr(state, "dominant_drive", None)
    identity_summary: str | None = getattr(state, "identity_summary", None)

    # StateSnapshotNodeV1 capturing the drive pressure map
    if drive_pressures:
        nodes.append(
            StateSnapshotNodeV1(
                node_id=f"sub-autonomy-state-{anchor}",
                anchor_scope=anchor,
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(
                    confidence=0.8,
                    salience=max(drive_pressures.values()) if drive_pressures else 0.0,
                ),
                snapshot_source="autonomy",
                dimensions={k: float(v) for k, v in drive_pressures.items()},
                metadata={
                    "dominant_drive": dominant_drive,
                    "identity_summary": identity_summary,
                    "anchor_strategy": getattr(state, "anchor_strategy", None),
                },
            )
        )

    # DriveNodeV1 for each drive
    all_drive_names: set[str] = set(active_drives) | set(drive_pressures.keys())
    for drive_name in sorted(n for n in all_drive_names if n):
        nodes.append(
            DriveNodeV1(
                node_id=f"sub-drive-{anchor}-{drive_name}",
                anchor_scope=anchor,
                drive_kind=drive_name,
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(
                    confidence=0.8,
                    salience=float(drive_pressures.get(drive_name, 0.5)),
                ),
                metadata={"active": drive_name in active_drives},
            )
        )

    # TensionNodeV1 for each tension kind
    for tension_kind in tension_kinds[:8]:
        if not tension_kind:
            continue
        nodes.append(
            TensionNodeV1(
                anchor_scope=anchor,
                tension_kind=tension_kind,
                intensity=0.5,
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
                metadata={"tension_kind": tension_kind},
            )
        )

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
        backend = (os.getenv("AUTONOMY_REPOSITORY_BACKEND") or "graph").strip().lower()
        if backend not in {"graph", "local", "shadow"}:
            backend = "graph"

        subjects = list(plan.subjects)
        if is_quick_autonomy_graph_lane(ctx):
            subject_workers = max(1, min(_env_int("AUTONOMY_SUBJECT_MAX_WORKERS", 3), len(subjects) or 1))
            subquery_workers = 1
        else:
            subject_workers = max(1, _env_int("AUTONOMY_SUBJECT_MAX_WORKERS", 3))
            subquery_workers = max(1, min(3, _env_int("AUTONOMY_SUBQUERY_MAX_WORKERS", 1)))

        repository = build_autonomy_repository(
            backend=backend,
            endpoint=plan.endpoint,
            timeout_sec=max(0.25, plan.timeout_sec),
            user=plan.user,
            password=plan.password,
            goals_limit=_env_int("AUTONOMY_GOALS_LIMIT", 3),
            subject_max_workers=subject_workers,
            subquery_max_workers=subquery_workers,
            active_subqueries=plan.active_subqueries,
        )
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
