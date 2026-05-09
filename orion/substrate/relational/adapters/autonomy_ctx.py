"""Autonomy context adapter — graphdb_durable tier.

Wraps ``build_autonomy_repository`` + ``map_autonomy_artifacts_to_substrate``
so that the autonomy producer lane can be registered in ProducerRegistryV1.

The adapter builds the autonomy repository using the same env config as the
existing ``_load_autonomy_state`` path in chat_stance.py, queries all three
subjects (orion, relationship, juniper), maps each available ``AutonomyStateV1``
into substrate nodes, and returns a single combined ``SubstrateGraphRecordV1``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

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
_SUBJECTS = ("orion", "relationship", "juniper")


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
        source_channel="graphdb.sparql",
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
    try:
        from orion.autonomy.repository import build_autonomy_repository  # noqa: PLC0415 — lazy to avoid spacy at import time
    except ImportError as exc:
        logger.debug("autonomy_ctx_adapter_import_failed error=%s", exc)
        return None

    try:
        endpoint_raw = (
            os.getenv("GRAPHDB_QUERY_ENDPOINT")
            or os.getenv("GRAPHDB_URL")
            or os.getenv("CONCEPT_PROFILE_GRAPHDB_ENDPOINT")
            or os.getenv("CONCEPT_PROFILE_GRAPHDB_URL")
            or ""
        ).strip()
        repo = (os.getenv("GRAPHDB_REPO") or os.getenv("CONCEPT_PROFILE_GRAPHDB_REPO") or "collapse").strip() or "collapse"
        user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
        password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None

        endpoint = endpoint_raw
        if endpoint and endpoint.rstrip("/").endswith("/repositories"):
            endpoint = f"{endpoint.rstrip('/')}/{repo}"
        elif endpoint and "/repositories/" not in endpoint:
            endpoint = f"{endpoint.rstrip('/')}/repositories/{repo}"

        backend = (os.getenv("AUTONOMY_REPOSITORY_BACKEND") or "graph").strip().lower()
        if backend not in {"graph", "local", "shadow"}:
            backend = "graph"

        repository = build_autonomy_repository(
            backend=backend,
            endpoint=endpoint or None,
            timeout_sec=max(0.25, _env_float("AUTONOMY_GRAPH_TIMEOUT_SEC", _env_float("GRAPHDB_TIMEOUT_SEC", 4.5))),
            user=user,
            password=password,
            goals_limit=_env_int("AUTONOMY_GOALS_LIMIT", 3),
            subject_max_workers=max(1, _env_int("AUTONOMY_SUBJECT_MAX_WORKERS", 3)),
            subquery_max_workers=max(1, min(3, _env_int("AUTONOMY_SUBQUERY_MAX_WORKERS", 1))),
        )
    except Exception as exc:
        logger.debug("autonomy_ctx_adapter_init_failed error=%s", exc)
        return None

    correlation_id = str(ctx.get("correlation_id") or ctx.get("trace_id") or "")
    session_id = str(ctx.get("session_id") or "")
    observer = {"consumer": "autonomy_ctx_adapter", "correlation_id": correlation_id, "session_id": session_id}

    try:
        lookups = repository.list_latest(list(_SUBJECTS), observer=observer)
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
