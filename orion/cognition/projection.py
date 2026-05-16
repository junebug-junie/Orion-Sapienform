"""Compact cognitive projection artifacts for Mind shadow wiring.

This module deliberately sits *above* the CognitiveUnificationLayer read model:
``UnifiedRelationalBeliefSetV1`` remains the substrate/unification output, while
``CognitiveProjectionV1`` is the bounded, LLM/Mind-friendly active frontier.

Do not put service clients or chat prompt logic here. This file is a pure adapter
from substrate beliefs to a portable projection contract.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1

ProjectionBucketV1 = Literal["concept", "tension", "goal", "drive", "snapshot", "event"]


class CognitiveProjectionItemV1(BaseModel):
    """One compact, source-linked item in the active cognitive frontier."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default_factory=lambda: f"cog-proj-item-{uuid4()}")
    anchor: str
    bucket: ProjectionBucketV1
    node_id: str
    node_kind: str
    label: str = ""
    summary: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    producer: str | None = None
    authority: str | None = None
    tier_rank: int | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CognitiveProjectionAnchorV1(BaseModel):
    """Bounded projection for one anchor scope."""

    model_config = ConfigDict(extra="forbid")

    anchor: str
    degraded: bool = False
    tier_outcomes: list[str] = Field(default_factory=list)
    items: list[CognitiveProjectionItemV1] = Field(default_factory=list)


class CognitiveProjectionV1(BaseModel):
    """Portable pre-LLM cognitive projection for Mind shadow consumption."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cognitive.projection.v1"] = "cognitive.projection.v1"
    projection_id: str = Field(default_factory=lambda: f"cog-proj-{uuid4()}")
    generated_at: str
    source: Literal["cognitive_unification_layer"] = "cognitive_unification_layer"
    anchors: dict[str, CognitiveProjectionAnchorV1] = Field(default_factory=dict)
    cold_anchors: list[str] = Field(default_factory=list)
    degraded_producers: list[str] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)
    item_count: int = 0
    notes: list[str] = Field(default_factory=list)


def _compact(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _node_label(node: BaseSubstrateNodeV1) -> str:
    for attr in ("label", "summary", "goal_text", "hypothesis_text", "target_state", "drive_kind", "tension_kind", "event_type", "snapshot_source"):
        value = getattr(node, attr, None)
        if value:
            return _compact(value, limit=120)
    subject_ref = getattr(node, "subject_ref", None)
    if subject_ref:
        return _compact(subject_ref, limit=120)
    return _compact(node.node_id, limit=120)


def _node_summary(node: BaseSubstrateNodeV1) -> str:
    if hasattr(node, "definition") and getattr(node, "definition"):
        return _compact(getattr(node, "definition"), limit=240)
    if hasattr(node, "target_state") and getattr(node, "target_state"):
        return _compact(getattr(node, "target_state"), limit=240)
    if hasattr(node, "dimensions") and getattr(node, "dimensions"):
        return _compact(getattr(node, "dimensions"), limit=240)
    meta = getattr(node, "metadata", {}) or {}
    for key in ("summary", "description", "excerpt", "text"):
        if isinstance(meta, dict) and meta.get(key):
            return _compact(meta.get(key), limit=240)
    return _node_label(node)


def _item_from_node(anchor: str, bucket: ProjectionBucketV1, node: BaseSubstrateNodeV1) -> CognitiveProjectionItemV1:
    provenance = node.provenance
    signals = node.signals
    return CognitiveProjectionItemV1(
        anchor=anchor,
        bucket=bucket,
        node_id=node.node_id,
        node_kind=str(node.node_kind),
        label=_node_label(node),
        summary=_node_summary(node),
        salience=float(signals.salience or 0.0),
        confidence=float(signals.confidence or 0.0),
        producer=provenance.producer,
        authority=provenance.authority,
        tier_rank=provenance.tier_rank,
        evidence_refs=list(provenance.evidence_refs or [])[:8],
        metadata={
            "promotion_state": node.promotion_state,
            "risk_tier": node.risk_tier,
            "observed_at": node.temporal.observed_at.isoformat(),
        },
    )


def _rank_items(items: list[CognitiveProjectionItemV1]) -> list[CognitiveProjectionItemV1]:
    return sorted(
        items,
        key=lambda item: (
            float(item.salience or 0.0),
            float(item.confidence or 0.0),
            -float(item.tier_rank or 99),
            item.label,
        ),
        reverse=True,
    )


def _items_for_anchor(anchor_slice: AnchorBeliefSliceV1, *, max_items_per_bucket: int) -> list[CognitiveProjectionItemV1]:
    bucket_nodes: list[tuple[ProjectionBucketV1, list[BaseSubstrateNodeV1]]] = [
        ("concept", list(anchor_slice.concepts or [])),
        ("tension", list(anchor_slice.tensions or [])),
        ("goal", list(anchor_slice.goals or [])),
        ("drive", list(anchor_slice.drives or [])),
        ("snapshot", list(anchor_slice.snapshots or [])),
        ("event", list(anchor_slice.events or [])),
    ]
    out: list[CognitiveProjectionItemV1] = []
    for bucket, nodes in bucket_nodes:
        ranked = _rank_items([_item_from_node(anchor_slice.anchor, bucket, node) for node in nodes])
        out.extend(ranked[:max_items_per_bucket])
    return _rank_items(out)


def project_unified_beliefs_for_mind(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    *,
    max_items_per_bucket: int = 6,
    max_total_items: int = 48,
) -> CognitiveProjectionV1 | None:
    """Compress unified beliefs into a bounded Mind/LLM-facing projection.

    Returns ``None`` only when there is no belief set at all. Empty/skipped belief
    sets still produce a projection with lineage/degradation information so Mind
    and Hub can distinguish "no signal" from "not run".
    """
    if beliefs is None:
        return None

    anchors: dict[str, CognitiveProjectionAnchorV1] = {}
    total_items = 0
    for anchor_name, anchor_slice in (beliefs.anchors or {}).items():
        items = _items_for_anchor(anchor_slice, max_items_per_bucket=max(1, int(max_items_per_bucket)))
        remaining = max(0, int(max_total_items) - total_items)
        items = items[:remaining]
        total_items += len(items)
        anchors[anchor_name] = CognitiveProjectionAnchorV1(
            anchor=anchor_name,
            degraded=bool(anchor_slice.degraded),
            tier_outcomes=list(anchor_slice.tier_outcomes or [])[:12],
            items=items,
        )
        if total_items >= int(max_total_items):
            break

    notes: list[str] = []
    if not anchors:
        notes.append("no_anchors")
    if total_items == 0:
        notes.append("no_active_projection_items")
    if beliefs.cold_anchors:
        notes.append("cold_path_materialized")
    if beliefs.degraded_producers:
        notes.append("degraded_producers_present")

    return CognitiveProjectionV1(
        generated_at=beliefs.generated_at,
        anchors=anchors,
        cold_anchors=list(beliefs.cold_anchors or []),
        degraded_producers=list(beliefs.degraded_producers or []),
        lineage=list(beliefs.lineage or []),
        item_count=total_items,
        notes=notes,
    )
