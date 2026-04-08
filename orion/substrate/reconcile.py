from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateEdgeV1,
)


@dataclass(frozen=True)
class NodeMergeDecision:
    canonical_node_id: str
    merged: bool
    reason: str


@dataclass(frozen=True)
class EdgeMergeDecision:
    canonical_edge_id: str
    merged: bool
    reason: str


class SubstrateIdentityResolver:
    """Conservative deterministic identity resolver for substrate materialization."""

    def canonical_node_key(self, node: BaseSubstrateNodeV1) -> str | None:
        subject = str(node.subject_ref or "")
        scope = node.anchor_scope
        if node.node_kind == "concept":
            concept_id = str(node.metadata.get("concept_id") or "").strip().lower()
            if concept_id:
                return f"concept|{scope}|{subject}|{concept_id}"
            label = str(getattr(node, "label", "")).strip().lower()
            if label:
                return f"concept|{scope}|{subject}|label:{label}"
            return None
        if node.node_kind == "drive":
            drive_kind = str(getattr(node, "drive_kind", "")).strip().lower()
            return f"drive|{scope}|{subject}|{drive_kind}" if drive_kind else None
        if node.node_kind == "goal":
            signature = str(node.metadata.get("proposal_signature") or "").strip().lower()
            return f"goal|{scope}|{subject}|sig:{signature}" if signature else None
        if node.node_kind == "evidence":
            content_ref = str(getattr(node, "content_ref", "")).strip().lower()
            evidence_type = str(getattr(node, "evidence_type", "")).strip().lower()
            return f"evidence|{scope}|{subject}|{evidence_type}|{content_ref}" if content_ref else None
        if node.node_kind == "tension":
            artifact_id = str(node.metadata.get("artifact_id") or "").strip().lower()
            if artifact_id:
                return f"tension|{scope}|{subject}|artifact:{artifact_id}"
            return None
        if node.node_kind == "contradiction":
            delta_id = str(node.metadata.get("delta_id") or "").strip().lower()
            if delta_id:
                return f"contradiction|{scope}|{subject}|delta:{delta_id}"
            summary = str(getattr(node, "summary", "")).strip().lower()
            involved = ",".join(sorted(getattr(node, "involved_node_ids", [])))
            if summary and involved:
                digest = hashlib.sha256(f"{summary}|{involved}".encode("utf-8", errors="ignore")).hexdigest()[:16]
                return f"contradiction|{scope}|{subject}|hash:{digest}"
            return None
        if node.node_kind == "state_snapshot":
            # snapshots are distinct events by default; do not over-collapse
            return None
        if node.node_kind == "hypothesis":
            return None
        return None

    @staticmethod
    def canonical_edge_key(edge: SubstrateEdgeV1) -> str:
        return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"


def merge_node(existing: BaseSubstrateNodeV1, incoming: BaseSubstrateNodeV1, *, source_graph_id: str) -> BaseSubstrateNodeV1:
    lineage = list(existing.metadata.get("materialization_lineage") or [])
    lineage.append(
        {
            "source_graph_id": source_graph_id,
            "source_node_id": incoming.node_id,
            "observed_at": incoming.temporal.observed_at.isoformat(),
            "provenance": incoming.provenance.model_dump(mode="json"),
        }
    )
    lineage = lineage[-50:]

    merged_evidence_refs = sorted({*existing.provenance.evidence_refs, *incoming.provenance.evidence_refs})
    merged_provenance = existing.provenance.model_copy(update={"evidence_refs": merged_evidence_refs})

    valid_from_candidates = [value for value in (existing.temporal.valid_from, incoming.temporal.valid_from) if value is not None]
    valid_to_candidates = [value for value in (existing.temporal.valid_to, incoming.temporal.valid_to) if value is not None]
    merged_temporal = existing.temporal.model_copy(
        update={
            "observed_at": max(existing.temporal.observed_at, incoming.temporal.observed_at),
            "valid_from": min(valid_from_candidates) if valid_from_candidates else None,
            "valid_to": max(valid_to_candidates) if valid_to_candidates else None,
        }
    )

    merged_metadata: dict[str, Any] = {**incoming.metadata, **existing.metadata}
    merged_metadata["materialization_lineage"] = lineage

    merged_signals = existing.signals.model_copy(
        update={
            "confidence": max(existing.signals.confidence, incoming.signals.confidence),
            "salience": max(existing.signals.salience, incoming.signals.salience),
            "activation": existing.signals.activation.model_copy(
                update={
                    "activation": max(existing.signals.activation.activation, incoming.signals.activation.activation),
                    "recency_score": max(existing.signals.activation.recency_score, incoming.signals.activation.recency_score),
                    "decay_half_life_seconds": existing.signals.activation.decay_half_life_seconds
                    or incoming.signals.activation.decay_half_life_seconds,
                    "decay_floor": max(existing.signals.activation.decay_floor, incoming.signals.activation.decay_floor),
                }
            ),
        }
    )

    return existing.model_copy(
        update={
            "temporal": merged_temporal,
            "provenance": merged_provenance,
            "signals": merged_signals,
            "metadata": merged_metadata,
        }
    )


def merge_edge(existing: SubstrateEdgeV1, incoming: SubstrateEdgeV1, *, source_graph_id: str) -> SubstrateEdgeV1:
    lineage = list(existing.metadata.get("materialization_lineage") or [])
    lineage.append(
        {
            "source_graph_id": source_graph_id,
            "source_edge_id": incoming.edge_id,
            "observed_at": incoming.temporal.observed_at.isoformat(),
        }
    )
    lineage = lineage[-100:]
    merged_metadata = {**incoming.metadata, **existing.metadata, "materialization_lineage": lineage}
    merged_evidence_refs = sorted({*existing.provenance.evidence_refs, *incoming.provenance.evidence_refs})
    return existing.model_copy(
        update={
            "confidence": max(existing.confidence, incoming.confidence),
            "salience": max(existing.salience, incoming.salience),
            "provenance": existing.provenance.model_copy(update={"evidence_refs": merged_evidence_refs}),
            "temporal": existing.temporal.model_copy(update={"observed_at": max(existing.temporal.observed_at, incoming.temporal.observed_at)}),
            "metadata": merged_metadata,
        }
    )
