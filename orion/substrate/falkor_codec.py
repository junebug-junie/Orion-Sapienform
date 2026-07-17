"""Cypher-native FalkorDB encoding for substrate graph records.

This module is intentionally pure: no Redis client, no store cache, no graph
queries. It owns the durable property allowlist used by FalkorSubstrateStore.

Durable Falkor support is intentionally Concept + SubstrateEdge first. Other
node kinds must not be silently persisted as incomplete native rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    ConceptNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)

_LABEL_BY_KIND: dict[str, str] = {
    "entity": "Entity",
    "concept": "Concept",
    "event": "Event",
    "evidence": "Evidence",
    "contradiction": "Contradiction",
    "tension": "Tension",
    "drive": "Drive",
    "goal": "Goal",
    "state_snapshot": "StateSnapshot",
    "hypothesis": "Hypothesis",
    "ontology_branch": "OntologyBranch",
}


def node_label_for_kind(node_kind: str) -> str:
    return _LABEL_BY_KIND.get(str(node_kind), "Generic")


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_list(values: list[Any] | None) -> str:
    return json.dumps(list(values or []), ensure_ascii=False, sort_keys=True)


def _parse_json_list(raw: Any) -> list[Any]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _common_node_properties(node: BaseSubstrateNodeV1, identity_key: str | None) -> dict[str, Any]:
    activation = node.signals.activation
    provenance = node.provenance
    temporal = node.temporal
    return {
        "node_id": node.node_id,
        "node_kind": node.node_kind,
        "identity_key": identity_key or "",
        "anchor_scope": node.anchor_scope,
        "subject_ref": node.subject_ref,
        "promotion_state": node.promotion_state,
        "risk_tier": node.risk_tier,
        "confidence": float(node.signals.confidence),
        "salience": float(node.signals.salience),
        "activation": float(activation.activation),
        "recency_score": float(activation.recency_score),
        "decay_half_life_seconds": activation.decay_half_life_seconds,
        "decay_floor": float(activation.decay_floor),
        "observed_at": _dt(temporal.observed_at),
        "valid_from": _dt(temporal.valid_from),
        "valid_to": _dt(temporal.valid_to),
        "provenance_authority": provenance.authority,
        "provenance_source_kind": provenance.source_kind,
        "provenance_source_channel": provenance.source_channel,
        "provenance_producer": provenance.producer,
        "provenance_model_name": provenance.model_name,
        "provenance_correlation_id": provenance.correlation_id,
        "provenance_trace_id": provenance.trace_id,
        "provenance_tier_rank": provenance.tier_rank,
        "evidence_refs_json": _json_list(provenance.evidence_refs),
    }


def encode_node_properties(node: BaseSubstrateNodeV1, identity_key: str | None) -> dict[str, Any]:
    if node.node_kind != "concept":
        raise ValueError(
            f"falkor durable path supports concept nodes only; got node_kind={node.node_kind!r}"
        )
    props = _common_node_properties(node, identity_key)
    props["label"] = getattr(node, "label")
    props["definition"] = getattr(node, "definition", None)
    props["taxonomy_path_json"] = _json_list(getattr(node, "taxonomy_path", None))
    return props


def encode_edge_properties(edge: SubstrateEdgeV1, identity_key: str) -> dict[str, Any]:
    provenance = edge.provenance
    temporal = edge.temporal
    return {
        "edge_id": edge.edge_id,
        "identity_key": identity_key,
        "source_id": edge.source.node_id,
        "source_kind": edge.source.node_kind,
        "target_id": edge.target.node_id,
        "target_kind": edge.target.node_kind,
        "predicate": edge.predicate,
        "substrate_edge": True,
        "confidence": float(edge.confidence),
        "salience": float(edge.salience),
        "observed_at": _dt(temporal.observed_at),
        "valid_from": _dt(temporal.valid_from),
        "valid_to": _dt(temporal.valid_to),
        "provenance_authority": provenance.authority,
        "provenance_source_kind": provenance.source_kind,
        "provenance_source_channel": provenance.source_channel,
        "provenance_producer": provenance.producer,
        "provenance_model_name": provenance.model_name,
        "provenance_correlation_id": provenance.correlation_id,
        "provenance_trace_id": provenance.trace_id,
        "provenance_tier_rank": provenance.tier_rank,
        "evidence_refs_json": _json_list(provenance.evidence_refs),
    }


def _temporal_from_row(row: Mapping[str, Any]) -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(
        observed_at=row["observed_at"],
        valid_from=row.get("valid_from"),
        valid_to=row.get("valid_to"),
    )


def _provenance_from_row(row: Mapping[str, Any]) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority=row["provenance_authority"],
        source_kind=row["provenance_source_kind"],
        source_channel=row["provenance_source_channel"],
        producer=row["provenance_producer"],
        model_name=row.get("provenance_model_name"),
        correlation_id=row.get("provenance_correlation_id"),
        trace_id=row.get("provenance_trace_id"),
        evidence_refs=[str(item) for item in _parse_json_list(row.get("evidence_refs_json"))],
        tier_rank=row.get("provenance_tier_rank"),
    )


def _signals_from_row(row: Mapping[str, Any]) -> SubstrateSignalBundleV1:
    return SubstrateSignalBundleV1(
        confidence=float(row.get("confidence") or 0.5),
        salience=float(row.get("salience") or 0.0),
        activation=SubstrateActivationV1(
            activation=float(row.get("activation") or 0.0),
            recency_score=float(row.get("recency_score") or 0.0),
            decay_half_life_seconds=row.get("decay_half_life_seconds"),
            decay_floor=float(row.get("decay_floor") or 0.0),
        ),
    )


def decode_concept_node(row: Mapping[str, Any]) -> ConceptNodeV1 | None:
    if row.get("node_kind") != "concept":
        return None
    return ConceptNodeV1(
        node_id=str(row["node_id"]),
        label=str(row["label"]),
        definition=row.get("definition"),
        taxonomy_path=[str(item) for item in _parse_json_list(row.get("taxonomy_path_json"))],
        anchor_scope=row["anchor_scope"],
        subject_ref=row.get("subject_ref"),
        promotion_state=row.get("promotion_state") or "proposed",
        risk_tier=row.get("risk_tier") or "low",
        temporal=_temporal_from_row(row),
        signals=_signals_from_row(row),
        provenance=_provenance_from_row(row),
        metadata={},
    )


def decode_edge(row: Mapping[str, Any]) -> SubstrateEdgeV1 | None:
    if not row.get("edge_id"):
        return None
    return SubstrateEdgeV1(
        edge_id=str(row["edge_id"]),
        source=NodeRefV1(
            node_id=str(row["source_id"]),
            node_kind=row.get("source_kind") or "concept",
        ),
        target=NodeRefV1(
            node_id=str(row["target_id"]),
            node_kind=row.get("target_kind") or "concept",
        ),
        predicate=row["predicate"],
        temporal=_temporal_from_row(row),
        confidence=float(row.get("confidence") or 0.5),
        salience=float(row.get("salience") or 0.0),
        provenance=_provenance_from_row(row),
        metadata={},
    )
