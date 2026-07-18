"""Cypher-native FalkorDB encoding for substrate graph records.

This module is intentionally pure: no Redis client, no store cache, no graph
queries. It owns the durable property allowlist used by FalkorSubstrateStore.

Durable Falkor support is intentionally Concept + Evidence + SubstrateEdge.
Other node kinds must not be silently persisted as incomplete native rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    ConceptNodeV1,
    EvidenceNodeV1,
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


def _parse_json_list(raw: Any, *, field: str = "json_list") -> list[Any]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {field}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"invalid {field}: expected JSON list")
    return list(parsed)


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
    if node.node_kind not in ("concept", "evidence"):
        raise ValueError(
            "falkor durable path supports concept and evidence nodes only; "
            f"got node_kind={node.node_kind!r}"
        )
    props = _common_node_properties(node, identity_key)
    if node.node_kind == "concept":
        props["label"] = getattr(node, "label")
        props["definition"] = getattr(node, "definition", None)
        props["taxonomy_path_json"] = _json_list(getattr(node, "taxonomy_path", None))
        props.update(_dynamics_properties_from_metadata(node.metadata))
    else:
        props["evidence_type"] = getattr(node, "evidence_type")
        props["content_ref"] = getattr(node, "content_ref")
    return props


# Closed allowlist of dynamics-engine metadata keys promoted to native Cypher
# scalar properties (concept nodes only). These are the only metadata keys
# already-shipped consumers (SubstrateDynamicsEngine.tick(),
# attention_broadcast._node_salience(), substrate_pressure_signals()'s
# evidence_refs, and the AST/HOT reducer's optional harness_closure_signal
# narrative -- orion/substrate/attention_self_model.py) actually read/write;
# see docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md
# item 3's "promote to first-class Cypher properties only with a second
# consumer" escape hatch. `contributing_turn_ids` was added under this same
# rule once a second real consumer existed (see the two consumers named
# above). Do NOT add a generic metadata-dump path here.
DYNAMICS_METADATA_KEYS: tuple[str, ...] = (
    "dynamic_pressure",
    "dynamic_pressure_reason",
    "dormant",
    "dormancy_updated_at",
    "prediction_error",
    "contributing_turn_ids",
)

# Subset of DYNAMICS_METADATA_KEYS owned by SubstrateDynamicsEngine.tick()
# rather than by whichever writer is upserting this call -- any caller that
# upserts a concept node via a freshly-constructed model (not one round-tripped
# from the store) must carry these forward from the pre-existing node or its
# own upsert_node() call will durably reset dynamics.py's computed pressure/
# dormancy state back to defaults. See
# services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node.
DYNAMICS_ENGINE_OWNED_METADATA_KEYS: tuple[str, ...] = (
    "dynamic_pressure",
    "dynamic_pressure_reason",
    "dormant",
    "dormancy_updated_at",
)


def _safe_float(value: Any, *, default: float | None) -> float | None:
    """Tolerant numeric coercion matching the `.get(key) or default` pattern
    every current consumer of these keys already uses (attention_broadcast.py's
    `_f()`, dynamics.py, pressure.py). A corrupted/non-numeric metadata value
    must not abort the whole encode call -- and by extension must not abort
    SubstrateDynamicsEngine.tick()'s per-node loop -- just for one bad node.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dynamics_properties_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = metadata or {}
    return {
        "dynamic_pressure": _safe_float(meta.get("dynamic_pressure"), default=0.0),
        "dynamic_pressure_reason": meta.get("dynamic_pressure_reason"),
        "dormant": bool(meta.get("dormant", False)),
        "dormancy_updated_at": meta.get("dormancy_updated_at"),
        "prediction_error": _safe_float(meta.get("prediction_error"), default=None),
        "contributing_turn_ids_json": _json_list(meta.get("contributing_turn_ids")),
    }


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
        evidence_refs=[str(item) for item in _parse_json_list(row.get("evidence_refs_json"), field="evidence_refs_json")],
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


def _dynamics_metadata_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    # dynamic_pressure/dormant always come back as a real scalar -- dynamics.py
    # and pressure.py already tolerate a present-but-default value via
    # `.get(key) or default`/`.get(key, False)`, so there is no ambiguity in
    # always including them. prediction_error/dynamic_pressure_reason/
    # dormancy_updated_at are omitted entirely when absent so this codec keeps
    # "no prediction error yet" (key absent) distinguishable from "computed
    # prediction error of exactly 0" (key present, value 0.0) at the storage
    # layer -- even though today's readers of prediction_error (pressure.py,
    # attention_broadcast.py) still read it via `.get(key) or 0.0` and so do
    # not yet observe the difference themselves. This preserves headroom for
    # a future consumer that does need the distinction, per the closed
    # DYNAMICS_METADATA_KEYS allowlist this codec owns. contributing_turn_ids
    # follows the same "omit when absent/empty" convention, decoded below.
    metadata: dict[str, Any] = {
        "dynamic_pressure": _safe_float(row.get("dynamic_pressure"), default=0.0),
        "dormant": bool(row.get("dormant") or False),
    }
    reason = row.get("dynamic_pressure_reason")
    if reason is not None:
        metadata["dynamic_pressure_reason"] = reason
    dormancy_updated_at = row.get("dormancy_updated_at")
    if dormancy_updated_at is not None:
        metadata["dormancy_updated_at"] = dormancy_updated_at
    prediction_error = _safe_float(row.get("prediction_error"), default=None)
    if prediction_error is not None:
        metadata["prediction_error"] = prediction_error
    # Fail-open, matching every other decode field in this function: a
    # malformed contributing_turn_ids_json value (corrupt JSON, wrong shape)
    # must not abort decoding the whole node -- treat it as absent rather
    # than raising, same tolerance _parse_json_list already gives callers
    # that choose to swallow the error (contrast with _provenance_from_row's
    # evidence_refs_json, which intentionally does raise -- that field is
    # load-bearing identity data; contributing_turn_ids is best-effort
    # attribution, so silent omission on corruption is the safer default).
    try:
        contributing_turn_ids = [
            str(item)
            for item in _parse_json_list(
                row.get("contributing_turn_ids_json"), field="contributing_turn_ids_json"
            )
        ]
    except ValueError:
        contributing_turn_ids = []
    if contributing_turn_ids:
        metadata["contributing_turn_ids"] = contributing_turn_ids
    return metadata


def decode_concept_node(row: Mapping[str, Any]) -> ConceptNodeV1 | None:
    if row.get("node_kind") != "concept":
        return None
    return ConceptNodeV1(
        node_id=str(row["node_id"]),
        label=str(row["label"]),
        definition=row.get("definition"),
        taxonomy_path=[str(item) for item in _parse_json_list(row.get("taxonomy_path_json"), field="taxonomy_path_json")],
        anchor_scope=row["anchor_scope"],
        subject_ref=row.get("subject_ref"),
        promotion_state=row.get("promotion_state") or "proposed",
        risk_tier=row.get("risk_tier") or "low",
        temporal=_temporal_from_row(row),
        signals=_signals_from_row(row),
        provenance=_provenance_from_row(row),
        metadata=_dynamics_metadata_from_row(row),
    )


def decode_evidence_node(row: Mapping[str, Any]) -> EvidenceNodeV1 | None:
    if row.get("node_kind") != "evidence":
        return None
    return EvidenceNodeV1(
        node_id=str(row["node_id"]),
        evidence_type=str(row["evidence_type"]),
        content_ref=str(row["content_ref"]),
        anchor_scope=row["anchor_scope"],
        subject_ref=row.get("subject_ref"),
        promotion_state=row.get("promotion_state") or "proposed",
        risk_tier=row.get("risk_tier") or "low",
        temporal=_temporal_from_row(row),
        signals=_signals_from_row(row),
        provenance=_provenance_from_row(row),
        metadata={},
    )


def decode_node(row: Mapping[str, Any]) -> BaseSubstrateNodeV1 | None:
    node = decode_concept_node(row)
    if node is not None:
        return node
    return decode_evidence_node(row)


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
