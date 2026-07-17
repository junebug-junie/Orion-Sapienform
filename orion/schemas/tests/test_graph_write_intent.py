from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.graph_write_intent import (
    GRAPH_WRITE_INTENT_KIND,
    GraphWriteCompatibilityV1,
    GraphWriteEdgePayloadV1,
    GraphWriteIntentV1,
    GraphWriteNodePayloadV1,
    GraphWriteProvenanceV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY, resolve


def _provenance() -> GraphWriteProvenanceV1:
    return GraphWriteProvenanceV1(
        producer="substrate-runtime",
        source_refs=["trace-abc"],
        observed_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )


def test_graph_write_intent_happy_path() -> None:
    intent = GraphWriteIntentV1(
        workload="substrate.drive_state",
        operation="upsert_node",
        identity_key="drive:curiosity",
        node=GraphWriteNodePayloadV1(
            kind="drive",
            id="drive:curiosity",
            properties={"activation": 0.7},
        ),
        provenance=_provenance(),
        compatibility=GraphWriteCompatibilityV1(rdf_graph_name="orion/substrate"),
        routing_hint="falkor",
    )
    assert intent.workload == "substrate.drive_state"
    assert intent.operation == "upsert_node"
    assert intent.node is not None
    assert intent.node.properties == {"activation": 0.7}
    assert intent.compatibility is not None
    assert intent.compatibility.rdf_graph_name == "orion/substrate"
    assert intent.routing_hint == "falkor"

    dumped = intent.model_dump(mode="json")
    restored = GraphWriteIntentV1.model_validate(dumped)
    assert restored == intent


def test_graph_write_intent_edge_payload_happy_path() -> None:
    intent = GraphWriteIntentV1(
        workload="substrate.concept",
        operation="upsert_edge",
        identity_key="edge:supports:src:tgt",
        edge=GraphWriteEdgePayloadV1(
            predicate="supports",
            source_id="node:src",
            target_id="node:tgt",
            properties={"confidence": 0.8},
        ),
        provenance=_provenance(),
    )
    assert intent.edge is not None
    assert intent.edge.predicate == "supports"


def test_graph_write_intent_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GraphWriteIntentV1(
            workload="substrate.drive_state",
            operation="upsert_node",
            identity_key="drive:curiosity",
            provenance=_provenance(),
            unexpected_field="nope",
        )

    with pytest.raises(ValidationError):
        GraphWriteNodePayloadV1(
            kind="drive",
            id="drive:curiosity",
            properties={},
            extra="nope",
        )


def test_graph_write_intent_rejects_empty_identity_key() -> None:
    with pytest.raises(ValidationError):
        GraphWriteIntentV1(
            workload="substrate.drive_state",
            operation="delete_node",
            identity_key="",
            provenance=_provenance(),
        )


def test_graph_write_intent_registered() -> None:
    assert "GraphWriteIntentV1" in _REGISTRY
    assert resolve("GraphWriteIntentV1") is GraphWriteIntentV1
    assert SCHEMA_REGISTRY["GraphWriteIntentV1"].kind == GRAPH_WRITE_INTENT_KIND
    assert SCHEMA_REGISTRY["GraphWriteIntentV1"].model is GraphWriteIntentV1
