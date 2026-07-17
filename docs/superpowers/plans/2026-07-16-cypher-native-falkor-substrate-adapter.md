# Cypher-Native Falkor Substrate Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `FalkorSubstrateStore`'s durable `payload_json` model with Cypher-native labels, relationship types, and scalar properties for the Concept Atlas/substrate graph path.

**Architecture:** Keep `SubstrateGraphStore` as the producer-facing API and make the Falkor adapter responsible for translating typed substrate nodes/edges into native Cypher properties. This plan covers the first independently shippable slice only: the adapter redesign plus tests that prove Concept Atlas still works against the store contract. Drive measurement Postgres migration and substrate-runtime cutover are separate follow-on plans.

**Tech Stack:** Python 3, Pydantic v2 substrate schemas, redis-py Graph client, pytest, FalkorDB Cypher via `graph_query(cypher, params)`.

## Global Constraints

- No SPARQL smoosh into Falkor: new Falkor writes must not use `payload_json` / `payloadJson` as durable schema.
- Cypher-native model: labels, relationship types, closed scalar property allowlists, queryable without JSON parse.
- Postgres where sane: drive measurement, history, receipts, projections, and latest-row reads are not part of this adapter patch.
- Bus to `orion-sql-writer` for future Postgres writes; no direct HTTPS/API writes to sql-writer.
- Split by workload shape, not by service name.
- Do not flip `services/orion-substrate-runtime/.env` to Falkor in this plan.
- Do not migrate drive-state graph snapshots in this plan.
- Do not change `orion-rdf-writer` in this plan.
- No new dependencies.

---

## Scope Check

The design spec covers three subsystems:

1. Cypher-native Falkor substrate adapter.
2. Drive measurement Postgres-via-bus SoR and chat stance consumer migration.
3. Substrate-runtime graph-shaped writer cutover.

This plan implements only subsystem 1. It produces working, reviewable software on its own: Hub/Concept Atlas can keep using `SubstrateGraphStore`, while Falkor durable writes become Cypher-native. Subsystems 2 and 3 need their own plans after this lands.

---

## File Structure

- Modify: `orion/substrate/falkor_store.py`
  - Owns Falkor client, durable Cypher writes, hydration from durable Falkor rows, cache retagging, and env builder.
  - After this plan, it imports codec helpers instead of serializing `payload_json`.

- Create: `orion/substrate/falkor_codec.py`
  - Owns Cypher-safe label/type mapping and exact scalar property allowlists.
  - Produces `encode_node_properties()`, `encode_edge_properties()`, `decode_concept_node()`, and `decode_edge()`.
  - Does not open Redis, mutate stores, or know about `FalkorSubstrateStore`.

- Modify: `orion/substrate/tests/test_falkor_store.py`
  - Adapter-level tests: recorded Cypher, cache round-trip, hydration, env builder.

- Create: `orion/substrate/tests/test_falkor_codec.py`
  - Pure codec tests for property allowlists and reconstruction from native rows.

- Modify: `services/orion-hub/tests/test_concept_atlas_routes.py`
  - Add a store-contract regression that uses a hydrated Falkor store test double to prove Concept Atlas summary/network behavior survives the native-property adapter.

- Modify: `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`
  - Mark the adapter implementation acceptance check once the patch is complete.

---

### Task 1: Lock Adapter Regressions Against `payload_json`

**Files:**
- Modify: `orion/substrate/tests/test_falkor_store.py`

**Interfaces:**
- Consumes: `FalkorSubstrateStore`, `FalkorSubstrateStoreConfig`, `RecordingFalkorClient`, `_concept()`.
- Produces: failing tests that later tasks satisfy:
  - `test_falkor_upsert_concept_uses_native_cypher_properties()`
  - `test_falkor_hydrates_concept_from_native_properties()`

- [ ] **Step 1: Write the failing adapter tests**

Append these tests to `orion/substrate/tests/test_falkor_store.py` after `test_falkor_upsert_round_trip_via_cache()`:

```python
def test_falkor_upsert_concept_uses_native_cypher_properties():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    node = _concept(node_id="concept-native-alpha")

    store.upsert_node(identity_key="concept:alpha", node=node)

    cypher, params = client.calls[-1]
    assert "payload_json" not in cypher
    assert params is not None
    assert "payload_json" not in params
    assert "MERGE (n:SubstrateNode:Concept {node_id: $node_id})" in cypher
    assert "n.label = $label" in cypher
    assert "n.promotion_state = $promotion_state" in cypher
    assert "n.salience = $salience" in cypher
    assert params["node_id"] == "concept-native-alpha"
    assert params["node_kind"] == "concept"
    assert params["identity_key"] == "concept:alpha"
    assert params["label"] == "alpha"
    assert params["anchor_scope"] == "orion"
    assert params["promotion_state"] == "proposed"
    assert params["risk_tier"] == "low"
    assert params["salience"] == 0.0
    assert params["activation"] == 0.0
    assert params["recency_score"] == 0.0
    assert params["confidence"] == 0.5


def test_falkor_hydrates_concept_from_native_properties():
    client = RecordingFalkorClient(
        hydrate_rows=[
            {
                "node_id": "concept-hydrated",
                "node_kind": "concept",
                "identity_key": "concept:hydrated",
                "label": "Hydrated concept",
                "definition": "Loaded from Falkor native properties",
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.75,
                "salience": 0.66,
                "activation": 0.44,
                "recency_score": 0.33,
                "decay_floor": 0.1,
                "decay_half_life_seconds": None,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "valid_from": None,
                "valid_to": None,
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:falkor",
                "provenance_producer": "test_falkor_store",
                "provenance_model_name": None,
                "provenance_correlation_id": None,
                "provenance_trace_id": None,
                "provenance_tier_rank": None,
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-hydrated")
    assert node is not None
    assert node.node_kind == "concept"
    assert node.label == "Hydrated concept"
    assert node.definition == "Loaded from Falkor native properties"
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.75
    assert node.signals.salience == 0.66
    assert node.signals.activation.activation == 0.44
    assert node.signals.activation.recency_score == 0.33
    assert store.get_node_id_by_identity("concept:hydrated") == "concept-hydrated"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest orion/substrate/tests/test_falkor_store.py::test_falkor_upsert_concept_uses_native_cypher_properties \
  orion/substrate/tests/test_falkor_store.py::test_falkor_hydrates_concept_from_native_properties -q
```

Expected: FAIL because current `upsert_node()` writes `payload_json`, and hydration queries `RETURN n.payload_json`.

- [ ] **Step 3: Commit failing tests**

```bash
git add orion/substrate/tests/test_falkor_store.py
git commit -m "test: require cypher-native falkor substrate writes"
```

---

### Task 2: Add Native Falkor Codec Helpers

**Files:**
- Create: `orion/substrate/falkor_codec.py`
- Create: `orion/substrate/tests/test_falkor_codec.py`

**Interfaces:**
- Produces:
  - `node_label_for_kind(node_kind: str) -> str`
  - `encode_node_properties(node: BaseSubstrateNodeV1, identity_key: str | None) -> dict[str, Any]`
  - `encode_edge_properties(edge: SubstrateEdgeV1, identity_key: str) -> dict[str, Any]`
  - `decode_concept_node(row: Mapping[str, Any]) -> ConceptNodeV1 | None`
  - `decode_edge(row: Mapping[str, Any]) -> SubstrateEdgeV1 | None`
- Consumes: `BaseSubstrateNodeV1`, `ConceptNodeV1`, `SubstrateEdgeV1`, `SubstrateActivationV1`, `SubstrateSignalBundleV1`, `SubstrateTemporalWindowV1`, `SubstrateProvenanceV1`.

- [ ] **Step 1: Write codec tests**

Create `orion/substrate/tests/test_falkor_codec.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_codec import (
    decode_concept_node,
    decode_edge,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)


def _provenance() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="test:falkor_codec",
        producer="test_falkor_codec",
    )


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc))


def _concept() -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id="concept-alpha",
        label="Alpha",
        definition="A test concept",
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=0.7,
            activation=SubstrateActivationV1(
                activation=0.6,
                recency_score=0.5,
                decay_floor=0.1,
            ),
        ),
        metadata={"quarantine": "not durable SoR"},
    )


def test_node_label_for_kind_uses_closed_mapping():
    assert node_label_for_kind("concept") == "Concept"
    assert node_label_for_kind("state_snapshot") == "StateSnapshot"
    assert node_label_for_kind("unknown_kind") == "Generic"


def test_encode_concept_node_properties_are_native_scalars_without_payload_json():
    props = encode_node_properties(_concept(), identity_key="concept:alpha")

    assert "payload_json" not in props
    assert "metadata" not in props
    assert props == {
        "node_id": "concept-alpha",
        "node_kind": "concept",
        "identity_key": "concept:alpha",
        "anchor_scope": "orion",
        "subject_ref": None,
        "promotion_state": "canonical",
        "risk_tier": "low",
        "confidence": 0.8,
        "salience": 0.7,
        "activation": 0.6,
        "recency_score": 0.5,
        "decay_half_life_seconds": None,
        "decay_floor": 0.1,
        "observed_at": "2026-07-16T00:00:00+00:00",
        "valid_from": None,
        "valid_to": None,
        "provenance_authority": "local_inferred",
        "provenance_source_kind": "test",
        "provenance_source_channel": "test:falkor_codec",
        "provenance_producer": "test_falkor_codec",
        "provenance_model_name": None,
        "provenance_correlation_id": None,
        "provenance_trace_id": None,
        "provenance_tier_rank": None,
        "label": "Alpha",
        "definition": "A test concept",
    }


def test_decode_concept_node_reconstructs_minimal_typed_model():
    row = encode_node_properties(_concept(), identity_key="concept:alpha")

    node = decode_concept_node(row)

    assert node is not None
    assert node.node_id == "concept-alpha"
    assert node.node_kind == "concept"
    assert node.label == "Alpha"
    assert node.definition == "A test concept"
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.8
    assert node.signals.salience == 0.7
    assert node.signals.activation.activation == 0.6
    assert node.metadata == {}


def test_encode_edge_properties_are_native_scalars_without_payload_json():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(),
        metadata={"ignored": "not durable SoR"},
    )

    props = encode_edge_properties(edge, identity_key="edge:alpha-beta")

    assert "payload_json" not in props
    assert "metadata" not in props
    assert props["edge_id"] == "edge-alpha-beta"
    assert props["source_id"] == "concept-alpha"
    assert props["target_id"] == "concept-beta"
    assert props["predicate"] == "contradicts"
    assert props["identity_key"] == "edge:alpha-beta"
    assert props["confidence"] == 0.9
    assert props["salience"] == 0.4


def test_decode_edge_reconstructs_typed_edge():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(),
    )
    row = encode_edge_properties(edge, identity_key="edge:alpha-beta")

    decoded = decode_edge(row)

    assert decoded is not None
    assert decoded.edge_id == "edge-alpha-beta"
    assert decoded.source.node_id == "concept-alpha"
    assert decoded.target.node_id == "concept-beta"
    assert decoded.predicate == "contradicts"
    assert decoded.confidence == 0.9
    assert decoded.salience == 0.4
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest orion/substrate/tests/test_falkor_codec.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'orion.substrate.falkor_codec'`.

- [ ] **Step 3: Create codec implementation**

Create `orion/substrate/falkor_codec.py`:

```python
"""Cypher-native FalkorDB encoding for substrate graph records.

This module is intentionally pure: no Redis client, no store cache, no graph
queries. It owns the durable property allowlist used by FalkorSubstrateStore.
"""

from __future__ import annotations

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
    }


def encode_node_properties(node: BaseSubstrateNodeV1, identity_key: str | None) -> dict[str, Any]:
    props = _common_node_properties(node, identity_key)
    if node.node_kind == "concept":
        props["label"] = getattr(node, "label")
        props["definition"] = getattr(node, "definition", None)
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
        evidence_refs=[],
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
```

- [ ] **Step 4: Run codec tests to verify pass**

Run:

```bash
pytest orion/substrate/tests/test_falkor_codec.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit codec**

```bash
git add orion/substrate/falkor_codec.py orion/substrate/tests/test_falkor_codec.py
git commit -m "feat: add cypher-native falkor substrate codec"
```

---

### Task 3: Wire Falkor Store Writes to Native Properties

**Files:**
- Modify: `orion/substrate/falkor_store.py`
- Modify: `orion/substrate/tests/test_falkor_store.py`

**Interfaces:**
- Consumes from Task 2:
  - `node_label_for_kind(node_kind: str) -> str`
  - `encode_node_properties(node: BaseSubstrateNodeV1, identity_key: str | None) -> dict[str, Any]`
  - `encode_edge_properties(edge: SubstrateEdgeV1, identity_key: str) -> dict[str, Any]`
- Produces:
  - `_set_assignments(alias: str, params: dict[str, Any], *, skip: set[str]) -> str`
  - Native-property `upsert_node()`
  - Native-property `upsert_edge()`

- [ ] **Step 1: Update store imports**

In `orion/substrate/falkor_store.py`, replace:

```python
import json
```

with no replacement import. Then replace:

```python
from pydantic import TypeAdapter

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1, SubstrateNodeV1
```

with:

```python
from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.substrate.falkor_codec import (
    decode_concept_node,
    decode_edge,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)
```

Delete:

```python
NODE_ADAPTER = TypeAdapter(SubstrateNodeV1)
```

- [ ] **Step 2: Update `RecordingFalkorClient` hydrate trigger**

Replace the `graph_query()` method in `RecordingFalkorClient` with:

```python
    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((cypher, params))
        if "RETURN n.node_id AS node_id" in cypher or "RETURN e.edge_id AS edge_id" in cypher:
            return self._hydrate_rows
        return []
```

- [ ] **Step 3: Add deterministic SET assignment helper**

Add this helper above `class FalkorSubstrateStore`:

```python
def _set_assignments(alias: str, params: dict[str, Any], *, skip: set[str]) -> str:
    keys = sorted(k for k in params if k not in skip)
    return ", ".join(f"{alias}.{key} = ${key}" for key in keys)
```

- [ ] **Step 4: Replace `upsert_node()` with native Cypher**

Replace `FalkorSubstrateStore.upsert_node()` with:

```python
    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        node = _with_sanitized_metadata(node)
        params = encode_node_properties(node, identity_key)
        label = node_label_for_kind(str(node.node_kind))
        assignments = _set_assignments("n", params, skip={"node_id"})
        cypher = (
            f"MERGE (n:SubstrateNode:{label} {{node_id: $node_id}}) "
            f"SET {assignments}"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_node_failed node_id=%s error=%s", node.node_id, exc)
            raise
        self._cache.upsert_node(identity_key=identity_key, node=node)
```

- [ ] **Step 5: Replace `upsert_edge()` with native Cypher**

Replace `FalkorSubstrateStore.upsert_edge()` with:

```python
    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        edge = _with_sanitized_metadata(edge)
        params = encode_edge_properties(edge, identity_key)
        relationship_type = edge.predicate
        assignments = _set_assignments("e", params, skip={"edge_id", "source_id", "target_id"})
        cypher = (
            "MERGE (source:SubstrateNode {node_id: $source_id}) "
            "MERGE (target:SubstrateNode {node_id: $target_id}) "
            f"MERGE (source)-[e:`{relationship_type}` {{edge_id: $edge_id}}]->(target) "
            f"SET {assignments}"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_edge_failed edge_id=%s error=%s", edge.edge_id, exc)
            raise
        self._cache.upsert_edge(identity_key=identity_key, edge=edge)
```

- [ ] **Step 6: Update existing edge test assertion**

In `test_falkor_edge_is_persisted_as_typed_relationship()`, replace:

```python
    assert "e.substrate_edge = true" in cypher
```

with:

```python
    assert "payload_json" not in cypher
    assert "e.substrate_edge = $substrate_edge" in cypher
```

- [ ] **Step 7: Run adapter write tests**

Run:

```bash
pytest orion/substrate/tests/test_falkor_store.py::test_falkor_upsert_concept_uses_native_cypher_properties \
  orion/substrate/tests/test_falkor_store.py::test_falkor_edge_is_persisted_as_typed_relationship \
  orion/substrate/tests/test_falkor_store.py::test_falkor_sanitizes_metadata_cathedral -q
```

Expected: PASS.

- [ ] **Step 8: Commit native writes**

```bash
git add orion/substrate/falkor_store.py orion/substrate/tests/test_falkor_store.py
git commit -m "feat: write falkor substrate records as native cypher"
```

---

### Task 4: Hydrate Cache From Native Falkor Rows

**Files:**
- Modify: `orion/substrate/falkor_store.py`
- Modify: `orion/substrate/tests/test_falkor_store.py`

**Interfaces:**
- Consumes from Task 2:
  - `decode_concept_node(row: Mapping[str, Any]) -> ConceptNodeV1 | None`
  - `decode_edge(row: Mapping[str, Any]) -> SubstrateEdgeV1 | None`
- Produces:
  - `NATIVE_NODE_RETURN_FIELDS: tuple[str, ...]`
  - `NATIVE_EDGE_RETURN_FIELDS: tuple[str, ...]`
  - `_return_clause(alias: str, fields: tuple[str, ...]) -> str`
  - Native-property `_hydrate_from_durable()`

- [ ] **Step 1: Add return field constants**

Add these constants above `class RecordingFalkorClient` in `orion/substrate/falkor_store.py`:

```python
NATIVE_NODE_RETURN_FIELDS: tuple[str, ...] = (
    "node_id",
    "node_kind",
    "identity_key",
    "label",
    "definition",
    "anchor_scope",
    "subject_ref",
    "promotion_state",
    "risk_tier",
    "confidence",
    "salience",
    "activation",
    "recency_score",
    "decay_half_life_seconds",
    "decay_floor",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
)

NATIVE_EDGE_RETURN_FIELDS: tuple[str, ...] = (
    "edge_id",
    "identity_key",
    "source_id",
    "source_kind",
    "target_id",
    "target_kind",
    "predicate",
    "substrate_edge",
    "confidence",
    "salience",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
)


def _return_clause(alias: str, fields: tuple[str, ...]) -> str:
    return ", ".join(f"{alias}.{field} AS {field}" for field in fields)
```

- [ ] **Step 2: Replace hydration implementation**

Replace `_hydrate_from_durable()` with:

```python
    def _hydrate_from_durable(self) -> None:
        try:
            node_rows = self._client.graph_query(
                "MATCH (n:SubstrateNode) RETURN " + _return_clause("n", NATIVE_NODE_RETURN_FIELDS)
            )
            edge_rows = self._client.graph_query(
                "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
                "WHERE e.substrate_edge = true "
                "RETURN "
                + _return_clause("e", NATIVE_EDGE_RETURN_FIELDS)
            )
        except Exception as exc:
            logger.warning("falkor_substrate_hydrate_failed error=%s", exc)
            return
        for row in _normalize_rows(node_rows):
            try:
                node = decode_concept_node(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_node_invalid")
                continue
            if node is None:
                continue
            identity = row.get("identity_key")
            self._cache.upsert_node(identity_key=str(identity) if identity else None, node=node)
        for row in _normalize_rows(edge_rows):
            try:
                edge = decode_edge(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_edge_invalid")
                continue
            if edge is None:
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            self._cache.upsert_edge(identity_key=str(identity), edge=edge)
```

- [ ] **Step 3: Update raw row normalization test**

Replace `test_normalize_rows_parses_raw_graph_query_header_and_stats()` with:

```python
def test_normalize_rows_parses_raw_graph_query_header_and_stats():
    raw = [
        [["n.node_id", 1], ["n.identity_key", 1]],
        [["concept-alpha", "concept:alpha"]],
        ["Cached execution: 0", "Query internal execution time: 0.1 milliseconds"],
    ]

    assert _normalize_rows(raw) == [
        {"node_id": "concept-alpha", "identity_key": "concept:alpha"}
    ]
```

- [ ] **Step 4: Update `_normalize_rows()` tuple fallback**

In `_normalize_rows()`, replace:

```python
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                # hydrate script may return [[payload, identity], ...]
                if len(item) >= 2:
                    out.append({"payload_json": item[0], "identity_key": item[1]})
                else:
                    out.append({"payload_json": item[0], "identity_key": ""})
```

with:

```python
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                if len(item) >= 2:
                    out.append({"node_id": item[0], "identity_key": item[1]})
                else:
                    out.append({"node_id": item[0], "identity_key": ""})
```

- [ ] **Step 5: Run hydration tests**

Run:

```bash
pytest orion/substrate/tests/test_falkor_store.py::test_falkor_hydrates_concept_from_native_properties \
  orion/substrate/tests/test_falkor_store.py::test_normalize_rows_parses_raw_graph_query_header_and_stats -q
```

Expected: PASS.

- [ ] **Step 6: Commit native hydration**

```bash
git add orion/substrate/falkor_store.py orion/substrate/tests/test_falkor_store.py
git commit -m "feat: hydrate falkor substrate cache from native properties"
```

---

### Task 5: Prove Concept Atlas Works Against Hydrated Falkor Store

**Files:**
- Modify: `services/orion-hub/tests/test_concept_atlas_routes.py`
- Modify: `orion/substrate/tests/test_falkor_store.py`

**Interfaces:**
- Consumes:
  - `FalkorSubstrateStore(cfg, client=RecordingFalkorClient(hydrate_rows=[...]), hydrate=True)`
  - Hub route resolver monkeypatch: `monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)`
- Produces:
  - Cross-service contract test that Concept Atlas summary and network can read a hydrated Falkor-backed store.

- [ ] **Step 1: Add store-level query regression**

Append this test to `orion/substrate/tests/test_falkor_store.py`:

```python
def test_falkor_hydrated_concepts_support_concept_region_query():
    client = RecordingFalkorClient(
        hydrate_rows=[
            {
                "node_id": "concept-alpha",
                "node_kind": "concept",
                "identity_key": "concept:alpha",
                "label": "Alpha",
                "definition": None,
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.8,
                "salience": 0.7,
                "activation": 0.5,
                "recency_score": 0.4,
                "decay_floor": 0.0,
                "decay_half_life_seconds": None,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "valid_from": None,
                "valid_to": None,
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:falkor",
                "provenance_producer": "test_falkor_store",
                "provenance_model_name": None,
                "provenance_correlation_id": None,
                "provenance_trace_id": None,
                "provenance_tier_rank": None,
            }
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    result = store.query_concept_region(limit_nodes=10, limit_edges=10)

    assert result.source_kind == "falkor"
    assert [node.node_id for node in result.nodes] == ["concept-alpha"]
```

- [ ] **Step 2: Add Hub Concept Atlas route regression**

Append this test to `services/orion-hub/tests/test_concept_atlas_routes.py`:

```python
def test_summary_reads_hydrated_falkor_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.falkor_store import (
        FalkorSubstrateStore,
        FalkorSubstrateStoreConfig,
        RecordingFalkorClient,
    )

    falkor_client = RecordingFalkorClient(
        hydrate_rows=[
            {
                "node_id": "concept-native-atlas",
                "node_kind": "concept",
                "identity_key": "concept:native-atlas",
                "label": "Native Atlas",
                "definition": None,
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.8,
                "salience": 0.7,
                "activation": 0.5,
                "recency_score": 0.4,
                "decay_floor": 0.0,
                "decay_half_life_seconds": None,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "valid_from": None,
                "valid_to": None,
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:concept_atlas",
                "provenance_producer": "test_concept_atlas_routes",
                "provenance_model_name": None,
                "provenance_correlation_id": None,
                "provenance_trace_id": None,
                "provenance_tier_rank": None,
            }
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=falkor_client,
        hydrate=True,
    )
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["total_concepts"] == 1
    assert body["by_promotion_state"]["canonical"] == 1
    assert body["by_anchor_scope"]["orion"] == 1
```

- [ ] **Step 3: Run cross-service contract tests**

Run:

```bash
pytest orion/substrate/tests/test_falkor_store.py::test_falkor_hydrated_concepts_support_concept_region_query \
  services/orion-hub/tests/test_concept_atlas_routes.py::test_summary_reads_hydrated_falkor_store -q
```

Expected: PASS.

- [ ] **Step 4: Commit Concept Atlas regression**

```bash
git add orion/substrate/tests/test_falkor_store.py services/orion-hub/tests/test_concept_atlas_routes.py
git commit -m "test: prove concept atlas reads native falkor substrate store"
```

---

### Task 6: Update Docs and Run Focused Gates

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`

**Interfaces:**
- Consumes: completed adapter implementation.
- Produces: updated acceptance checklist and verification evidence.

- [ ] **Step 1: Update spec acceptance checklist**

In `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`, change:

```markdown
- [ ] `FalkorSubstrateStore` writes typed Cypher properties; unit tests assert MERGE/SET shape **without** `payload_json` as SoR
```

to:

```markdown
- [x] `FalkorSubstrateStore` writes typed Cypher properties; unit tests assert MERGE/SET shape **without** `payload_json` as SoR
```

Add this note immediately under the acceptance checklist:

```markdown
**Adapter evidence:** `orion/substrate/tests/test_falkor_store.py` and
`orion/substrate/tests/test_falkor_codec.py` verify native Cypher properties,
native hydration, metadata quarantine behavior, and Hub Concept Atlas route
compatibility through a hydrated Falkor store test double.
```

- [ ] **Step 2: Run focused Python tests**

Run:

```bash
pytest orion/substrate/tests/test_falkor_codec.py \
  orion/substrate/tests/test_falkor_store.py \
  services/orion-hub/tests/test_concept_atlas_routes.py -q
```

Expected: PASS.

- [ ] **Step 3: Run repository contract checks**

Run:

```bash
python3 scripts/check_schema_registry.py
python3 scripts/check_bus_channels.py
git diff --check
```

Expected: each command exits 0.

- [ ] **Step 4: Run graphify AST update**

Run:

```bash
scripts/safe_graphify_update.sh
```

Expected: exits 0 or refuses with a node-loss warning. If it refuses, do not retry blindly; report the refusal in the PR description and leave graph files unchanged.

- [ ] **Step 5: Commit docs and graph metadata if safe**

If `scripts/safe_graphify_update.sh` changed only safe graphify outputs, include them. Do not stage `graphify-out/cache/`.

```bash
git status --short
git add docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md
git add graphify-out/graph.json graphify-out/GRAPH_REPORT.md graphify-out/manifest.json || true
git diff --cached --stat
git commit -m "docs: record cypher-native falkor adapter evidence"
```

---

### Task 7: Final Review Gate and PR Report

**Files:**
- Create: `docs/superpowers/pr-reports/2026-07-16-cypher-native-falkor-substrate-adapter-pr.md`

**Interfaces:**
- Consumes: all implementation commits and test output.
- Produces: PR-ready report using Orion template.

- [ ] **Step 1: Run final focused checks**

Run:

```bash
pytest orion/substrate/tests/test_falkor_codec.py \
  orion/substrate/tests/test_falkor_store.py \
  services/orion-hub/tests/test_concept_atlas_routes.py -q
python3 scripts/check_schema_registry.py
python3 scripts/check_bus_channels.py
git diff --check
git status --short
```

Expected: tests pass; schema/channel checks exit 0; `git diff --check` exits 0; `git status --short` shows only intentional files.

- [ ] **Step 2: Request code review subagent**

Use a code-review subagent with this prompt:

```markdown
Review the Cypher-native Falkor substrate adapter patch in this worktree.

Focus on:
- Whether `payload_json` remains a durable SoR in Falkor writes or hydration.
- Whether Cypher labels/relationship types are closed and injection-safe.
- Whether Concept Atlas still works against the `SubstrateGraphStore` contract.
- Whether the patch accidentally changes drive-state, sql-writer, runtime env, rdf-writer, or Graphiti.
- Whether tests cover native write shape, hydration, metadata quarantine, and route compatibility.

Return findings ordered by severity with exact file references and suggested fixes.
```

- [ ] **Step 3: Fix material review findings**

For each material finding, write a regression test first. Example for a remaining `payload_json` write:

```python
def test_falkor_adapter_has_no_payload_json_in_recorded_write_paths():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"),
        client=client,
        hydrate=False,
    )
    store.upsert_node(identity_key="concept:alpha", node=_concept(node_id="concept-alpha"))
    for cypher, params in client.calls:
        assert "payload_json" not in cypher
        assert not (params and "payload_json" in params)
```

Run:

```bash
pytest orion/substrate/tests/test_falkor_store.py -q
```

Expected: PASS after the fix.

- [ ] **Step 4: Write PR report**

Create `docs/superpowers/pr-reports/2026-07-16-cypher-native-falkor-substrate-adapter-pr.md`:

```markdown
# PR report: Cypher-native Falkor substrate adapter

## Summary

- Replaced Falkor substrate durable writes from `payload_json` blobs to Cypher-native node/edge properties.
- Added pure codec helpers for closed native property allowlists and typed hydration.
- Updated Falkor store hydration to rebuild concept nodes and edges from native rows.
- Added Concept Atlas route regression against a hydrated Falkor store test double.
- Left drive measurement, substrate-runtime env, rdf-writer, and Graphiti untouched.

## Outcome moved

Falkor is now a property graph for the Concept Atlas/substrate graph seam instead of RDF-style payload blobs behind Cypher syntax.

## Current architecture

Before this patch, `FalkorSubstrateStore` persisted `n.payload_json` and `e.payload_json`, then hydrated cache state by parsing JSON blobs. Hub Concept Atlas could persist across restarts, but Falkor queries could not use native properties beyond a few helper scalars.

## Architecture touched

- `orion/substrate/falkor_store.py`: durable Falkor adapter.
- `orion/substrate/falkor_codec.py`: Cypher-native encode/decode helpers.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: route contract regression.

## Files changed

- `orion/substrate/falkor_codec.py`: native property encode/decode allowlist.
- `orion/substrate/falkor_store.py`: native Cypher writes and hydration.
- `orion/substrate/tests/test_falkor_codec.py`: codec unit tests.
- `orion/substrate/tests/test_falkor_store.py`: adapter write/hydration/cache tests.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: Concept Atlas compatibility test.
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: acceptance evidence.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: Falkor substrate durable representation is native Cypher properties instead of `payload_json` SoR.
- Compatibility notes: `SubstrateGraphStore` caller API remains unchanged. SPARQL store remains legacy and unchanged. Runtime env is not flipped.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/substrate/tests/test_falkor_codec.py orion/substrate/tests/test_falkor_store.py services/orion-hub/tests/test_concept_atlas_routes.py -q
python3 scripts/check_schema_registry.py
python3 scripts/check_bus_channels.py
git diff --check
```

## Evals run

```text
No eval harness exists for the Falkor adapter seam; focused deterministic tests cover the adapter and Hub route contract.
```

## Docker/build/smoke checks

```text
No Docker smoke required for this adapter-only patch. Live Falkor restart smoke remains required before runtime cutover.
```

## Review findings fixed

- Finding:
 - Fix:
 - Evidence:

## Restart required

```text
No restart required for merged code until the affected services are redeployed. For live validation after deploy, restart orion-hub only; do not flip substrate-runtime yet.
```

## Risks / concerns

- Severity: Medium
- Concern: Hydration reconstructs the native Concept Atlas subset first; non-concept substrate node kinds remain outside the first cutover.
- Mitigation: Runtime cutover is deferred until graph-shaped runtime writers have their own tests and codec coverage.

## PR link

<link>
```

- [ ] **Step 5: Commit PR report**

```bash
git add docs/superpowers/pr-reports/2026-07-16-cypher-native-falkor-substrate-adapter-pr.md
git diff --cached --stat
git commit -m "docs: add cypher-native falkor adapter pr report"
```

---

## Plan Self-Review

Spec coverage:

- Cypher-native Falkor adapter: Tasks 1-5.
- No `payload_json` SoR: Tasks 1, 3, 4.
- Concept Atlas compatibility: Task 5.
- Postgres/bus split: Global constraints and non-scope preserved; no SQL writer work in this first plan.
- Runtime cutover deferred: Scope Check and Global Constraints.
- Drive graph snapshot not migrated: Global Constraints.

Placeholder scan:

- No placeholder markers or vague test instructions are present.
- Each code-changing task includes concrete snippets and exact commands.

Type consistency:

- `encode_node_properties()`, `encode_edge_properties()`, `decode_concept_node()`, and `decode_edge()` are defined in Task 2 and consumed by Tasks 3-4.
- `RecordingFalkorClient` remains the injectable test double used by existing tests.
- `FalkorSubstrateStore` constructor signature remains unchanged.
