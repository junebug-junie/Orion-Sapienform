from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    DriveNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
    RedisGraphQueryClient,
    _normalize_rows,
    _resolve_falkor_snapshot_force_refresh_ceiling_sec,
)
from orion.substrate.graphdb_store import build_substrate_store_from_env
from orion.substrate.store import InMemorySubstrateGraphStore


def _concept(*, node_id: str = "sub-node-a", metadata: dict | None = None) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        label="alpha",
        taxonomy_path=["root"],
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
            evidence_refs=["ev:store"],
        ),
        metadata=metadata or {},
    )


def _evidence(*, node_id: str = "evidence-node-a", metadata: dict | None = None) -> EvidenceNodeV1:
    return EvidenceNodeV1(
        node_id=node_id,
        evidence_type="chat_turn",
        content_ref="ev-content-ref-store",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
            evidence_refs=["ev:store"],
        ),
        metadata=metadata or {},
    )


def _assert_no_payload_json_sor(cypher: str, params: dict | None) -> None:
    assert "$payload_json" not in cypher
    assert "payload_json =" not in cypher.replace("REMOVE n.payload_json", "").replace(
        "REMOVE e.payload_json", ""
    )
    assert params is not None
    assert "payload_json" not in params


def test_falkor_upsert_round_trip_via_cache():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )
    node = _concept()
    store.upsert_node(identity_key="id-a", node=node)
    assert store.get_node_by_id(node.node_id) is not None
    assert store.get_node_id_by_identity("id-a") == node.node_id
    assert store.get_identity_key_by_node_id(node.node_id) == "id-a"
    # Inspect the cache directly rather than via the public snapshot() --
    # snapshot() now (correctly) forces a live re-hydration on its first
    # call (see test_falkor_snapshot_* below for that behavior's own
    # coverage), which for this bare RecordingFalkorClient (no scripted
    # hydrate_node_rows) would re-fetch an empty durable state and discard
    # the node this test just upserted -- a test-double artifact, not a
    # production bug: real Falkor's MATCH query issued by that refresh
    # would correctly see the just-written node, since upsert_node's own
    # Cypher write already landed durably by this point. This assertion's
    # actual intent -- proving upsert_node's immediate cache write-through
    # -- doesn't depend on snapshot()'s refresh behavior at all.
    assert node.node_id in store._cache.snapshot().nodes
    assert any("MERGE (n:SubstrateNode" in cypher for cypher, _ in client.calls)


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
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE n.payload_json" in cypher
    assert "MERGE (n:SubstrateNode:Concept {node_id: $node_id})" in cypher
    assert "n.label = $label" in cypher
    assert "n.promotion_state = $promotion_state" in cypher
    assert "n.salience = $salience" in cypher
    assert "n.evidence_refs_json = $evidence_refs_json" in cypher
    assert "n.taxonomy_path_json = $taxonomy_path_json" in cypher
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
    assert params["evidence_refs_json"] == '["ev:store"]'
    assert params["taxonomy_path_json"] == '["root"]'


def test_falkor_upsert_concept_sets_native_dynamics_properties():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    node = _concept(
        node_id="concept-dynamics-alpha",
        metadata={
            "dynamic_pressure": 0.42,
            "dynamic_pressure_reason": "prediction_error_seed",
            "dormant": True,
            "dormancy_updated_at": "2026-07-17T00:00:00+00:00",
            "prediction_error": 0.8,
        },
    )

    store.upsert_node(identity_key="concept:dynamics-alpha", node=node)

    cypher, params = client.calls[-1]
    assert "n.dynamic_pressure = $dynamic_pressure" in cypher
    assert "n.dynamic_pressure_reason = $dynamic_pressure_reason" in cypher
    assert "n.dormant = $dormant" in cypher
    assert "n.dormancy_updated_at = $dormancy_updated_at" in cypher
    assert "n.prediction_error = $prediction_error" in cypher
    assert params["dynamic_pressure"] == 0.42
    assert params["dynamic_pressure_reason"] == "prediction_error_seed"
    assert params["dormant"] is True
    assert params["dormancy_updated_at"] == "2026-07-17T00:00:00+00:00"
    assert params["prediction_error"] == 0.8


def test_falkor_hydrates_dynamics_properties_into_metadata():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-dynamics-hydrated",
                "node_kind": "concept",
                "identity_key": "concept:dynamics-hydrated",
                "label": "Hydrated dynamics concept",
                "definition": None,
                "taxonomy_path_json": "[]",
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
                "evidence_refs_json": "[]",
                "dynamic_pressure": 0.55,
                "dynamic_pressure_reason": "drive_seed",
                "dormant": True,
                "dormancy_updated_at": "2026-07-17T01:00:00+00:00",
                "prediction_error": 0.9,
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-dynamics-hydrated")
    assert node is not None
    assert node.metadata.get("dynamic_pressure") == 0.55
    assert node.metadata.get("dynamic_pressure_reason") == "drive_seed"
    assert node.metadata.get("dormant") is True
    assert node.metadata.get("dormancy_updated_at") == "2026-07-17T01:00:00+00:00"
    assert node.metadata.get("prediction_error") == 0.9


def test_falkor_rejects_non_concept_durable_write():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    drive = DriveNodeV1(
        node_id="drive-curiosity",
        drive_kind="curiosity",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
        ),
    )

    with pytest.raises(ValueError, match="concept and evidence nodes only"):
        store.upsert_node(identity_key="drive:curiosity", node=drive)

    assert client.calls == []


def test_falkor_upsert_evidence_uses_native_cypher_properties():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    node = _evidence(node_id="evidence-native-alpha")

    store.upsert_node(identity_key="evidence:alpha", node=node)

    cypher, params = client.calls[-1]
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE n.payload_json" in cypher
    assert "MERGE (n:SubstrateNode:Evidence {node_id: $node_id})" in cypher
    assert "n.evidence_type = $evidence_type" in cypher
    assert "n.content_ref = $content_ref" in cypher
    assert params["node_id"] == "evidence-native-alpha"
    assert params["node_kind"] == "evidence"
    assert params["identity_key"] == "evidence:alpha"
    assert params["evidence_type"] == "chat_turn"
    assert params["content_ref"] == "ev-content-ref-store"
    assert "label" not in params
    assert "definition" not in params
    assert "taxonomy_path_json" not in params


def test_falkor_hydrates_concept_and_evidence_from_mixed_native_rows():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-mixed",
                "node_kind": "concept",
                "identity_key": "concept:mixed",
                "label": "Mixed concept",
                "definition": None,
                "taxonomy_path_json": "[]",
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.7,
                "salience": 0.6,
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
                "evidence_refs_json": "[]",
            },
            {
                "node_id": "evidence-mixed",
                "node_kind": "evidence",
                "identity_key": "evidence:mixed",
                "evidence_type": "chat_turn",
                "content_ref": "ev-content-ref-mixed",
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "proposed",
                "risk_tier": "low",
                "confidence": 0.65,
                "salience": 0.55,
                "activation": 0.45,
                "recency_score": 0.35,
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
                "evidence_refs_json": '["ev:mixed"]',
            },
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    concept_node = store.get_node_by_id("concept-mixed")
    assert concept_node is not None
    assert concept_node.node_kind == "concept"
    assert concept_node.label == "Mixed concept"
    assert store.get_node_id_by_identity("concept:mixed") == "concept-mixed"

    evidence_node = store.get_node_by_id("evidence-mixed")
    assert evidence_node is not None
    assert evidence_node.node_kind == "evidence"
    assert evidence_node.evidence_type == "chat_turn"
    assert evidence_node.content_ref == "ev-content-ref-mixed"
    assert evidence_node.provenance.evidence_refs == ["ev:mixed"]
    assert store.get_node_id_by_identity("evidence:mixed") == "evidence-mixed"


def test_falkor_hydrates_concept_from_native_properties():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-hydrated",
                "node_kind": "concept",
                "identity_key": "concept:hydrated",
                "label": "Hydrated concept",
                "definition": "Loaded from Falkor native properties",
                "taxonomy_path_json": '["atlas"]',
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
                "evidence_refs_json": '["ev:hydrate"]',
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
    assert node.taxonomy_path == ["atlas"]
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.75
    assert node.signals.salience == 0.66
    assert node.signals.activation.activation == 0.44
    assert node.signals.activation.recency_score == 0.33
    assert node.provenance.evidence_refs == ["ev:hydrate"]
    assert store.get_node_id_by_identity("concept:hydrated") == "concept-hydrated"


def test_falkor_hydrates_legacy_payload_json_and_rewrites_native():
    legacy_node = ConceptNodeV1(
        node_id="concept-legacy",
        label="Legacy concept",
        definition="From blob SoR",
        taxonomy_path=["legacy"],
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=SubstrateTemporalWindowV1(
            observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:legacy",
            producer="test_falkor_store",
            evidence_refs=["ev:legacy"],
        ),
    )
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[
            {
                "payload_json": legacy_node.model_dump_json(),
                "identity_key": "concept:legacy",
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-legacy")
    assert node is not None
    assert node.label == "Legacy concept"
    assert node.taxonomy_path == ["legacy"]
    assert node.provenance.evidence_refs == ["ev:legacy"]
    assert store.get_node_id_by_identity("concept:legacy") == "concept-legacy"

    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls, "expected native rewrite after legacy hydrate"
    cypher, params = write_calls[-1]
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE n.payload_json" in cypher
    assert params["label"] == "Legacy concept"
    assert params["evidence_refs_json"] == '["ev:legacy"]'


def test_falkor_legacy_migration_deletes_orphaned_duplicate_row():
    # Regression (2026-07-18 incident): upsert_node()'s MERGE keys on
    # (SubstrateNode:<type-label> {node_id}), a pattern the un-migrated
    # legacy row (SubstrateNode only, no type label) can never itself
    # satisfy -- so the rewrite always lands on a *different* node, leaving
    # this legacy row (and its stale payload_json) permanently orphaned.
    # Confirmed live: the orphaned row got re-parsed and re-clobbered the
    # real node's data on every subsequent hydrate, silently reverting
    # PR #1173's golden-concept salience fix within one restart cycle, and
    # cascaded into duplicate edges via the same `:SubstrateNode`-only
    # ambiguity on edge MERGE's source/target match. The fix issues an
    # explicit cleanup DELETE for the orphaned row after a successful
    # migration; this test asserts that cleanup query is actually issued.
    #
    # Coverage limit, stated explicitly: RecordingFalkorClient is a scripted
    # test double that dispatches on cypher substrings -- it never parses or
    # executes Cypher, so this only proves the *right query text* is issued,
    # not that DETACH DELETE actually cascades relationship removal the way
    # this fix's design depends on. That MERGE-label-matching and
    # DETACH-DELETE-cascade behavior was verified directly against
    # FalkorDB's own docs (docs.falkordb.com/cypher/delete.html: "DETACH
    # DELETE automatically removes all relationships before deleting the
    # node") and against the live orion-athena-falkordb container during
    # this incident's investigation, not by this test suite. A future
    # FalkorDB dialect change to either behavior would not fail here.
    legacy_node = ConceptNodeV1(
        node_id="concept-legacy",
        label="Legacy concept",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:legacy",
            producer="test_falkor_store",
        ),
    )
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[
            {"payload_json": legacy_node.model_dump_json(), "identity_key": "concept:legacy"}
        ]
    )

    FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    cleanup_calls = [
        (cypher, params)
        for cypher, params in client.calls
        if "DETACH DELETE" in cypher and "payload_json IS NOT NULL" in cypher
    ]
    assert cleanup_calls, "expected an orphaned-duplicate cleanup query after migration"
    cypher, params = cleanup_calls[-1]
    assert params == {"node_id": "concept-legacy"}


def test_falkor_legacy_migration_does_not_delete_a_row_without_a_duplicate():
    # The cleanup query only matches rows that STILL carry payload_json.
    # upsert_node()'s own SET clause already removes payload_json from
    # whatever node the migration write landed on, so in the common case
    # (no lingering duplicate) the cleanup query naturally matches nothing --
    # this is exercised implicitly by every other passing hydrate test, this
    # test just names that expectation explicitly via the recorded query
    # shape (a targeted match-by-payload_json, never a bare node_id match).
    #
    # Same coverage limit as the test above: this only proves the query's
    # literal text targets payload_json specifically (not a blanket
    # node_id match that could delete a real node) -- it does not execute
    # against a real graph engine to observe zero rows actually being
    # matched/deleted. See the comment above for what was independently
    # verified and where.
    legacy_node = ConceptNodeV1(
        node_id="concept-solo",
        label="Solo concept",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:legacy",
            producer="test_falkor_store",
        ),
    )
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[
            {"payload_json": legacy_node.model_dump_json(), "identity_key": "concept:solo"}
        ]
    )

    FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    cleanup_calls = [
        (cypher, params) for cypher, params in client.calls if "DETACH DELETE" in cypher
    ]
    assert len(cleanup_calls) == 1
    cypher, _ = cleanup_calls[0]
    assert "WHERE n.payload_json IS NOT NULL" in cypher


def test_falkor_legacy_migration_now_includes_evidence_nodes():
    # Regression: this migration path used to skip node_kind="evidence"
    # entirely (logged as falkor_substrate_legacy_node_skipped, forever, on
    # every hydrate) even though upsert_node() itself has always supported
    # both "concept" and "evidence". Confirmed live: 2 of the 7 orphaned
    # legacy rows found in the 2026-07-18 incident were evidence nodes,
    # permanently un-migrated by this now-fixed gap.
    legacy_evidence = _evidence(node_id="evidence-legacy")
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[
            {"payload_json": legacy_evidence.model_dump_json(), "identity_key": "evidence:legacy"}
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("evidence-legacy")
    assert node is not None
    assert node.node_kind == "evidence"
    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls, "expected native rewrite for the legacy evidence node"


def test_falkor_legacy_migration_still_skips_unsupported_node_kinds():
    contradiction_json = (
        '{"node_id": "contra-legacy", "node_kind": "contradiction", "anchor_scope": "orion", '
        '"temporal": {"observed_at": "2026-07-16T00:00:00+00:00"}, '
        '"provenance": {"authority": "local_inferred", "source_kind": "test", '
        '"source_channel": "test:legacy", "producer": "test_falkor_store"}, '
        '"summary": "conflict", "involved_node_ids": ["a", "b"]}'
    )
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[{"payload_json": contradiction_json, "identity_key": "contra:legacy"}]
    )

    FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls == []


def test_falkor_edge_hydrate_query_derives_source_target_from_matched_nodes():
    # Regression (2026-07-18 incident): the edge hydration RETURN clause
    # used to read e.source_id/e.target_id as edge *properties* -- but
    # upsert_edge() deliberately never writes those two fields onto the
    # edge (skip={"edge_id", "source_id", "target_id"}), since the real
    # linkage lives in the graph topology the MATCH pattern itself already
    # encodes. Reading them as properties always returned NULL, which
    # decode_edge() then str()-coerced into the literal string "None" for
    # every hydrated edge -- confirmed live: every edge in the running
    # graph silently lost its real source/target node_id on every hydrate.
    # This test asserts the query text itself, since RecordingFalkorClient
    # is a scripted test double that can't execute real Cypher to prove
    # the derivation actually happens end-to-end (see the round-trip test
    # below for what a fixed query's *output* should decode to).
    client = RecordingFalkorClient()
    FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    edge_hydrate_calls = [
        cypher for cypher, _ in client.calls if "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode)" in cypher
    ]
    assert edge_hydrate_calls, "expected the edge hydration query to run"
    cypher = edge_hydrate_calls[0]
    assert "source.node_id AS source_id" in cypher
    assert "target.node_id AS target_id" in cypher
    assert "e.source_id AS source_id" not in cypher
    assert "e.target_id AS target_id" not in cypher


def test_falkor_hydrates_edge_source_target_node_ids_correctly():
    # Complements the query-text test above: given a hydrate row shaped the
    # way the *fixed* query actually returns results (source_id/target_id
    # populated with real node_id strings, since they're now derived from
    # the matched source/target nodes rather than absent edge properties),
    # decode_edge() must produce real NodeRefV1.node_id values -- not the
    # literal string "None" that a missing/NULL source_id previously
    # produced via decode_edge()'s str(row["source_id"]) coercion.
    client = RecordingFalkorClient(
        hydrate_edge_rows=[
            {
                "edge_id": "sub-edge-a-b",
                "identity_key": "a|supports|b",
                "source_id": "sub-concept-a",
                "source_kind": "concept",
                "target_id": "sub-concept-b",
                "target_kind": "concept",
                "predicate": "supports",
                "substrate_edge": True,
                "confidence": 0.8,
                "salience": 0.6,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:edge",
                "provenance_producer": "test_falkor_store",
                "evidence_refs_json": "[]",
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    edge = store.get_edge_by_id("sub-edge-a-b")
    assert edge is not None
    assert edge.source.node_id == "sub-concept-a"
    assert edge.target.node_id == "sub-concept-b"
    assert edge.source.node_id != "None"
    assert edge.target.node_id != "None"


def test_recording_client_splits_node_and_edge_hydrate_rows():
    client = RecordingFalkorClient(
        hydrate_node_rows=[{"node_id": "n1", "node_kind": "concept"}],
        hydrate_edge_rows=[{"edge_id": "e1", "predicate": "supports"}],
    )

    node_rows = client.graph_query(
        "MATCH (n:SubstrateNode) RETURN n.node_id AS node_id, n.node_kind AS node_kind"
    )
    edge_rows = client.graph_query(
        "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
        "WHERE e.substrate_edge = true "
        "RETURN e.edge_id AS edge_id"
    )

    assert node_rows == [{"node_id": "n1", "node_kind": "concept"}]
    assert edge_rows == [{"edge_id": "e1", "predicate": "supports"}]


def test_falkor_sanitizes_metadata_cathedral():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"),
        client=client,
        hydrate=False,
    )
    meta = {f"k{i}": i for i in range(20)}
    node = _concept(metadata=meta)
    store.upsert_node(identity_key="id-b", node=node)
    stored = store.get_node_by_id(node.node_id)
    assert stored is not None
    assert len(stored.metadata) <= 16


def test_falkor_edge_is_persisted_as_typed_relationship():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"),
        client=client,
        hydrate=False,
    )
    edge = SubstrateEdgeV1(
        edge_id="sub-edge-a",
        source=NodeRefV1(node_id="sub-node-a", node_kind="concept"),
        target=NodeRefV1(node_id="sub-node-b", node_kind="concept"),
        predicate="supports",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
        ),
    )

    store.upsert_edge(identity_key="edge-a", edge=edge)

    cypher, params = client.calls[-1]
    assert "MERGE (source)-[e:`supports`" in cypher
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE e.payload_json" in cypher
    assert "e.substrate_edge = $substrate_edge" in cypher
    assert params["source_id"] == "sub-node-a"
    assert params["target_id"] == "sub-node-b"


def test_build_substrate_store_falkor(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.setenv("FALKORDB_URI", "redis://orion-athena-falkordb:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")

    # Avoid opening a real redis connection during construction hydrate.
    from orion.substrate import falkor_store as fs

    fake = RecordingFalkorClient()

    class _NoRedis(FalkorSubstrateStore):
        def __init__(self, cfg, *, client=None, hydrate=True):
            super().__init__(cfg, client=client or fake, hydrate=False)

    monkeypatch.setattr(fs, "FalkorSubstrateStore", _NoRedis)
    store = build_substrate_store_from_env()
    assert isinstance(store, FalkorSubstrateStore)


def test_build_substrate_store_falkor_missing_uri(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    store = build_substrate_store_from_env()
    assert isinstance(store, InMemorySubstrateGraphStore)


def test_redis_graph_client_returns_named_dicts_from_header():
    class _Result:
        header = [[1, "node_id"], [1, "identity_key"]]
        result_set = [["concept-alpha", "concept:alpha"]]

    class _Graph:
        def __init__(self):
            self.calls = []

        def query(self, cypher, params=None):
            self.calls.append((cypher, params))
            return _Result()

    graph = _Graph()
    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    client._graph = graph

    result = client.graph_query("RETURN $node_id", {"node_id": "sub-node-a"})

    assert result == [{"node_id": "concept-alpha", "identity_key": "concept:alpha"}]
    assert graph.calls == [("RETURN $node_id", {"node_id": "sub-node-a"})]


def test_normalize_rows_parses_raw_graph_query_header_and_stats():
    raw = [
        [["n.node_id", 1], ["n.identity_key", 1]],
        [["concept-alpha", "concept:alpha"]],
        ["Cached execution: 0", "Query internal execution time: 0.1 milliseconds"],
    ]

    assert _normalize_rows(raw) == [
        {"node_id": "concept-alpha", "identity_key": "concept:alpha"}
    ]


def test_normalize_rows_zips_redis_py_list_rows_to_native_fields():
    from orion.substrate.falkor_store import NATIVE_NODE_RETURN_FIELDS

    values = []
    for field in NATIVE_NODE_RETURN_FIELDS:
        if field == "node_id":
            values.append("concept-list")
        elif field == "node_kind":
            values.append("concept")
        elif field == "identity_key":
            values.append("concept:list")
        elif field == "label":
            values.append("List Row")
        elif field == "definition":
            values.append("From redis-py lists")
        elif field == "taxonomy_path_json":
            values.append('["list"]')
        elif field == "anchor_scope":
            values.append("orion")
        elif field == "promotion_state":
            values.append("canonical")
        elif field == "risk_tier":
            values.append("low")
        elif field in {"confidence", "salience", "activation", "recency_score", "decay_floor"}:
            values.append(0.5)
        elif field == "observed_at":
            values.append("2026-07-16T00:00:00+00:00")
        elif field == "provenance_authority":
            values.append("local_inferred")
        elif field == "provenance_source_kind":
            values.append("test")
        elif field == "provenance_source_channel":
            values.append("test:list")
        elif field == "provenance_producer":
            values.append("test_falkor_store")
        elif field == "evidence_refs_json":
            values.append('["ev:list"]')
        else:
            values.append(None)

    rows = _normalize_rows([values], fields=NATIVE_NODE_RETURN_FIELDS)
    assert rows[0]["node_id"] == "concept-list"
    assert rows[0]["label"] == "List Row"
    assert rows[0]["evidence_refs_json"] == '["ev:list"]'


def test_falkor_hydrates_from_redis_py_result_set_lists():
    from orion.substrate.falkor_store import NATIVE_NODE_RETURN_FIELDS

    values = []
    for field in NATIVE_NODE_RETURN_FIELDS:
        defaults = {
            "node_id": "concept-redis-py",
            "node_kind": "concept",
            "identity_key": "concept:redis-py",
            "label": "Redis Py",
            "definition": "list hydrate",
            "taxonomy_path_json": '["r"]',
            "anchor_scope": "orion",
            "promotion_state": "canonical",
            "risk_tier": "low",
            "confidence": 0.8,
            "salience": 0.7,
            "activation": 0.5,
            "recency_score": 0.4,
            "decay_floor": 0.0,
            "observed_at": "2026-07-16T00:00:00+00:00",
            "provenance_authority": "local_inferred",
            "provenance_source_kind": "test",
            "provenance_source_channel": "test:redis-py",
            "provenance_producer": "test_falkor_store",
            "evidence_refs_json": '["ev:r"]',
        }
        values.append(defaults.get(field))

    legacy = ConceptNodeV1(
        node_id="concept-legacy-list",
        label="Legacy list",
        taxonomy_path=["legacy"],
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=SubstrateTemporalWindowV1(
            observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:legacy-list",
            producer="test_falkor_store",
            evidence_refs=["ev:legacy-list"],
        ),
    )

    class ListRowClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | None]] = []

        def graph_query(self, cypher: str, params: dict | None = None):
            self.calls.append((cypher, params))
            if "WHERE n.payload_json IS NOT NULL" in cypher:
                return [[legacy.model_dump_json(), "concept:legacy-list"]]
            if "WHERE e.payload_json IS NOT NULL" in cypher:
                return []
            if "RETURN n.node_id AS node_id" in cypher:
                return [values]
            if "RETURN e.edge_id AS edge_id" in cypher:
                return []
            return []

    client = ListRowClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    native = store.get_node_by_id("concept-redis-py")
    assert native is not None
    assert native.label == "Redis Py"
    assert native.provenance.evidence_refs == ["ev:r"]

    migrated = store.get_node_by_id("concept-legacy-list")
    assert migrated is not None
    assert migrated.label == "Legacy list"
    assert any("MERGE (n:SubstrateNode" in c for c, _ in client.calls)


def test_falkor_hydrates_edge_from_redis_py_result_set_lists():
    # Edge-path analog of test_falkor_hydrates_from_redis_py_result_set_lists
    # above. That existing test forces _normalize_rows()'s real positional
    # (list/tuple) zip path to run for *node* hydration -- a dict-shaped
    # scripted row (as used by test_falkor_hydrates_edge_source_target_node_ids_correctly)
    # bypasses that zip entirely (dict items pass through _normalize_rows
    # near-verbatim), so it can't catch a column-order mismatch between
    # NATIVE_EDGE_RETURN_FIELDS and _edge_hydrate_return_clause()'s emitted
    # column order. This test closes that gap for edges specifically: a
    # raw positional list, in NATIVE_EDGE_RETURN_FIELDS order, forces the
    # real zip and proves source_id/target_id land in the right positions.
    from orion.substrate.falkor_store import NATIVE_EDGE_RETURN_FIELDS

    values = []
    for field in NATIVE_EDGE_RETURN_FIELDS:
        defaults = {
            "edge_id": "sub-edge-redis-py",
            "identity_key": "a|supports|b",
            "source_id": "sub-concept-redis-py-a",
            "source_kind": "concept",
            "target_id": "sub-concept-redis-py-b",
            "target_kind": "concept",
            "predicate": "supports",
            "substrate_edge": True,
            "confidence": 0.8,
            "salience": 0.6,
            "observed_at": "2026-07-16T00:00:00+00:00",
            "provenance_authority": "local_inferred",
            "provenance_source_kind": "test",
            "provenance_source_channel": "test:edge-redis-py",
            "provenance_producer": "test_falkor_store",
            "evidence_refs_json": "[]",
        }
        values.append(defaults.get(field))

    class EdgeListRowClient:
        def graph_query(self, cypher: str, params: dict | None = None):
            if "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode)" in cypher:
                return [values]
            return []

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=EdgeListRowClient(),
        hydrate=True,
    )

    edge = store.get_edge_by_id("sub-edge-redis-py")
    assert edge is not None
    assert edge.source.node_id == "sub-concept-redis-py-a"
    assert edge.target.node_id == "sub-concept-redis-py-b"
    assert edge.predicate == "supports"


def test_falkor_legacy_migrate_keeps_cache_when_rewrite_fails():
    legacy = ConceptNodeV1(
        node_id="concept-cache-seed",
        label="Cache seed",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(
            observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:cache-seed",
            producer="test_falkor_store",
        ),
    )

    class FailRewriteClient(RecordingFalkorClient):
        def graph_query(self, cypher: str, params: dict | None = None):
            if "MERGE (n:SubstrateNode" in cypher:
                raise RuntimeError("falkor unavailable")
            return super().graph_query(cypher, params)

    client = FailRewriteClient(
        hydrate_legacy_node_rows=[
            {"payload_json": legacy.model_dump_json(), "identity_key": "concept:cache-seed"}
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-cache-seed")
    assert node is not None
    assert node.label == "Cache seed"


def test_falkor_legacy_migration_cleanup_failure_logs_as_migrate_failed_not_migrated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression: _delete_orphaned_legacy_node_duplicate() used to swallow
    # its own exceptions and log a distinct "..._cleanup_failed" line, but
    # _migrate_legacy_payload_nodes() would still unconditionally log
    # "..._migrated" (success) right after, since the cleanup call never
    # raised. A failed cleanup reproduces exactly the bug this whole patch
    # exists to fix (the orphaned row survives to re-clobber the canonical
    # node on the next hydrate) -- it must not be indistinguishable from a
    # real success in the logs. Fixed by letting the cleanup exception
    # propagate into the existing outer except block instead.
    legacy = ConceptNodeV1(
        node_id="concept-cleanup-fail",
        label="Cleanup fail",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:cleanup-fail",
            producer="test_falkor_store",
        ),
    )

    class FailCleanupClient(RecordingFalkorClient):
        def graph_query(self, cypher: str, params: dict | None = None):
            if "DETACH DELETE" in cypher:
                raise RuntimeError("falkor unavailable during cleanup")
            return super().graph_query(cypher, params)

    client = FailCleanupClient(
        hydrate_legacy_node_rows=[
            {"payload_json": legacy.model_dump_json(), "identity_key": "concept:cleanup-fail"}
        ]
    )

    with caplog.at_level("INFO", logger="orion.substrate.falkor_store"):
        FalkorSubstrateStore(
            FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
            client=client,
            hydrate=True,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("falkor_substrate_legacy_node_migrate_failed" in m for m in messages)
    assert not any("falkor_substrate_legacy_node_migrated" in m for m in messages)


def test_falkor_hydrated_concepts_support_concept_region_query():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-alpha",
                "node_kind": "concept",
                "identity_key": "concept:alpha",
                "label": "Alpha",
                "definition": None,
                "taxonomy_path_json": "[]",
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
                "evidence_refs_json": "[]",
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
    assert [node.node_id for node in result.slice.nodes] == ["concept-alpha"]


# --- snapshot() refresh/ceiling behavior --------------------------------------
#
# Mirrors GraphDBSubstrateStore's already-proven write-generation + ceiling
# mechanism (test_graphdb_store.py's own test_snapshot_* suite) -- this is the
# fix for a real, live-confirmed bug: FalkorSubstrateStore.snapshot() used to
# always return self._cache.snapshot() with no refresh at all, so (a) a node
# deleted directly from Falkor (bypassing this process, e.g. an operator
# running Cypher DELETE by hand) stayed resurrected in the cache forever, and
# (b) the decay scheduler (services/orion-hub/scripts/api_routes.py::
# decay_concept_activations) durably re-upserts every node in every snapshot()
# it reads on every tick -- so a stale cache didn't just show old data, it
# actively wrote deleted data back into Falkor on the next tick, undoing the
# deletion. Confirmed live in production before this fix.
#
# RecordingFalkorClient returns the SAME scripted rows on every graph_query()
# call (no dynamic state tracking) -- these tests script hydrate_node_rows
# once, then mutate client._hydrate_node_rows directly between snapshot()
# calls to simulate durable state changing out from under this process
# (exactly what a direct external Cypher DELETE looks like from Falkor's
# side).


def _hydrated_node_row(node_id: str, identity_key: str) -> dict:
    return {
        "node_id": node_id,
        "node_kind": "concept",
        "identity_key": identity_key,
        "label": "Hydrated",
        "definition": None,
        "taxonomy_path_json": "[]",
        "anchor_scope": "orion",
        "subject_ref": None,
        "promotion_state": "canonical",
        "risk_tier": "low",
        "confidence": 0.7,
        "salience": 0.5,
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
        "evidence_refs_json": "[]",
    }


def test_falkor_snapshot_same_generation_reuses_cache_no_new_query():
    """Baseline: nothing written since the first fetch -- the second call is
    served from cache with zero new live queries."""
    client = RecordingFalkorClient(hydrate_node_rows=[_hydrated_node_row("concept-a", "concept:a")])
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri="redis://localhost:6379", graph_name="orion_substrate", snapshot_force_refresh_ceiling_sec=60.0
        ),
        client=client,
        hydrate=False,
    )
    client.calls.clear()

    first = store.snapshot()
    calls_after_first = len(client.calls)
    assert calls_after_first == 4  # node query, edge query, 2 legacy queries
    assert "concept-a" in first.nodes

    second = store.snapshot()
    assert len(client.calls) == calls_after_first  # no new live query
    assert second.nodes.keys() == first.nodes.keys()


def test_falkor_snapshot_write_between_calls_forces_new_query():
    """A write bumps the generation counter, so the very next snapshot() call
    issues a live query even though it's well within the ceiling."""
    client = RecordingFalkorClient(hydrate_node_rows=[_hydrated_node_row("concept-a", "concept:a")])
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri="redis://localhost:6379", graph_name="orion_substrate", snapshot_force_refresh_ceiling_sec=60.0
        ),
        client=client,
        hydrate=False,
    )
    client.calls.clear()

    store.snapshot()
    calls_after_first = len(client.calls)

    store.upsert_node(identity_key="concept:b", node=_concept(node_id="concept-b"))
    calls_after_upsert = len(client.calls)
    # RecordingFalkorClient returns the SAME static scripted rows on every
    # graph_query() call -- it doesn't track the upsert_node() write that
    # just happened the way real Falkor durably would. Update the script to
    # match what a real MATCH query would now return, so the refresh this
    # test is actually checking for reflects reality rather than the fake's
    # limitation.
    client._hydrate_node_rows = [
        _hydrated_node_row("concept-a", "concept:a"),
        _hydrated_node_row("concept-b", "concept:b"),
    ]

    result = store.snapshot()
    assert len(client.calls) > calls_after_upsert  # forced a live re-hydration
    assert "concept-a" in result.nodes  # still present, from the scripted rows
    assert "concept-b" in result.nodes  # the just-upserted node, durably reflected


def test_falkor_snapshot_ceiling_forces_refresh_despite_same_generation():
    """The safety net: even with no writes at all (same_generation stays true
    indefinitely), the ceiling forces a periodic live re-check to catch
    changes made by a different process, or a direct external mutation, that
    this instance's own write counter can't see."""
    import time

    client = RecordingFalkorClient(hydrate_node_rows=[_hydrated_node_row("concept-a", "concept:a")])
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri="redis://localhost:6379", graph_name="orion_substrate", snapshot_force_refresh_ceiling_sec=0.01
        ),
        client=client,
        hydrate=False,
    )
    client.calls.clear()

    store.snapshot()
    calls_after_first = len(client.calls)

    time.sleep(0.02)
    store.snapshot()
    assert len(client.calls) > calls_after_first  # ceiling forced a live query


def test_falkor_snapshot_ceiling_zero_trusts_generation_forever():
    """ceiling <= 0 disables the periodic safety-net refresh entirely -- the
    cache is trusted for as long as the write generation hasn't moved."""
    import time

    client = RecordingFalkorClient(hydrate_node_rows=[_hydrated_node_row("concept-a", "concept:a")])
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri="redis://localhost:6379", graph_name="orion_substrate", snapshot_force_refresh_ceiling_sec=0.0
        ),
        client=client,
        hydrate=False,
    )
    client.calls.clear()

    store.snapshot()
    calls_after_first = len(client.calls)

    time.sleep(0.02)
    store.snapshot()
    assert len(client.calls) == calls_after_first  # still cached, no ceiling to force a refresh


def test_falkor_snapshot_refresh_removes_externally_deleted_node():
    """The actual regression this fix closes: a node deleted directly from
    Falkor (bypassing this process entirely -- simulated here by mutating the
    scripted hydrate rows out from under an already-hydrated store) must
    disappear from the cache on the next forced refresh, not stay
    resurrected forever."""
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            _hydrated_node_row("concept-a", "concept:a"),
            _hydrated_node_row("concept-junk", "concept:junk"),
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri="redis://localhost:6379", graph_name="orion_substrate", snapshot_force_refresh_ceiling_sec=0.01
        ),
        client=client,
        hydrate=True,
    )

    first = store.snapshot()
    assert "concept-a" in first.nodes
    assert "concept-junk" in first.nodes

    # Simulate an external `MATCH (n) WHERE ... DETACH DELETE n` against
    # Falkor directly -- "concept-junk" is gone from what the durable graph
    # now returns, but this process's cache still has it until a refresh.
    client._hydrate_node_rows = [_hydrated_node_row("concept-a", "concept:a")]

    import time

    time.sleep(0.02)  # clear the ceiling
    second = store.snapshot()
    assert "concept-a" in second.nodes
    assert "concept-junk" not in second.nodes  # the fix: no longer resurrected


def test_falkor_snapshot_failure_fallback_still_returns_cache():
    """A failed refresh (e.g. Falkor unreachable) must degrade to the
    existing cache, not an empty result or a raised exception."""

    class _FailingClient:
        def __init__(self) -> None:
            self.calls = 0

        def graph_query(self, cypher, params=None):
            self.calls += 1
            raise RuntimeError("falkor unreachable")

    failing_client = _FailingClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=failing_client,
        hydrate=False,
    )
    # Seed the cache directly (bypassing the failing client) to prove a
    # subsequent failed refresh preserves it rather than clearing it.
    store._cache.upsert_node(identity_key="concept:seed", node=_concept(node_id="concept-seeded"))

    result = store.snapshot()
    assert "concept-seeded" in result.nodes
    assert failing_client.calls > 0  # the refresh really was attempted


def test_falkor_write_generation_bumps_on_upsert_node_and_edge():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    assert store._write_generation == 0

    store.upsert_node(identity_key="concept:a", node=_concept(node_id="concept-a"))
    assert store._write_generation == 1

    edge = SubstrateEdgeV1(
        source=NodeRefV1(node_id="concept-a", node_kind="concept"),
        target=NodeRefV1(node_id="concept-b", node_kind="concept"),
        predicate="associated_with",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="test", producer="test_falkor_store"
        ),
    )
    store.upsert_edge(identity_key="a|associated_with|b", edge=edge)
    assert store._write_generation == 2


def test_resolve_falkor_snapshot_ceiling_uses_falkor_specific_override(monkeypatch):
    monkeypatch.setenv("FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "12.5")
    monkeypatch.delenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", raising=False)
    assert _resolve_falkor_snapshot_force_refresh_ceiling_sec() == 12.5


def test_resolve_falkor_snapshot_ceiling_falls_back_to_shared_setting(monkeypatch):
    monkeypatch.delenv("FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", raising=False)
    monkeypatch.setenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "45.0")
    assert _resolve_falkor_snapshot_force_refresh_ceiling_sec() == 45.0


def test_resolve_falkor_snapshot_ceiling_invalid_value_falls_back_to_shared_setting(monkeypatch):
    monkeypatch.setenv("FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "not-a-number")
    monkeypatch.delenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", raising=False)
    assert _resolve_falkor_snapshot_force_refresh_ceiling_sec() == 30.0  # shared default
