"""Falkor post-save materialization for concept profiles (concept-only, Cypher-native)."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.concept_induction import (
    ConceptCluster,
    ConceptEvidenceRef,
    ConceptItem,
    ConceptProfile,
    StateEstimate,
)
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.falkor_materialization import (
    filter_concept_atlas_record,
    materialize_concept_profile_to_falkor,
)
from orion.spark.concept_induction.settings import ConceptSettings
from orion.spark.concept_induction.inducer import WindowEvent
from orion.substrate.adapters.concept_induction import map_concept_profile_to_substrate
from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)


def _fixture_profile(*, with_cluster: bool = True) -> ConceptProfile:
    evidence = ConceptEvidenceRef(
        message_id=uuid4(),
        timestamp=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
        channel="orion:chat:history:log",
    )
    concepts = [
        ConceptItem(
            concept_id="concept-1",
            label="coherence",
            type="motif",
            salience=0.9,
            confidence=0.8,
            aliases=["consistency"],
            evidence=[evidence],
            metadata={"definition": "keeping a thread"},
        ),
        ConceptItem(
            concept_id="concept-2",
            label="curiosity",
            type="motif",
            salience=0.7,
            confidence=0.75,
            evidence=[evidence],
        ),
    ]
    clusters = []
    if with_cluster:
        clusters = [
            ConceptCluster(
                cluster_id="cluster-1",
                label="core",
                summary="Core concepts",
                concept_ids=["concept-1", "concept-2"],
                cohesion_score=0.75,
            )
        ]
    return ConceptProfile(
        profile_id="profile-abc",
        subject="orion",
        revision=7,
        created_at=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        window_start=datetime(2026, 3, 24, 10, 15, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        concepts=concepts,
        clusters=clusters,
        state_estimate=StateEstimate(
            dimensions={"novelty": 0.2},
            trend={"novelty": -0.1},
            confidence=0.6,
            window_start=datetime(2026, 3, 24, 10, 15, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        ),
        metadata={"algorithm": "concept_induction.v1"},
    )


def _assert_no_payload_json_sor(cypher: str, params: dict | None) -> None:
    assert "$payload_json" not in cypher
    assert "payload_json =" not in cypher.replace("REMOVE n.payload_json", "").replace(
        "REMOVE e.payload_json", ""
    )
    assert params is not None
    assert "payload_json" not in params


def test_filter_skips_non_concept_nodes_and_edges() -> None:
    record = map_concept_profile_to_substrate(profile=_fixture_profile())
    assert any(getattr(n, "node_kind", None) != "concept" for n in record.nodes)

    concepts, edges, skipped_nodes, skipped_edges = filter_concept_atlas_record(record)
    assert len(concepts) == 2
    assert skipped_nodes >= 1  # evidence + hypothesis at least
    assert all(n.node_kind == "concept" for n in concepts)
    assert all(
        e.source.node_kind == "concept" and e.target.node_kind == "concept" for e in edges
    )
    assert skipped_edges >= 1


def test_materialize_writes_native_cypher_without_payload_json_sor() -> None:
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    result = materialize_concept_profile_to_falkor(profile=_fixture_profile(), store=store)
    assert result.concept_nodes == 2
    assert result.skipped_nodes >= 1
    # The profile mapper emits no concept↔concept edges today: live outcome is
    # concept NODES only. If this assertion breaks, the mapper gained
    # concept↔concept relations — update docs and verify Hub identity merge.
    assert result.concept_edges == 0
    assert result.skipped_edges >= 1

    node_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert node_calls
    for cypher, params in node_calls:
        _assert_no_payload_json_sor(cypher, params)
        assert "n.evidence_refs_json = $evidence_refs_json" in cypher
        assert params["node_kind"] == "concept"
        assert "REMOVE n.payload_json" in cypher

    # Non-concept kinds must never be written (would raise on store; also no Drive/Evidence MERGE)
    assert all("Evidence" not in c and "Hypothesis" not in c for c, _ in client.calls)


def test_materialize_populates_write_through_cache() -> None:
    """Write-through cache readback (hydrate=False) — not a durable Falkor hydrate."""
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    materialize_concept_profile_to_falkor(profile=_fixture_profile(), store=store)
    snap = store.snapshot()
    assert "sub-concept-concept-1" in snap.nodes
    assert "sub-concept-concept-2" in snap.nodes
    assert store.get_node_by_id("sub-concept-concept-1").label == "coherence"


class _ExplodingStore:
    def upsert_node(self, *, identity_key, node):
        raise RuntimeError("falkor_down")

    def upsert_edge(self, *, identity_key, edge):
        raise RuntimeError("falkor_down")


class StubInducer:
    def __init__(self, profile, save_hook=None):
        self.profile = profile
        self.save_hook = save_hook

    async def run(self, subject, window):
        from orion.core.schemas.concept_induction import ConceptProfileDelta
        from types import SimpleNamespace

        if self.save_hook:
            self.save_hook(subject, self.profile)
        return SimpleNamespace(profile=self.profile, delta=None)


class FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _env_with_text(text: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.message",
        source=ServiceRef(name="test", version="0.0.0"),
        payload={"content": text},
    )


def test_falkor_backend_does_not_publish_rdf_write_request() -> None:
    with tempfile.TemporaryDirectory() as td:
        client = RecordingFalkorClient()
        store = FalkorSubstrateStore(
            FalkorSubstrateStoreConfig(uri="redis://127.0.0.1:6380", graph_name="orion_substrate"),
            client=client,
            hydrate=False,
        )
        worker = ConceptWorker(
            ConceptSettings(
                orion_bus_enabled=False,
                store_path=str(Path(td) / "state.json"),
                concept_profile_graph_backend="falkor",
            ),
            substrate_store=store,
        )
        profile = _fixture_profile(with_cluster=False)
        worker.inducer = StubInducer(profile=profile)
        worker.bus = FakeBus()
        worker.window["orion"] = [
            WindowEvent(
                text="Orion keeps coherence.",
                timestamp=datetime.now(timezone.utc),
                envelope=_env_with_text("Orion keeps coherence."),
                intake_channel="orion:chat:history:log",
            )
        ]
        asyncio.run(worker.run_for_subject("orion"))
        published = worker.bus.published
        assert not any(env.kind == "rdf.write.request" for _, env in published)
        assert any(env.kind == "memory.concepts.profile.v1" for _, env in published)
        assert any("MERGE (n:SubstrateNode" in c for c, _ in client.calls)


def test_rdf_backend_still_publishes_rdf_write_request() -> None:
    with tempfile.TemporaryDirectory() as td:
        worker = ConceptWorker(
            ConceptSettings(
                orion_bus_enabled=False,
                store_path=str(Path(td) / "state.json"),
                concept_profile_graph_backend="rdf",
            )
        )
        profile = _fixture_profile(with_cluster=False)
        worker.inducer = StubInducer(profile=profile)
        worker.bus = FakeBus()
        worker.window["orion"] = [
            WindowEvent(
                text="Orion keeps coherence.",
                timestamp=datetime.now(timezone.utc),
                envelope=_env_with_text("Orion keeps coherence."),
                intake_channel="orion:chat:history:log",
            )
        ]
        asyncio.run(worker.run_for_subject("orion"))
        published = worker.bus.published
        assert any(
            channel == worker.cfg.forward_rdf_channel and env.kind == "rdf.write.request"
            for channel, env in published
        )


def test_falkor_materialization_failure_isolated_from_local_write() -> None:
    with tempfile.TemporaryDirectory() as td:
        store_path = Path(td) / "state.json"
        worker = ConceptWorker(
            ConceptSettings(
                orion_bus_enabled=False,
                store_path=str(store_path),
                concept_profile_graph_backend="falkor",
            ),
            substrate_store=_ExplodingStore(),
        )
        profile = _fixture_profile(with_cluster=False)

        def _save_local(subject: str, p):
            worker.store.save(subject, p, "hash-local-test")

        worker.inducer = StubInducer(profile=profile, save_hook=_save_local)
        worker.bus = FakeBus()
        worker.window["orion"] = [
            WindowEvent(
                text="Orion forms concepts.",
                timestamp=datetime.now(timezone.utc),
                envelope=_env_with_text("Orion forms concepts."),
                intake_channel="orion:chat:history:log",
            )
        ]
        asyncio.run(worker.run_for_subject("orion"))
        reloaded = worker.store.load("orion")
        assert reloaded is not None
        assert reloaded.subject == "orion"
        assert any(env.kind == "memory.concepts.profile.v1" for _, env in worker.bus.published)
        assert not any(env.kind == "rdf.write.request" for _, env in worker.bus.published)


def test_hub_store_sees_spark_writes_only_after_rehydrate() -> None:
    """Cross-process visibility contract: a Hub-side FalkorSubstrateStore reads
    from its in-process cache hydrated at construction. Spark writes from a
    separate process are invisible to an already-running Hub until its store
    re-hydrates (today: Hub restart). This test locks that contract down so a
    future refresh-on-query change surfaces here."""

    class SharedDurableClient:
        """Simulates the shared FalkorDB: records writes, serves hydrate reads
        from what has been written so far (native rows via encode params)."""

        def __init__(self) -> None:
            self.node_rows: dict[str, dict] = {}
            self.calls: list[tuple[str, dict | None]] = []

        def graph_query(self, cypher: str, params: dict | None = None):
            self.calls.append((cypher, params))
            if "MERGE (n:SubstrateNode" in cypher and params:
                self.node_rows[params["node_id"]] = dict(params)
                return []
            if "WHERE n.payload_json IS NOT NULL" in cypher or "WHERE e.payload_json IS NOT NULL" in cypher:
                return []
            if "MATCH (n:SubstrateNode) RETURN" in cypher:
                return list(self.node_rows.values())
            return []

    shared = SharedDurableClient()

    # Hub hydrates first, before Spark has written anything.
    hub_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://127.0.0.1:6380", graph_name="orion_substrate"),
        client=shared,
        hydrate=True,
    )
    assert hub_store.get_node_by_id("sub-concept-concept-1") is None

    # Spark writes through its own store instance (separate process in prod).
    spark_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://127.0.0.1:6380", graph_name="orion_substrate"),
        client=shared,
        hydrate=False,
    )
    materialize_concept_profile_to_falkor(profile=_fixture_profile(), store=spark_store)
    assert "sub-concept-concept-1" in shared.node_rows

    # Already-running Hub store is stale — the durable write is NOT visible.
    assert hub_store.get_node_by_id("sub-concept-concept-1") is None

    # A re-hydrated Hub store (restart) sees the Spark-written concept natively.
    hub_store_restarted = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://127.0.0.1:6380", graph_name="orion_substrate"),
        client=shared,
        hydrate=True,
    )
    node = hub_store_restarted.get_node_by_id("sub-concept-concept-1")
    assert node is not None
    assert node.label == "coherence"


def test_invalid_graph_backend_fails_closed_to_disabled() -> None:
    cfg = ConceptSettings(concept_profile_graph_backend="wat")
    assert cfg.concept_profile_graph_backend == "disabled"


def test_disabled_backend_writes_nothing_and_succeeds() -> None:
    with tempfile.TemporaryDirectory() as td:
        client = RecordingFalkorClient()
        store = FalkorSubstrateStore(
            FalkorSubstrateStoreConfig(uri="redis://127.0.0.1:6380", graph_name="orion_substrate"),
            client=client,
            hydrate=False,
        )
        worker = ConceptWorker(
            ConceptSettings(
                orion_bus_enabled=False,
                store_path=str(Path(td) / "state.json"),
                concept_profile_graph_backend="disabled",
            ),
            substrate_store=store,
        )
        profile = _fixture_profile(with_cluster=False)
        worker.inducer = StubInducer(profile=profile)
        worker.bus = FakeBus()
        worker.window["orion"] = [
            WindowEvent(
                text="Orion keeps coherence.",
                timestamp=datetime.now(timezone.utc),
                envelope=_env_with_text("Orion keeps coherence."),
                intake_channel="orion:chat:history:log",
            )
        ]
        asyncio.run(worker.run_for_subject("orion"))
        published = worker.bus.published
        assert not any(env.kind == "rdf.write.request" for _, env in published)
        assert client.calls == []
        assert any(env.kind == "memory.concepts.profile.v1" for _, env in published)


def test_edge_identity_matches_store_canonical_format() -> None:
    """edge_identity_key must match FalkorSubstrateStore._edge_identity
    (``src|pred|tgt``) so identity caches agree with Hub when
    concept↔concept edges appear."""
    from orion.core.schemas.cognitive_substrate import (
        NodeRefV1,
        SubstrateEdgeV1,
        SubstrateProvenanceV1,
        SubstrateTemporalWindowV1,
    )
    from orion.spark.concept_induction.falkor_materialization import edge_identity_key

    edge = SubstrateEdgeV1(
        source=NodeRefV1(node_id="sub-concept-a", node_kind="concept"),
        target=NodeRefV1(node_id="sub-concept-b", node_kind="concept"),
        predicate="associated_with",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_materialization",
        ),
    )
    assert edge_identity_key(edge) == FalkorSubstrateStore._edge_identity(edge)
    assert edge_identity_key(edge) == "sub-concept-a|associated_with|sub-concept-b"
