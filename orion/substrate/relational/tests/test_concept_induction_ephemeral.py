"""Unified-turn latency L6 step 2 (docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md).

concept_induction is no longer write-through. These tests pin:
- concept beliefs still reach the stance (slices + lineage), once each;
- the durable node's values win over the ephemeral re-read copy;
- a stance build does one Falkor rehydrate, not two (no write bumps the
  store's generation, so the layer's second snapshot() is a cache hit);
- the adapter reads the layer's store, so it sees concepts written after boot;
- anchors go cold only when a producer's last pull is actually older than its
  TTL, not because a seed node's observed_at is a day old.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.cognition.projection_builder import build_projection_unification_registry
from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateProvenanceV1
from orion.substrate.adapters._common import make_temporal
from orion.substrate.falkor_codec import encode_node_properties
from orion.substrate.falkor_store import (
    NATIVE_NODE_RETURN_FIELDS,
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)
from orion.substrate.relational import (
    CONCEPT_INDUCED,
    CognitiveUnificationLayer,
    ProducerEntryV1,
    ProducerRegistryV1,
)
from orion.substrate.relational.adapters import concept_induction_ctx as adapter_module
from orion.substrate.store import InMemorySubstrateGraphStore

_ANCHORS = ("orion", "relationship", "juniper")
_DAY_OLD = datetime.now(timezone.utc) - timedelta(hours=23)


@pytest.fixture(autouse=True)
def _reset_adapter_singleton():
    adapter_module._STORE = None
    yield
    adapter_module._STORE = None


def _concept(node_id: str, anchor: str, *, activation: float = 1.0, observed_at: datetime = _DAY_OLD) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope=anchor,
        subject_ref=f"entity:{anchor}",
        temporal=make_temporal(observed_at=observed_at),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="seed_concept",
            source_channel="substrate.seed",
            producer="test_fixture",
        ),
        label=f"{anchor}:seed",
        signals={"activation": {"activation": activation}},
    )


def _row(node: ConceptNodeV1) -> dict:
    props = encode_node_properties(node, f"concept|{node.anchor_scope}|{node.node_id}")
    return {key: props.get(key) for key in NATIVE_NODE_RETURN_FIELDS}


def _counting_falkor_store(nodes: list[ConceptNodeV1]) -> tuple[FalkorSubstrateStore, RecordingFalkorClient, list[int]]:
    client = RecordingFalkorClient(hydrate_node_rows=[_row(n) for n in nodes])
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    hydrates = [0]
    real_hydrate = store._hydrate_from_durable

    def _counting_hydrate() -> None:
        hydrates[0] += 1
        real_hydrate()

    store._hydrate_from_durable = _counting_hydrate  # type: ignore[method-assign]
    return store, client, hydrates


def _concept_entry(registry: ProducerRegistryV1) -> ProducerEntryV1:
    return next(p for p in registry.producers if p.producer_id == "concept_induction")


def _seeds() -> list[ConceptNodeV1]:
    return [
        _concept("sub-concept-seed-orion", "orion", activation=0.97),
        _concept("sub-concept-seed-relationship", "relationship", activation=0.96),
        _concept("sub-concept-seed-juniper", "juniper", activation=0.95),
    ]


def _concept_layer(store, *, tier=None) -> CognitiveUnificationLayer:
    entry = _concept_entry(build_projection_unification_registry(concept_store=store))
    if tier is not None:
        entry = ProducerEntryV1(
            producer_id=entry.producer_id,
            trust_tier=tier,
            anchor_scopes=entry.anchor_scopes,
            freshness_ttl_sec=entry.freshness_ttl_sec,
            pull_on_cold=entry.pull_on_cold,
            adapter_fn=entry.adapter_fn,
        )
    return CognitiveUnificationLayer(registry=ProducerRegistryV1(producers=[entry]), store=store)


def _merge_writes(client: RecordingFalkorClient) -> list[str]:
    return [cypher for cypher, _ in client.calls if cypher.startswith("MERGE")]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_concept_induction_is_not_write_through_and_keeps_its_tier_name() -> None:
    entry = _concept_entry(build_projection_unification_registry())
    assert entry.trust_tier.write_through is False
    assert entry.trust_tier.name == "concept_induced"
    assert entry.trust_tier.rank == 3
    # Still a cold-path producer: it must not move into the snapshot_ephemeral
    # always-run lane (that lane requires the snapshot_ephemeral tier name).
    assert entry.pull_on_cold is True
    # CONCEPT_INDUCED itself is unchanged; spark still uses it.
    assert CONCEPT_INDUCED.write_through is True


# ---------------------------------------------------------------------------
# One rehydrate per stance build, no writes
# ---------------------------------------------------------------------------


def test_stance_build_does_one_rehydrate_and_no_writes() -> None:
    store, client, hydrates = _counting_falkor_store(_seeds())
    layer = _concept_layer(store)

    beliefs = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    assert set(beliefs.cold_anchors) == set(_ANCHORS)  # first turn pulls
    assert "concept_induction:concept_induced" in beliefs.lineage
    assert beliefs.degraded_producers == []
    assert hydrates[0] == 1, "second snapshot() must be a cache hit"
    assert _merge_writes(client) == []
    assert store._write_generation == 0


def test_write_through_tier_is_what_cost_the_second_rehydrate() -> None:
    """Measurement against the old behavior: same build, old tier -> 2 rehydrates."""
    store, client, hydrates = _counting_falkor_store(_seeds())
    layer = _concept_layer(store, tier=CONCEPT_INDUCED)

    layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    assert hydrates[0] == 2
    assert _merge_writes(client)


# ---------------------------------------------------------------------------
# Dedupe, durable values preserved
# ---------------------------------------------------------------------------


def test_each_concept_appears_once_with_durable_values() -> None:
    store, _client, _hydrates = _counting_falkor_store(_seeds())
    layer = _concept_layer(store)

    beliefs = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    expected = {
        "orion": ("sub-concept-seed-orion", 0.97),
        "relationship": ("sub-concept-seed-relationship", 0.96),
        "juniper": ("sub-concept-seed-juniper", 0.95),
    }
    for anchor, (node_id, activation) in expected.items():
        ids = [n.node_id for n in beliefs.anchors[anchor].concepts]
        assert ids == [node_id], f"{anchor}: {ids}"
        node = beliefs.anchors[anchor].concepts[0]
        # Durable copy, not the adapter's re-read: the adapter stamps
        # tier_rank=3 and a concept_type default; the durable node has neither.
        assert node.provenance.tier_rank is None
        assert "concept_type" not in (node.metadata or {})
        assert node.signals.activation.activation == pytest.approx(activation)


def test_ephemeral_only_concepts_still_surface_when_durable_store_lacks_them() -> None:
    """orion-cortex-orch's cold build uses a fresh in-memory durable store; the
    adapter's unbound fallback must still put concepts in the slices."""
    adapter_module._STORE = InMemorySubstrateGraphStore()
    for node in _seeds():
        adapter_module._STORE.upsert_node(identity_key=f"concept|{node.anchor_scope}", node=node)
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[_concept_entry(build_projection_unification_registry())]),
        store=InMemorySubstrateGraphStore(),
    )

    beliefs = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    for anchor in _ANCHORS:
        assert [n.node_id for n in beliefs.anchors[anchor].concepts] == [f"sub-concept-seed-{anchor}"]
    assert "concept_induction:concept_induced" in beliefs.lineage


# ---------------------------------------------------------------------------
# Adapter reads through the layer's store
# ---------------------------------------------------------------------------


def test_bound_adapter_sees_a_concept_written_after_boot() -> None:
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="concept|orion|a", node=_concept("c-a", "orion"))
    adapter_fn = _concept_entry(build_projection_unification_registry(concept_store=store)).adapter_fn

    first = adapter_fn({})
    store.upsert_node(identity_key="concept|juniper|b", node=_concept("c-b", "juniper"))
    second = adapter_fn({})

    assert {n.node_id for n in first.nodes} == {"c-a"}
    assert {n.node_id for n in second.nodes} == {"c-a", "c-b"}
    # Bound: never opened its own store.
    assert adapter_module._STORE is None


def test_unbound_fallback_refreshes_its_own_store_every_call() -> None:
    class _SnapshotCountingStore(InMemorySubstrateGraphStore):
        snapshots = 0

        def snapshot(self):
            type(self).snapshots += 1
            return super().snapshot()

    adapter_module._STORE = _SnapshotCountingStore()
    adapter_module._STORE.upsert_node(identity_key="concept|orion|a", node=_concept("c-a", "orion"))

    adapter_module.map_concept_induction_ctx_to_substrate({})
    adapter_module.map_concept_induction_ctx_to_substrate({})

    assert _SnapshotCountingStore.snapshots == 2


# ---------------------------------------------------------------------------
# Anchor staleness judged by the last pull, not by a seed's age
# ---------------------------------------------------------------------------


def _none_producer(calls: list[int]) -> ProducerEntryV1:
    def _adapter(ctx):
        calls.append(1)
        return None

    return ProducerEntryV1(
        producer_id="nothing_to_add",
        trust_tier=CONCEPT_INDUCED,
        anchor_scopes=_ANCHORS,
        freshness_ttl_sec=300,
        pull_on_cold=True,
        adapter_fn=_adapter,
    )


def _store_with_day_old_seeds() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    for node in _seeds():
        store.upsert_node(identity_key=f"concept|{node.anchor_scope}", node=node)
    return store


def test_anchor_is_warm_after_a_pull_that_returned_nothing() -> None:
    calls: list[int] = []
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[_none_producer(calls)]),
        store=_store_with_day_old_seeds(),
    )

    first = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})
    second = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    assert set(first.cold_anchors) == set(_ANCHORS)
    assert second.cold_anchors == []
    assert len(calls) == 1


def test_anchor_goes_cold_again_once_the_last_pull_exceeds_ttl() -> None:
    calls: list[int] = []
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[_none_producer(calls)]),
        store=_store_with_day_old_seeds(),
    )
    layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    layer._last_materialized_at["nothing_to_add"] = datetime.now(timezone.utc) - timedelta(seconds=301)
    third = layer.beliefs_for_stance(anchors=_ANCHORS, ctx={})

    assert set(third.cold_anchors) == set(_ANCHORS)
    assert len(calls) == 2


def test_a_failing_producer_keeps_its_anchors_cold() -> None:
    def _boom(ctx):
        raise RuntimeError("down")

    entry = ProducerEntryV1(
        producer_id="flaky",
        trust_tier=CONCEPT_INDUCED,
        anchor_scopes=("juniper",),
        freshness_ttl_sec=300,
        pull_on_cold=True,
        adapter_fn=_boom,
    )
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[entry]),
        store=_store_with_day_old_seeds(),
    )

    layer.beliefs_for_stance(anchors=("juniper",), ctx={})
    second = layer.beliefs_for_stance(anchors=("juniper",), ctx={})

    assert second.cold_anchors == ["juniper"]
    assert "flaky" in second.degraded_producers


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


def test_tier_name_lookup_still_resolves_to_write_through_concept_tier() -> None:
    """CONCEPT_INDUCED_EPHEMERAL shares the name on purpose; a name lookup must
    never be used to rebuild a producer's tier, or write-through comes back."""
    from orion.substrate.relational import CONCEPT_INDUCED_EPHEMERAL, TIER_BY_NAME

    assert TIER_BY_NAME["concept_induced"] is CONCEPT_INDUCED
    assert CONCEPT_INDUCED_EPHEMERAL not in TIER_BY_NAME.values()


def test_transient_concept_failure_keeps_the_anchor_cold() -> None:
    """A failed pull must not count as fresh: degraded, untracked, retried."""
    class _DownStore(InMemorySubstrateGraphStore):
        def query_concept_region(self, **kwargs):
            raise ConnectionError("falkor down")

    store = _DownStore()
    for node in _seeds():
        store.upsert_node(identity_key=f"concept|{node.anchor_scope}", node=node)
    layer = _concept_layer(store)

    layer.beliefs_for_stance(anchors=("juniper",), ctx={})
    second = layer.beliefs_for_stance(anchors=("juniper",), ctx={})

    assert second.cold_anchors == ["juniper"]
    assert "concept_induction" in second.degraded_producers
    assert "concept_induction" not in layer._last_materialized_at


def test_full_registry_rehydrates_per_cold_build() -> None:
    """Whole live registry, not concept_induction alone. Spark is still
    CONCEPT_INDUCED write-through; this pins what it costs today."""
    from orion.schemas.telemetry.spark import SparkStateSnapshotV1

    spark_ctx = {
        "spark_state_json": SparkStateSnapshotV1(
            source_service="test", producer_boot_id="boot-1", seq=1,
            snapshot_ts=datetime.now(timezone.utc),
        ).model_dump_json()
    }
    counts = {}
    for label, ctx in (("no_spark", {}), ("spark", spark_ctx)):
        store, _client, hydrates = _counting_falkor_store(_seeds())
        layer = CognitiveUnificationLayer(
            registry=build_projection_unification_registry(concept_store=store), store=store
        )
        beliefs = layer.beliefs_for_stance(anchors=_ANCHORS, ctx=ctx)
        counts[label] = (hydrates[0], store._write_generation, "spark" in beliefs.degraded_producers)

    assert counts["no_spark"] == (1, 0, False)
    # Spark's node is a state_snapshot, which Falkor refuses as a durable kind,
    # so its write-through raises before any generation bump: still one
    # rehydrate (spark degrades, as identity_yaml did before #2508). Live
    # 2026-10-06: zero spark materialize failures in 24 h, so live ctx carries
    # no spark state. If spark ever becomes storable this becomes 2.
    assert counts["spark"] == (1, 0, True)


def test_dedupe_is_scoped_to_rereads() -> None:
    """A per-turn ephemeral producer that reuses a durable id keeps its copy."""
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
    from orion.substrate.relational import SNAPSHOT_EPHEMERAL

    fresh = _concept("sub-concept-seed-juniper", "juniper", activation=0.11,
                     observed_at=datetime.now(timezone.utc))
    entry = ProducerEntryV1(
        producer_id="per_turn",
        trust_tier=SNAPSHOT_EPHEMERAL,
        anchor_scopes=("juniper",),
        freshness_ttl_sec=0,
        pull_on_cold=False,
        adapter_fn=lambda ctx: SubstrateGraphRecordV1(anchor_scope="juniper", nodes=[fresh]),
    )
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[entry]), store=_store_with_day_old_seeds()
    )

    beliefs = layer.beliefs_for_stance(anchors=("juniper",), ctx={})

    activations = sorted(n.signals.activation.activation for n in beliefs.anchors["juniper"].concepts)
    assert activations == [pytest.approx(0.11), pytest.approx(0.95)]
