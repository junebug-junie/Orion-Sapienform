"""Unit tests for CognitiveUnificationLayer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.relational.beliefs import UnifiedRelationalBeliefSetV1
from orion.substrate.relational.layer import CognitiveUnificationLayer
from orion.substrate.relational.registry import (
    CONCEPT_INDUCED,
    OPERATOR_STATIC,
    SNAPSHOT_EPHEMERAL,
    ProducerEntryV1,
    ProducerRegistryV1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prov(*, tier_rank: int | None = None) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="test",
        producer="test",
        tier_rank=tier_rank,
    )


def _make_temporal(*, age_sec: float = 0.0) -> SubstrateTemporalWindowV1:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    return SubstrateTemporalWindowV1(observed_at=ts)


def _concept_record(anchor: str, label: str, *, tier_rank: int | None = None, age_sec: float = 0.0) -> SubstrateGraphRecordV1:
    node = ConceptNodeV1(
        anchor_scope=anchor,
        label=label,
        temporal=_make_temporal(age_sec=age_sec),
        provenance=_make_prov(tier_rank=tier_rank),
        signals=SubstrateSignalBundleV1(confidence=0.8),
        metadata={"concept_id": label.lower().replace(" ", "_")},
    )
    return SubstrateGraphRecordV1(anchor_scope=anchor, nodes=[node])


def _snapshot_record(anchor: str, source: str, *, tier_rank: int | None = None) -> SubstrateGraphRecordV1:
    node = StateSnapshotNodeV1(
        anchor_scope=anchor,
        temporal=_make_temporal(),
        provenance=_make_prov(tier_rank=tier_rank),
        snapshot_source=source,
        dimensions={},
        metadata={"source": source},
    )
    return SubstrateGraphRecordV1(anchor_scope=anchor, nodes=[node])


def _make_layer(producers: list[ProducerEntryV1]) -> tuple[CognitiveUnificationLayer, InMemorySubstrateGraphStore]:
    store = InMemorySubstrateGraphStore()
    registry = ProducerRegistryV1(producers=producers)
    layer = CognitiveUnificationLayer(registry=registry, store=store)
    return layer, store


# ---------------------------------------------------------------------------
# Warm path tests
# ---------------------------------------------------------------------------

class TestWarmPath:
    def test_warm_anchor_does_not_trigger_fan_out(self):
        call_count = {"n": 0}

        def adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            call_count["n"] += 1
            return _concept_record("orion", "some_concept", tier_rank=CONCEPT_INDUCED.rank)

        layer, store = _make_layer([
            ProducerEntryV1(
                producer_id="test_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=adapter,
            )
        ])

        # Pre-populate with a fresh node
        fresh_record = _concept_record("orion", "pre_existing", tier_rank=CONCEPT_INDUCED.rank, age_sec=10)
        from orion.substrate.materializer import SubstrateGraphMaterializer
        SubstrateGraphMaterializer(store=store).apply_record(fresh_record)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert call_count["n"] == 0, "Warm anchor should not trigger fan-out"
        assert "orion" not in beliefs.cold_anchors

    def test_stale_anchor_triggers_fan_out(self):
        call_count = {"n": 0}

        def adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            call_count["n"] += 1
            return None

        layer, store = _make_layer([
            ProducerEntryV1(
                producer_id="test_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=5,
                pull_on_cold=True,
                adapter_fn=adapter,
            )
        ])

        # Pre-populate with a stale node (older than TTL)
        stale_record = _concept_record("orion", "old_concept", tier_rank=CONCEPT_INDUCED.rank, age_sec=60)
        from orion.substrate.materializer import SubstrateGraphMaterializer
        SubstrateGraphMaterializer(store=store).apply_record(stale_record)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert call_count["n"] == 1, "Stale anchor should trigger fan-out"
        assert "orion" in beliefs.cold_anchors


# ---------------------------------------------------------------------------
# Cold path tests
# ---------------------------------------------------------------------------

class TestColdPath:
    def test_cold_anchor_triggers_fan_out(self):
        call_count = {"n": 0}

        def adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            call_count["n"] += 1
            return _concept_record("orion", "fresh_concept", tier_rank=CONCEPT_INDUCED.rank)

        layer, _ = _make_layer([
            ProducerEntryV1(
                producer_id="test_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=adapter,
            )
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert call_count["n"] == 1
        assert "orion" in beliefs.cold_anchors
        assert len(beliefs.anchors["orion"].concepts) == 1

    def test_cold_path_nodes_appear_in_output(self):
        def adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            return _concept_record("orion", "induced_concept", tier_rank=CONCEPT_INDUCED.rank)

        layer, _ = _make_layer([
            ProducerEntryV1(
                producer_id="p1",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=adapter,
            )
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert any(getattr(n, "label", "") == "induced_concept" for n in beliefs.anchors["orion"].concepts)


# ---------------------------------------------------------------------------
# Tier protection tests
# ---------------------------------------------------------------------------

class TestTierProtection:
    def test_operator_static_not_overwritten_by_concept_induced(self):
        """An operator_static node's confidence must not be raised by concept_induced."""
        op_static_label = "persistent identity fact"

        def op_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            return _concept_record("orion", op_static_label, tier_rank=OPERATOR_STATIC.rank)

        def concept_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            node = ConceptNodeV1(
                anchor_scope="orion",
                label=op_static_label,
                temporal=_make_temporal(),
                provenance=_make_prov(tier_rank=CONCEPT_INDUCED.rank),
                signals=SubstrateSignalBundleV1(confidence=1.0),  # higher than op_static's 0.8
                metadata={"concept_id": op_static_label.lower().replace(" ", "_")},
            )
            return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])

        layer, store = _make_layer([
            ProducerEntryV1(
                producer_id="op_producer",
                trust_tier=OPERATOR_STATIC,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=op_adapter,
            ),
            ProducerEntryV1(
                producer_id="concept_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=concept_adapter,
            ),
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        concepts = [n for n in beliefs.anchors["orion"].concepts if getattr(n, "label", "") == op_static_label]
        assert concepts, "operator_static concept node should be present"
        # Confidence must not exceed the operator_static producer's 0.8 (concept_induced set 1.0 but is blocked)
        assert concepts[0].signals.confidence <= 0.8 + 1e-6


# ---------------------------------------------------------------------------
# Ephemeral isolation tests
# ---------------------------------------------------------------------------

class TestEphemeralIsolation:
    def test_snapshot_ephemeral_nodes_absent_from_durable_store(self):
        def ephemeral_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            return _snapshot_record("orion", "social_bridge", tier_rank=SNAPSHOT_EPHEMERAL.rank)

        layer, store = _make_layer([
            ProducerEntryV1(
                producer_id="social",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion",),
                freshness_ttl_sec=120,
                pull_on_cold=False,
                adapter_fn=ephemeral_adapter,
            )
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        # Node must appear in beliefs output
        assert len(beliefs.anchors["orion"].snapshots) > 0
        # But must NOT be in the durable store
        durable_snapshot = store.snapshot()
        ephemeral_in_durable = [
            n for n in durable_snapshot.nodes.values()
            if getattr(n, "snapshot_source", "") == "social_bridge"
        ]
        assert len(ephemeral_in_durable) == 0, "snapshot_ephemeral nodes must not be in the durable store"


# ---------------------------------------------------------------------------
# Degraded producer tests
# ---------------------------------------------------------------------------

class TestDegradedProducer:
    def test_failing_adapter_marks_degraded_but_returns(self):
        def bad_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            raise RuntimeError("simulated failure")

        def good_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            return _concept_record("orion", "healthy_concept", tier_rank=CONCEPT_INDUCED.rank)

        layer, _ = _make_layer([
            ProducerEntryV1(
                producer_id="bad_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=bad_adapter,
            ),
            ProducerEntryV1(
                producer_id="good_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=good_adapter,
            ),
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert beliefs is not None
        assert "bad_producer" in beliefs.degraded_producers
        assert beliefs.anchors["orion"].degraded
        # Good producer still contributed
        assert len(beliefs.anchors["orion"].concepts) > 0

    def test_timeout_marks_slow_producer_degraded_not_entire_belief_set(self):
        """A single timed-out producer must not discard results from faster producers."""
        import time

        def slow_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            time.sleep(10)
            return None

        def fast_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            return _concept_record("orion", "fast_concept", tier_rank=CONCEPT_INDUCED.rank)

        layer, _ = _make_layer([
            ProducerEntryV1(
                producer_id="slow_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=slow_adapter,
            ),
            ProducerEntryV1(
                producer_id="fast_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=fast_adapter,
            ),
        ])

        beliefs = layer.beliefs_for_stance(anchors=["orion"], timeout_sec=0.5)
        assert beliefs is not None, "belief set must always be returned"
        assert "slow_producer" in beliefs.degraded_producers
        assert beliefs.anchors["orion"].degraded
        assert len(beliefs.anchors["orion"].concepts) > 0, "fast_producer result must survive"


# ---------------------------------------------------------------------------
# TTL staleness tests
# ---------------------------------------------------------------------------

class TestTtlStaleness:
    def test_node_older_than_ttl_triggers_re_materialization(self):
        call_count = {"n": 0}

        def adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
            call_count["n"] += 1
            return None

        layer, store = _make_layer([
            ProducerEntryV1(
                producer_id="stale_producer",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=10,
                pull_on_cold=True,
                adapter_fn=adapter,
            )
        ])

        # Seed a node that's 30 seconds old (older than TTL=10s)
        old_record = _concept_record("orion", "old_node", tier_rank=CONCEPT_INDUCED.rank, age_sec=30)
        from orion.substrate.materializer import SubstrateGraphMaterializer
        SubstrateGraphMaterializer(store=store).apply_record(old_record)

        call_count["n"] = 0
        layer.beliefs_for_stance(anchors=["orion"])
        assert call_count["n"] == 1
