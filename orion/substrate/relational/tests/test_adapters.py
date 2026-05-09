"""Per-adapter unit tests for the relational substrate adapters."""

from __future__ import annotations

import pytest

from orion.substrate.relational.adapters.identity_yaml import map_identity_yaml_to_substrate
from orion.substrate.relational.adapters.recall import map_recall_bundle_to_substrate
from orion.substrate.relational.adapters.social import map_social_ctx_to_substrate
from orion.substrate.relational.registry import OPERATOR_STATIC, SNAPSHOT_EPHEMERAL


# ---------------------------------------------------------------------------
# identity_yaml adapter
# ---------------------------------------------------------------------------

class TestIdentityYamlAdapter:
    def test_returns_none_for_empty_ctx(self):
        assert map_identity_yaml_to_substrate({}) is None

    def test_returns_none_for_none_ctx(self):
        assert map_identity_yaml_to_substrate(None) is None  # type: ignore[arg-type]

    def test_produces_concept_nodes_for_orion_summary(self):
        ctx = {"orion_identity_summary": ["Orion is a cognitive presence.", "Not a generic assistant."]}
        record = map_identity_yaml_to_substrate(ctx)
        assert record is not None
        concept_nodes = [n for n in record.nodes if n.node_kind == "concept"]
        assert len(concept_nodes) == 2

    def test_concept_nodes_have_operator_static_tier_rank(self):
        ctx = {"orion_identity_summary": ["Orion is a cognitive presence."]}
        record = map_identity_yaml_to_substrate(ctx)
        assert record is not None
        for n in record.nodes:
            assert n.provenance.tier_rank == OPERATOR_STATIC.rank

    def test_anchor_scope_is_orion(self):
        ctx = {"orion_identity_summary": ["continuity"]}
        record = map_identity_yaml_to_substrate(ctx)
        assert record is not None
        assert record.anchor_scope == "orion"
        for n in record.nodes:
            assert n.anchor_scope == "orion"

    def test_snapshot_node_carries_all_three_summaries(self):
        ctx = {
            "orion_identity_summary": ["continuity"],
            "juniper_relationship_summary": ["co-architect"],
            "response_policy_summary": ["answer directly"],
        }
        record = map_identity_yaml_to_substrate(ctx)
        assert record is not None
        snapshots = [n for n in record.nodes if n.node_kind == "state_snapshot"]
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.metadata["orion_identity_summary"] == ["continuity"]
        assert snap.metadata["juniper_relationship_summary"] == ["co-architect"]
        assert snap.metadata["response_policy_summary"] == ["answer directly"]
        assert snap.snapshot_source == "identity_yaml"

    def test_empty_string_items_are_skipped(self):
        ctx = {"orion_identity_summary": ["", "  ", "real fact"]}
        record = map_identity_yaml_to_substrate(ctx)
        assert record is not None
        concept_nodes = [n for n in record.nodes if n.node_kind == "concept"]
        assert len(concept_nodes) == 1


# ---------------------------------------------------------------------------
# recall adapter
# ---------------------------------------------------------------------------

class TestRecallAdapter:
    def test_returns_none_for_empty_ctx(self):
        assert map_recall_bundle_to_substrate({}) is None

    def test_returns_none_for_empty_fragments(self):
        assert map_recall_bundle_to_substrate({"recall_bundle": {"fragments": []}}) is None

    def test_journal_fragment_produces_concept_node(self):
        ctx = {"recall_bundle": {"fragments": [{"source": "journal", "snippet": "deep reflection on autonomy"}]}}
        record = map_recall_bundle_to_substrate(ctx)
        assert record is not None
        concepts = [n for n in record.nodes if n.node_kind == "concept"]
        assert len(concepts) == 1
        assert concepts[0].label == "deep reflection on autonomy"

    def test_tension_fragment_produces_tension_node(self):
        ctx = {"recall_bundle": {"fragments": [{"source": "tension:growth", "snippet": "conflict between rest and action"}]}}
        record = map_recall_bundle_to_substrate(ctx)
        assert record is not None
        tensions = [n for n in record.nodes if n.node_kind == "tension"]
        assert len(tensions) == 1

    def test_dream_fragment_produces_event_node(self):
        ctx = {"recall_bundle": {"fragments": [{"source": "dream", "snippet": "flying over the city"}]}}
        record = map_recall_bundle_to_substrate(ctx)
        assert record is not None
        events = [n for n in record.nodes if n.node_kind == "event"]
        assert len(events) == 1
        assert events[0].event_type == "dream"

    def test_all_nodes_have_snapshot_ephemeral_tier(self):
        ctx = {"recall_bundle": {"fragments": [{"source": "journal", "snippet": "reflection"}]}}
        record = map_recall_bundle_to_substrate(ctx)
        assert record is not None
        for n in record.nodes:
            assert n.provenance.tier_rank == SNAPSHOT_EPHEMERAL.rank

    def test_empty_snippet_items_skipped(self):
        ctx = {"recall_bundle": {"fragments": [{"source": "journal", "snippet": ""}]}}
        assert map_recall_bundle_to_substrate(ctx) is None

    def test_non_dict_fragment_skipped(self):
        ctx = {"recall_bundle": {"fragments": ["not a dict", None]}}
        assert map_recall_bundle_to_substrate(ctx) is None


# ---------------------------------------------------------------------------
# social adapter
# ---------------------------------------------------------------------------

class TestSocialAdapter:
    def test_returns_none_for_empty_ctx(self):
        assert map_social_ctx_to_substrate({}) is None

    def test_produces_state_snapshot_with_anchor_relationship(self):
        ctx = {"social_turn_policy": {"addressed": True, "should_speak": True, "decision": "speak"}}
        record = map_social_ctx_to_substrate(ctx)
        assert record is not None
        snapshots = [n for n in record.nodes if n.node_kind == "state_snapshot"]
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.anchor_scope == "relationship"
        assert snap.snapshot_source == "social_bridge"

    def test_posture_tags_derived_from_orientation(self):
        ctx = {
            "social_stance_snapshot": {"recent_social_orientation_summary": "direct and warm conversation"},
            "social_turn_policy": {"addressed": True},
        }
        record = map_social_ctx_to_substrate(ctx)
        assert record is not None
        snap = record.nodes[0]
        assert "direct" in snap.metadata["posture"]
        assert "warm" in snap.metadata["posture"]

    def test_all_nodes_have_snapshot_ephemeral_tier(self):
        ctx = {"social_turn_policy": {"addressed": True}}
        record = map_social_ctx_to_substrate(ctx)
        assert record is not None
        for n in record.nodes:
            assert n.provenance.tier_rank == SNAPSHOT_EPHEMERAL.rank


# ---------------------------------------------------------------------------
# Existing adapter smoke tests (registered, unchanged)
# ---------------------------------------------------------------------------

class TestExistingAdapterSmoke:
    def test_autonomy_adapter_still_importable(self):
        from orion.substrate.adapters.autonomy import map_autonomy_artifacts_to_substrate
        assert callable(map_autonomy_artifacts_to_substrate)

    def test_concept_induction_adapter_still_importable(self):
        from orion.substrate.adapters.concept_induction import map_concept_profile_to_substrate
        assert callable(map_concept_profile_to_substrate)

    def test_spark_adapter_still_importable(self):
        from orion.substrate.adapters.spark import map_spark_source_snapshot_to_substrate
        assert callable(map_spark_source_snapshot_to_substrate)


# ---------------------------------------------------------------------------
# self_study adapter — returns None without configured endpoint (no network)
# ---------------------------------------------------------------------------

class TestSelfStudyAdapter:
    def test_returns_none_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("SELF_STUDY_NAMED_GRAPH", raising=False)
        monkeypatch.delenv("GRAPHDB_QUERY_ENDPOINT", raising=False)
        monkeypatch.delenv("GRAPHDB_URL", raising=False)
        from orion.substrate.relational.adapters.self_study import map_self_study_to_substrate
        result = map_self_study_to_substrate({})
        assert result is None

    def test_empty_ctx_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("SELF_STUDY_NAMED_GRAPH", raising=False)
        from orion.substrate.relational.adapters.self_study import map_self_study_to_substrate
        try:
            map_self_study_to_substrate({})
        except Exception as exc:
            pytest.fail(f"self_study adapter raised on empty ctx: {exc}")


# ---------------------------------------------------------------------------
# orionmem adapter — anchor routing and None-on-missing-config
# ---------------------------------------------------------------------------

class TestOrionmemAdapter:
    def test_returns_none_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("GRAPHDB_QUERY_ENDPOINT", raising=False)
        monkeypatch.delenv("GRAPHDB_URL", raising=False)
        from orion.substrate.relational.adapters.orionmem import map_orionmem_to_substrate
        result = map_orionmem_to_substrate({})
        assert result is None

    def test_empty_ctx_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("GRAPHDB_QUERY_ENDPOINT", raising=False)
        from orion.substrate.relational.adapters.orionmem import map_orionmem_to_substrate
        try:
            map_orionmem_to_substrate({})
        except Exception as exc:
            pytest.fail(f"orionmem adapter raised on empty ctx: {exc}")

    def test_anchor_routing_relationship(self):
        from orion.substrate.relational.adapters.orionmem import _anchor_for_graph
        assert _anchor_for_graph("https://orion.local/mem/relationship_context") == "relationship"
        assert _anchor_for_graph("https://orion.local/mem/juniper_memory") == "relationship"

    def test_anchor_routing_orion_default(self):
        from orion.substrate.relational.adapters.orionmem import _anchor_for_graph
        assert _anchor_for_graph("https://orion.local/mem/orion_self") == "orion"
        assert _anchor_for_graph("") == "orion"


# ---------------------------------------------------------------------------
# recall adapter — anchor routing (juniper branch now reachable)
# ---------------------------------------------------------------------------

class TestRecallAnchorRouting:
    def test_juniper_routes_to_juniper_not_relationship(self):
        from orion.substrate.relational.adapters.recall import _anchor_for_fragment
        assert _anchor_for_fragment({"subject": "juniper recall"}) == "juniper"

    def test_relationship_routes_to_relationship(self):
        from orion.substrate.relational.adapters.recall import _anchor_for_fragment
        assert _anchor_for_fragment({"subject": "relationship memory"}) == "relationship"

    def test_orion_routes_to_default(self):
        from orion.substrate.relational.adapters.recall import _anchor_for_fragment
        assert _anchor_for_fragment({"subject": "orion self"}) == "orion"
        assert _anchor_for_fragment({}) == "orion"


# ---------------------------------------------------------------------------
# spark ctx adapter
# ---------------------------------------------------------------------------

class TestSparkCtxAdapter:
    def test_returns_none_without_spark_payload(self):
        from orion.substrate.relational.adapters.spark_ctx import map_spark_ctx_to_substrate

        assert map_spark_ctx_to_substrate({}) is None

    def test_maps_json_string_with_concept_induced_tier(self):
        from datetime import datetime, timezone

        from orion.schemas.telemetry.spark import SparkStateSnapshotV1
        from orion.substrate.relational.adapters.spark_ctx import map_spark_ctx_to_substrate
        from orion.substrate.relational.registry import CONCEPT_INDUCED

        snap = SparkStateSnapshotV1(
            source_service="test",
            producer_boot_id="boot-1",
            seq=1,
            snapshot_ts=datetime.now(timezone.utc),
        )
        ctx = {"spark_state_json": snap.model_dump_json()}
        record = map_spark_ctx_to_substrate(ctx)
        assert record is not None
        assert record.nodes
        for n in record.nodes:
            assert n.provenance.tier_rank == CONCEPT_INDUCED.rank
