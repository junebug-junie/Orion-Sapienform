from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate import InMemorySubstrateGraphStore
from orion.substrate.store import SubstrateNeighborhoodSliceV1, SubstrateQueryResultV1

from orion.spark.concept_induction.profile_repository import build_concept_profile_repository
from orion.spark.concept_induction.substrate_repository import SubstrateConceptProfileRepository


def _concept_node(*, label: str, anchor_scope: str, salience: float, concept_type: str | None = None) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=f"sub-node-{label}",
        anchor_scope=anchor_scope,
        temporal=SubstrateTemporalWindowV1(observed_at=datetime(2026, 8, 11, tzinfo=timezone.utc)),
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=salience),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="topic_foundry",
            source_channel="orion:topic:foundry:concept",
            producer="orion-topic-foundry",
        ),
        label=label,
        metadata={"concept_type": concept_type} if concept_type else {},
    )


class FakeSubstrateStore:
    def __init__(self, nodes: list[ConceptNodeV1], *, degraded: bool = False) -> None:
        self._nodes = nodes
        self._degraded = degraded
        self.query_calls = 0

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        self.query_calls += 1
        return SubstrateQueryResultV1(
            query_kind="concept_region",
            slice=SubstrateNeighborhoodSliceV1(nodes=self._nodes, edges=[]),
            source_kind="falkor",
            degraded=self._degraded,
        )


class BrokenSubstrateStore:
    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        raise RuntimeError("falkor down")


class TestSubstrateConceptProfileRepository:
    def test_status_reports_substrate_backend(self):
        repo = SubstrateConceptProfileRepository(store=FakeSubstrateStore([]))
        status = repo.status()
        assert status.backend == "substrate"
        assert status.source_available is True
        assert status.placeholder_default_in_use is False

    def test_list_latest_groups_nodes_by_anchor_scope_into_profiles(self):
        nodes = [
            _concept_node(label="tissue continuity", anchor_scope="orion", salience=0.9, concept_type="self"),
            _concept_node(label="daily check-in", anchor_scope="orion", salience=0.4, concept_type="self"),
            _concept_node(label="shared journal", anchor_scope="relationship", salience=0.6, concept_type="relationship"),
        ]
        repo = SubstrateConceptProfileRepository(store=FakeSubstrateStore(nodes))

        results = repo.list_latest(["orion", "relationship", "juniper"])
        by_subject = {item.subject: item for item in results}

        assert by_subject["orion"].availability == "available"
        orion_profile = by_subject["orion"].profile
        assert orion_profile is not None
        assert orion_profile.subject == "orion"
        # Sorted by salience descending.
        assert [c.label for c in orion_profile.concepts] == ["tissue continuity", "daily check-in"]
        assert orion_profile.concepts[0].salience == 0.9
        assert orion_profile.concepts[0].type == "self"
        assert orion_profile.clusters == []
        assert orion_profile.state_estimate is None

        assert by_subject["relationship"].availability == "available"
        assert by_subject["relationship"].profile.concepts[0].label == "shared journal"

        # No juniper-anchored concept nodes -> honestly empty, not fabricated.
        assert by_subject["juniper"].availability == "empty"
        assert by_subject["juniper"].profile is None

    def test_list_latest_reports_unavailable_when_store_missing(self):
        repo = SubstrateConceptProfileRepository(store=None)
        repo._get_store = lambda: None  # type: ignore[method-assign]

        results = repo.list_latest(["orion"])
        assert results[0].availability == "unavailable"
        assert results[0].unavailable_reason == "substrate_store_unavailable"

    def test_list_latest_reports_unavailable_on_query_error(self):
        repo = SubstrateConceptProfileRepository(store=BrokenSubstrateStore())

        results = repo.list_latest(["orion"])
        assert results[0].availability == "unavailable"
        assert results[0].unavailable_reason == "substrate_store_unavailable"

    def test_lazy_get_store_treats_in_memory_fallback_as_unavailable_not_empty(self, monkeypatch):
        """build_substrate_store_from_env() never raises when SUBSTRATE_STORE_BACKEND
        is unset, or is `falkor` with a missing FALKORDB_URI -- both silently
        resolve to a working InMemorySubstrateGraphStore. That must surface as
        "unavailable" (misconfigured), not "empty" (genuinely zero concepts) --
        otherwise a dropped env var silently skips concept_induction_pass's
        fail_open_local/fail_closed cutover."""
        from orion.spark.concept_induction import substrate_repository as module

        monkeypatch.setattr(module, "build_substrate_store_from_env", lambda: InMemorySubstrateGraphStore())

        repo = SubstrateConceptProfileRepository()  # no store injected -> lazy path
        results = repo.list_latest(["orion", "juniper", "relationship"])

        for result in results:
            assert result.availability == "unavailable"
            assert result.unavailable_reason == "substrate_store_unavailable"
            assert result.profile is None

    def test_status_reports_unavailable_when_store_resolves_to_in_memory(self, monkeypatch):
        from orion.spark.concept_induction import substrate_repository as module

        monkeypatch.setattr(module, "build_substrate_store_from_env", lambda: InMemorySubstrateGraphStore())

        repo = SubstrateConceptProfileRepository()
        status = repo.status()
        assert status.source_available is False

    def test_list_latest_reports_unavailable_when_degraded(self):
        repo = SubstrateConceptProfileRepository(store=FakeSubstrateStore([], degraded=True))

        results = repo.list_latest(["orion"])
        assert results[0].availability == "unavailable"

    def test_get_latest_matches_list_latest(self):
        nodes = [_concept_node(label="tick", anchor_scope="orion", salience=0.5)]
        repo = SubstrateConceptProfileRepository(store=FakeSubstrateStore(nodes))
        result = repo.get_latest("orion")
        assert result.availability == "available"
        assert result.profile.concepts[0].label == "tick"

    def test_factory_selects_substrate_backend(self, tmp_path):
        class Settings:
            concept_profile_repository_backend = "substrate"
            store_path = str(tmp_path / "local.json")

        repo = build_concept_profile_repository(Settings())
        assert isinstance(repo, SubstrateConceptProfileRepository)
        assert repo.status().backend == "substrate"
