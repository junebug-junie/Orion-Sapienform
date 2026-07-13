from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.governor import GovernorError, approve, reject, supersede
from orion.memory.crystallization.projection_cards import build_memory_card_projection, can_project_to_card
from orion.memory.crystallization.projection_chroma import build_chroma_upsert, chroma_bus_envelope_kind
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.projection_rdf import build_rdf_projection_hint
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.salience import infer_confidence, score_salience, apply_salience
from orion.memory.crystallization.bus_emit import LIFECYCLE_KINDS, CHANNEL_DEFAULTS
from orion.memory.crystallization.detection import detect_contradictions, detect_duplicates, merge_detection
from orion.memory.crystallization.schemas import (
    CrystallizationDynamicsV1,
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    CrystallizationLinkV1,
    MemoryCrystallizationProposeRequestV1,
    MemoryCrystallizationV1,
)
from orion.memory.crystallization.validator import validate_proposal
from orion.schemas.registry import resolve


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _base_request(**kwargs) -> MemoryCrystallizationProposeRequestV1:
    defaults = {
        "kind": "semantic",
        "subject": "Test subject",
        "summary": "Test summary",
        "scope": ["project:orion"],
        "evidence": [
            CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="card_123", excerpt="fact")
        ],
        "proposed_by": "test",
    }
    defaults.update(kwargs)
    return MemoryCrystallizationProposeRequestV1(**defaults)


def _active_crystallization(**kwargs) -> MemoryCrystallizationV1:
    req = _base_request(**kwargs)
    crys = propose(req)
    crys.governance.approved_by = "operator"
    crys.governance.validation_status = "valid"
    updated, _ = approve(crys, actor="operator")
    return updated


class TestMemoryCrystallizationSchema:
    def test_schema_validates(self):
        crys = _base_request().to_crystallization()
        parsed = MemoryCrystallizationV1.model_validate(crys.model_dump())
        assert parsed.schema_version == "memory_crystallization.v1"

    def test_registry_resolves(self):
        assert resolve("MemoryCrystallizationV1") is MemoryCrystallizationV1
        from orion.memory.crystallization.schemas import ActiveMemoryPacketV1

        assert resolve("ActiveMemoryPacketV1") is ActiveMemoryPacketV1


class TestGovernorPath:
    def test_proposal_cannot_become_active_without_governor(self):
        crys = propose(_base_request())
        with pytest.raises(GovernorError):
            approve(crys, actor="operator")

    def test_approve_after_manual_review(self):
        crys = _active_crystallization()
        assert crys.status == "active"
        assert crys.governance.approved_by == "operator"
        assert crys.dynamics.formed_at is not None
        assert crys.dynamics.activation > 0.0

    def test_rejected_does_not_project(self):
        crys = propose(_base_request())
        crys.governance.approved_by = "operator"
        crys.governance.validation_status = "valid"
        updated, _ = reject(crys, actor="operator")
        assert not can_project_to_card(updated)
        assert build_chroma_upsert(updated) is None

    def test_supersession_preserves_old(self):
        old = _active_crystallization()
        updated, history = supersede(old, actor="operator", superseded_by_id="crys_new")
        assert updated.status == "superseded"
        assert history["after"]["status"] == "superseded"


class TestValidationRules:
    def test_stance_requires_planning_effects(self):
        req = _base_request(
            kind="stance",
            planning_effects=[],
            retrieval_affordances=["retrieve_when:memory"],
        )
        crys = propose(req)
        result = validate_proposal(crys)
        assert not result.valid
        assert any("planning_effects" in e for e in result.errors)

    def test_contradiction_requires_two_targets(self):
        req = _base_request(
            kind="contradiction",
            links=[
                CrystallizationLinkV1(
                    target_crystallization_id="crys_a",
                    relation="contradicts",
                )
            ],
        )
        crys = propose(req)
        result = validate_proposal(crys)
        assert not result.valid

    def test_evidence_required_on_propose(self):
        with pytest.raises(ValidationError):
            _base_request(evidence=[])


class TestSourceReferences:
    def test_can_reference_memory_card(self):
        req = _base_request(source_card_ids=["card-uuid-1"])
        crys = propose(req)
        assert "card-uuid-1" in crys.source_card_ids

    def test_can_reference_grammar_event(self):
        req = _base_request(
            source_grammar_event_ids=["gev_abc"],
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="grammar_event", source_id="gev_abc")
            ],
        )
        crys = propose(req)
        assert "gev_abc" in crys.source_grammar_event_ids


class TestProjections:
    def test_active_projects_to_memory_card(self):
        crys = _active_crystallization(kind="stance", planning_effects=["prefer local"], retrieval_affordances=["retrieve_when:test"])
        card = build_memory_card_projection(crys)
        assert card is not None
        assert "crystallization" in card.types
        assert card.subschema["crystallization_ref"]["crystallization_id"] == crys.crystallization_id

    def test_chroma_projection_emits_vector_upsert_payload(self):
        crys = _active_crystallization()
        upsert = build_chroma_upsert(crys)
        assert upsert is not None
        assert upsert.kind == "memory.crystallization"
        assert chroma_bus_envelope_kind() == "memory.vector.upsert.v1"

    def test_graphiti_cannot_mutate_canonical(self):
        crys = _active_crystallization()
        adapter = GraphitiAdapter(enabled=True, url="http://graphiti")
        result = adapter.sync_crystallization(crys)
        updated = adapter.apply_projection_refs(crys, result)
        assert result.canonical_mutated is False
        assert updated.summary == crys.summary
        assert updated.governance.model_dump() == crys.governance.model_dump()


class TestActivePacket:
    def test_active_packet_includes_crystallization_refs(self):
        crys = _active_crystallization(kind="stance", planning_effects=["x"], retrieval_affordances=["y"])
        packet = build_active_packet(query="memory architecture", crystallizations=[crys])
        assert crys.crystallization_id in packet.crystallization_refs
        assert len(packet.stance) >= 1


class TestMemoryCardBackwardCompat:
    def test_memory_card_v1_unchanged_in_registry_gap(self):
        """MemoryCardV1 remains HTTP contract; crystallization is separate schema."""
        with pytest.raises(ValueError):
            resolve("MemoryCardV1")


class TestSalienceAndBus:
    def test_stance_scores_higher_salience(self):
        req = _base_request(kind="stance", planning_effects=["prefer local"], retrieval_affordances=["retrieve_when:test"])
        crys = apply_salience(propose(req))
        assert score_salience(crys) >= 0.8

    def test_bus_lifecycle_kinds_registered(self):
        assert "approved" in LIFECYCLE_KINDS
        assert CHANNEL_DEFAULTS["project"] == "orion:memory:crystallization:project"

    def test_apply_salience_sets_real_confidence_not_stale_default(self):
        # Spec acceptance check: formation call site produces a non-default confidence
        # when evidence supports it, computed on the same pass apply_salience() runs.
        req = _base_request(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c1", strength=0.6),
                CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c2", strength=0.6),
            ]
        )
        crys = apply_salience(propose(req))
        assert crys.confidence == "likely"

    def test_salience_moves_with_evidence_count(self):
        # Spec acceptance check 2: two otherwise-identical crystallizations that differ
        # only in evidence count get different confidence and therefore different
        # salience post-formation. Before this patch this was impossible by
        # construction (confidence was always the stale "likely" default).
        one_source = apply_salience(
            propose(
                _base_request(
                    evidence=[
                        CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c1", strength=0.5)
                    ]
                )
            )
        )
        three_sources = apply_salience(
            propose(
                _base_request(
                    evidence=[
                        CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c1", strength=0.5),
                        CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c2", strength=0.5),
                        CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="c3", strength=0.5),
                    ]
                )
            )
        )
        assert one_source.confidence != three_sources.confidence
        assert one_source.confidence == "possible"
        assert three_sources.confidence == "certain"
        assert score_salience(one_source) != score_salience(three_sources)


def _bare_crystallization(
    *,
    evidence: list[CrystallizationEvidenceRefV1] | None = None,
    reinforcement_count: int = 0,
) -> MemoryCrystallizationV1:
    """Construct a raw MemoryCrystallizationV1 with no formation-pipeline side effects,
    for exercising infer_confidence() directly against arbitrary evidence/reinforcement
    combinations that don't need real duplicate-detection/governor machinery."""
    now = _now()
    return MemoryCrystallizationV1(
        crystallization_id="crys_test",
        kind="semantic",
        subject="subject",
        summary="summary",
        evidence=list(evidence or []),
        dynamics=CrystallizationDynamicsV1(reinforcement_count=reinforcement_count),
        governance=CrystallizationGovernanceV1(proposed_by="test"),
        created_at=now,
        updated_at=now,
    )


class TestInferConfidence:
    def test_no_evidence_no_reinforcement_is_uncertain(self):
        crys = _bare_crystallization(evidence=[], reinforcement_count=0)
        assert infer_confidence(crys) == "uncertain"

    def test_weak_evidence_no_reinforcement_is_uncertain(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.1)
            ],
            reinforcement_count=0,
        )
        assert infer_confidence(crys) == "uncertain"

    def test_multiple_weak_evidence_avg_below_floor_no_reinforcement_is_uncertain(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.1),
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c2", strength=0.2),
            ],
            reinforcement_count=0,
        )
        assert infer_confidence(crys) == "uncertain"

    def test_single_moderate_evidence_no_reinforcement_is_possible(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.5)
            ],
            reinforcement_count=0,
        )
        assert infer_confidence(crys) == "possible"

    def test_two_moderate_evidence_no_reinforcement_is_likely(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.5),
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c2", strength=0.5),
            ],
            reinforcement_count=0,
        )
        assert infer_confidence(crys) == "likely"

    def test_no_evidence_reinforcement_one_is_likely(self):
        crys = _bare_crystallization(evidence=[], reinforcement_count=1)
        assert infer_confidence(crys) == "likely"

    def test_no_evidence_reinforcement_two_is_likely(self):
        crys = _bare_crystallization(evidence=[], reinforcement_count=2)
        assert infer_confidence(crys) == "likely"

    def test_single_evidence_reinforcement_one_is_likely(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.1)
            ],
            reinforcement_count=1,
        )
        assert infer_confidence(crys) == "likely"

    def test_three_evidence_sources_is_certain_regardless_of_reinforcement(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.3),
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c2", strength=0.3),
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c3", strength=0.3),
            ],
            reinforcement_count=0,
        )
        assert infer_confidence(crys) == "certain"

    def test_reinforcement_three_is_certain_regardless_of_evidence(self):
        crys = _bare_crystallization(evidence=[], reinforcement_count=3)
        assert infer_confidence(crys) == "certain"

    def test_two_evidence_and_one_reinforcement_is_certain(self):
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.5),
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c2", strength=0.5),
            ],
            reinforcement_count=1,
        )
        assert infer_confidence(crys) == "certain"

    def test_high_reinforcement_overrides_weak_evidence(self):
        # Recurrence is real evidentiary support even when the original evidence
        # excerpt was weak -- this is the "floor" only applying when reinforcement==0.
        crys = _bare_crystallization(
            evidence=[
                CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="c1", strength=0.05)
            ],
            reinforcement_count=3,
        )
        assert infer_confidence(crys) == "certain"


class TestRdfProjection:
    def test_rdf_hint_conservative(self):
        crys = _active_crystallization()
        hint = build_rdf_projection_hint(crys)
        assert hint.named_graph is not None
        assert "crystallization" in hint.named_graph


class TestGovernorDetection:
    def test_detect_duplicates(self):
        a = _active_crystallization(kind="stance", planning_effects=["x"], retrieval_affordances=["y"])
        b = a.model_copy(deep=True)
        b.crystallization_id = "crys_other"
        b.summary = a.summary
        result = detect_duplicates(b, [a])
        assert result.duplicates

    def test_detect_contradictions_negation(self):
        pos = _active_crystallization(summary="Use local-first memory governance for canonical loops")
        neg = pos.model_copy(deep=True)
        neg.crystallization_id = "crys_neg"
        neg.summary = "Do not use local-first memory governance for canonical loops"
        result = detect_contradictions(neg, [pos])
        assert result.contradictions or result.warnings


class TestActivePacketFusion:
    def test_active_packet_includes_card_rail(self):
        crys = _active_crystallization()
        packet = build_active_packet(
            query="test",
            crystallizations=[crys],
            active_cards=[{"card_id": "card-1", "summary": "card fact"}],
            task_type="architecture",
        )
        assert "card-1" in packet.card_refs
        assert packet.retrieval_trace.get("rails") == ["postgres_crystallizations", "postgres_memory_cards"]


@pytest.mark.asyncio
async def test_retriever_collects_linked_crystallization_from_graphiti_depth_two():
    from unittest.mock import MagicMock
    from orion.memory.crystallization.retriever import retrieve_active_packet

    crys = _active_crystallization()
    adapter = MagicMock()
    adapter.enabled = True
    adapter.health.return_value = {"backend": "orion_postgres"}
    adapter.neighborhood.return_value = {
        "enabled": True,
        "nodes": [
            {"crystallization_id": crys.crystallization_id},
            {"crystallization_id": "crys_linked"},
        ],
        "edges": [],
    }

    packet = await retrieve_active_packet(
        query="test",
        crystallizations=[crys],
        graphiti_adapter=adapter,
        seed_crystallization_id=crys.crystallization_id,
    )
    adapter.neighborhood.assert_called_once_with(crys.crystallization_id, depth=2)
    assert "crys_linked" in packet.graphiti_refs


@pytest.mark.asyncio
async def test_retriever_uses_graphiti_search_when_backend_is_graphiti_core():
    from unittest.mock import MagicMock
    from orion.memory.crystallization.retriever import retrieve_active_packet

    crys = _active_crystallization()
    adapter = MagicMock()
    adapter.enabled = True
    adapter.health.return_value = {"backend": "graphiti_core"}
    adapter.search.return_value = {
        "crystallization_ids": [crys.crystallization_id, "crys_search_hit"],
        "trace": {"backend": "graphiti_core"},
    }

    packet = await retrieve_active_packet(
        query="stance on memory",
        crystallizations=[crys],
        graphiti_adapter=adapter,
        seed_crystallization_id=crys.crystallization_id,
    )
    adapter.search.assert_called_once_with(
        "stance on memory",
        seed_crystallization_id=crys.crystallization_id,
    )
    adapter.neighborhood.assert_called_once_with(crys.crystallization_id, depth=2)
    assert "crys_search_hit" in packet.graphiti_refs
    rails = packet.retrieval_trace.get("rails", [])
    assert "graphiti_search" in rails
    assert "graphiti_neighborhood" in rails


class TestFetchSimilarCandidates:
    @pytest.mark.asyncio
    async def test_no_embed_host_url_short_circuits(self):
        from unittest.mock import MagicMock, patch

        from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates

        candidate = _active_crystallization()
        pool = MagicMock()

        with patch(
            "orion.memory.crystallization.candidate_retrieval._embed_query"
        ) as mock_embed, patch(
            "orion.memory.crystallization.candidate_retrieval.query_chroma_collection"
        ) as mock_chroma, patch(
            "orion.memory.crystallization.candidate_retrieval.get_crystallization"
        ) as mock_get:
            result = await fetch_similar_candidates(
                candidate,
                pool=pool,
                embed_host_url="",
                chroma_host="chroma.local",
            )

        assert result == []
        mock_embed.assert_not_called()
        mock_chroma.assert_not_called()
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_excludes_self_and_inactive(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates

        candidate = _active_crystallization()
        pool = MagicMock()

        active_neighbor = _active_crystallization()
        active_neighbor.crystallization_id = "crys_active_neighbor"

        rejected_neighbor = _active_crystallization()
        rejected_neighbor.crystallization_id = "crys_rejected_neighbor"
        rejected_neighbor.status = "rejected"

        hits = [
            {"doc_id": "doc-1", "metadata": {"crystallization_id": "crys_active_neighbor"}},
            {"doc_id": "doc-2", "metadata": {"crystallization_id": candidate.crystallization_id}},
            {"doc_id": "doc-3", "metadata": {"crystallization_id": "crys_rejected_neighbor"}},
        ]

        async def _fake_get_crystallization(_pool, cid):
            if cid == "crys_active_neighbor":
                return active_neighbor
            if cid == "crys_rejected_neighbor":
                return None
            raise AssertionError(f"unexpected crystallization_id {cid}")

        with patch(
            "orion.memory.crystallization.candidate_retrieval._embed_query",
            new=AsyncMock(return_value=[0.1, 0.2, 0.3]),
        ), patch(
            "orion.memory.crystallization.candidate_retrieval.query_chroma_collection",
            new=MagicMock(return_value=hits),
        ), patch(
            "orion.memory.crystallization.candidate_retrieval.get_crystallization",
            new=AsyncMock(side_effect=_fake_get_crystallization),
        ):
            result = await fetch_similar_candidates(
                candidate,
                pool=pool,
                embed_host_url="http://embed.local",
                chroma_host="chroma.local",
            )

        assert len(result) == 1
        assert result[0].crystallization_id == "crys_active_neighbor"

    @pytest.mark.asyncio
    async def test_limit_caps_returned_candidates(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates

        candidate = _active_crystallization()
        pool = MagicMock()

        neighbor_ids = ["crys_a", "crys_b", "crys_c", "crys_d"]
        neighbors = {}
        for nid in neighbor_ids:
            n = _active_crystallization()
            n.crystallization_id = nid
            neighbors[nid] = n

        hits = [{"doc_id": nid, "metadata": {"crystallization_id": nid}} for nid in neighbor_ids]

        async def _fake_get_crystallization(_pool, cid):
            return neighbors[cid]

        with patch(
            "orion.memory.crystallization.candidate_retrieval._embed_query",
            new=AsyncMock(return_value=[0.1, 0.2, 0.3]),
        ), patch(
            "orion.memory.crystallization.candidate_retrieval.query_chroma_collection",
            new=MagicMock(return_value=hits),
        ), patch(
            "orion.memory.crystallization.candidate_retrieval.get_crystallization",
            new=AsyncMock(side_effect=_fake_get_crystallization),
        ):
            result = await fetch_similar_candidates(
                candidate,
                pool=pool,
                embed_host_url="http://embed.local",
                chroma_host="chroma.local",
                limit=2,
            )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_failure_degrades_to_empty_list(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates

        candidate = _active_crystallization()
        pool = MagicMock()

        with patch(
            "orion.memory.crystallization.candidate_retrieval._embed_query",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch(
            "orion.memory.crystallization.candidate_retrieval.query_chroma_collection"
        ) as mock_chroma:
            result = await fetch_similar_candidates(
                candidate,
                pool=pool,
                embed_host_url="http://embed.local",
                chroma_host="chroma.local",
            )

        assert result == []
        mock_chroma.assert_not_called()

    @pytest.mark.asyncio
    async def test_limit_zero_returns_empty_not_one(self):
        # Regression: CONCEPT_RELATION_CANDIDATE_LIMIT=0 is documented (concept_relation.py)
        # as a legitimate "throttle to zero" operator setting. n_results=max(1, int(limit))
        # used to floor the Chroma query at 1 result regardless of limit, so limit=0 still
        # returned a single candidate instead of [].
        from unittest.mock import AsyncMock, MagicMock, patch

        from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates

        candidate = _active_crystallization()
        pool = MagicMock()

        with patch(
            "orion.memory.crystallization.candidate_retrieval._embed_query",
            new=AsyncMock(return_value=[0.1, 0.2, 0.3]),
        ) as mock_embed, patch(
            "orion.memory.crystallization.candidate_retrieval.query_chroma_collection"
        ) as mock_chroma:
            result = await fetch_similar_candidates(
                candidate,
                pool=pool,
                embed_host_url="http://embed.local",
                chroma_host="chroma.local",
                limit=0,
            )

        assert result == []
        mock_embed.assert_not_called()
        mock_chroma.assert_not_called()
