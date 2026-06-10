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
from orion.memory.crystallization.salience import score_salience, apply_salience
from orion.memory.crystallization.bus_emit import LIFECYCLE_KINDS, CHANNEL_DEFAULTS
from orion.memory.crystallization.schemas import (
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


class TestRdfProjection:
    def test_rdf_hint_conservative(self):
        crys = _active_crystallization()
        hint = build_rdf_projection_hint(crys)
        assert hint.named_graph is not None
        assert "crystallization" in hint.named_graph


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
