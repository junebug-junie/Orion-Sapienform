"""Active packet assembly and proposer helpers."""

from __future__ import annotations

from conftest import make_proposal

from orion.memory.crystallization import governor
from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.proposer import (
    build_proposal,
    evidence_from_cards,
    evidence_from_grammar_events,
)
from orion.memory.crystallization.validator import validate_proposal
from orion.schemas.memory_crystallization import ActiveMemoryPacketV1


def _active(**overrides):
    approved, _ = governor.approve(make_proposal(**overrides), "operator:juniper")
    return approved


def test_packet_groups_by_kind_and_records_refs() -> None:
    stance = _active()
    open_loop = _active(kind="open_loop", subject="Validate Graphiti", planning_effects=[])
    contradiction = _active(
        kind="contradiction",
        subject="Frontier boundary tension",
        planning_effects=[],
        links=[
            {"target_crystallization_id": "crys_a", "relation": "contradicts"},
            {"target_crystallization_id": "crys_b", "relation": "contradicts"},
        ],
    )
    packet = build_active_packet(
        query="memory architecture",
        crystallizations=[stance, open_loop, contradiction],
    )
    assert isinstance(packet, ActiveMemoryPacketV1)
    assert len(packet.stance) == 1
    assert packet.stance[0]["planning_effects"] == stance.planning_effects
    assert len(packet.open_loops) == 1
    assert len(packet.contradictions) == 1
    assert set(packet.crystallization_refs) == {
        stance.crystallization_id,
        open_loop.crystallization_id,
        contradiction.crystallization_id,
    }
    assert packet.retrieval_trace["included_count"] == 3


def test_packet_excludes_non_active_and_traces_it() -> None:
    stance = _active()
    proposal = make_proposal(subject="Not yet governed")
    packet = build_active_packet(query="q", crystallizations=[stance, proposal])
    assert packet.crystallization_refs == [stance.crystallization_id]
    excluded = packet.retrieval_trace["excluded"]
    assert excluded[0]["crystallization_id"] == proposal.crystallization_id
    assert excluded[0]["reason"] == "status:proposed"


def test_proposer_builds_valid_card_sourced_proposal() -> None:
    evidence = evidence_from_cards(
        [{"card_id": "card-1", "summary": "operator highlighted decision"}]
    )
    proposal = build_proposal(
        kind="decision",
        subject="Use MemoryCrystallizationV1 as separate schema",
        summary="Crystallizations are a new artifact, not MemoryCardV2.",
        proposed_by="local-model:proposer",
        scope=["project:orion"],
        evidence=evidence,
        planning_effects=["never rename MemoryCardV1"],
    )
    assert proposal.status == "proposed"
    assert proposal.governance.validation_status == "unvalidated"
    assert proposal.source_card_ids == ["card-1"]
    assert validate_proposal(proposal) == []


def test_proposer_grammar_event_evidence() -> None:
    evidence = evidence_from_grammar_events(["gev_001", "gev_002"])
    assert [e.source_id for e in evidence] == ["gev_001", "gev_002"]
    assert all(e.source_kind == "grammar_event" for e in evidence)
