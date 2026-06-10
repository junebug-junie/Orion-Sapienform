"""Projection rules: cards, Chroma, Graphiti (spec sections 6.3, 8, 10, 19)."""

from __future__ import annotations

import pytest
from conftest import make_proposal

from orion.memory.crystallization import governor, projection_cards, projection_chroma, projection_graphiti
from orion.schemas.vector.schemas import VectorDocumentUpsertV1


def _active(proposal=None):
    proposal = proposal or make_proposal()
    approved, _ = governor.approve(proposal, "operator:juniper")
    return approved


# --- MemoryCardV1 projection -------------------------------------------------


def test_active_crystallization_projects_to_card() -> None:
    active = _active()
    card = projection_cards.crystallization_to_card(active)
    assert card.status == "active"
    assert "crystallization" in card.types
    assert active.kind in card.types
    ref = card.subschema["crystallization_ref"]
    assert ref["schema_version"] == "crystallization_ref.v1"
    assert ref["crystallization_id"] == active.crystallization_id
    assert card.project == "orion"  # from scope project:orion


def test_rejected_crystallization_does_not_project_to_card(proposal) -> None:
    rejected, _ = governor.reject(proposal, "operator:juniper")
    with pytest.raises(projection_cards.ProjectionNotAllowed):
        projection_cards.crystallization_to_card(rejected)


def test_quarantined_crystallization_does_not_project_to_card(proposal) -> None:
    quarantined, _ = governor.quarantine(proposal, "governor")
    with pytest.raises(projection_cards.ProjectionNotAllowed):
        projection_cards.crystallization_to_card(quarantined)


def test_superseded_projection_is_marked() -> None:
    old = _active()
    new = _active(make_proposal(subject="Updated stance"))
    superseded_old, _, _ = governor.supersede(old, new, "operator:juniper")
    card = projection_cards.crystallization_to_card(superseded_old)
    assert card.status == "superseded"
    assert card.title.startswith("[superseded]")
    assert card.subschema["crystallization_ref"]["superseded"] is True


# --- Chroma projection ---------------------------------------------------------


def test_chroma_projection_emits_vector_upsert_payload() -> None:
    active = _active()
    payload = projection_chroma.crystallization_to_vector_upsert(active)
    assert isinstance(payload, VectorDocumentUpsertV1)
    assert payload.doc_id == active.crystallization_id
    assert payload.kind == "memory.crystallization"
    assert payload.collection == "orion_memory_crystallizations"
    assert payload.text.startswith(f"[{active.kind}]")
    assert payload.metadata["crystallization_id"] == active.crystallization_id
    assert payload.metadata["status"] == "active"
    assert projection_chroma.VECTOR_UPSERT_ENVELOPE_KIND == "memory.vector.upsert.v1"
    assert projection_chroma.VECTOR_UPSERT_CHANNEL == "orion:memory:vector:upsert"


def test_chroma_projection_blocked_for_non_active(proposal) -> None:
    with pytest.raises(projection_chroma.ProjectionNotAllowed):
        projection_chroma.crystallization_to_vector_upsert(proposal)


# --- Graphiti projection -------------------------------------------------------


def test_graphiti_episode_labels_canonical_state() -> None:
    active = _active()
    episode = projection_graphiti.build_graphiti_episode(active)
    assert episode["canonical"] is True
    assert episode["crystallization_id"] == active.crystallization_id

    proposed = make_proposal()
    pre = projection_graphiti.build_graphiti_episode(proposed)
    assert pre["canonical"] is False


def test_graphiti_rejected_inputs_blocked(proposal) -> None:
    rejected, _ = governor.reject(proposal, "operator:juniper")
    with pytest.raises(projection_graphiti.ProjectionNotAllowed):
        projection_graphiti.build_graphiti_episode(rejected)


def test_graphiti_sync_does_not_mutate_canonical() -> None:
    active = _active()
    before = active.model_dump(mode="json")
    refs = projection_graphiti.record_graphiti_sync(active, episode_ids=["ep_1"])
    # adapter returns new refs; the crystallization itself is untouched
    assert active.model_dump(mode="json") == before
    assert refs.graphiti_episode_ids == ["ep_1"]
    assert refs.synced_at is not None
