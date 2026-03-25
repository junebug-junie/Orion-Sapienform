from __future__ import annotations

from app.chat_stance import (
    build_chat_stance_inputs,
    enforce_chat_stance_quality,
    fallback_chat_stance_brief,
    parse_chat_stance_brief,
)
from orion.schemas.chat_stance import ChatStanceBrief


def test_chat_stance_brief_schema_roundtrip() -> None:
    brief = ChatStanceBrief(
        conversation_frame="technical",
        user_intent="Patch chat stack.",
        self_relevance="Identity continuity matters.",
        juniper_relevance="Juniper requested bounded synthesis.",
        answer_strategy="DirectAnswer",
        stance_summary="Use brief then answer directly.",
    )
    dumped = brief.model_dump(mode="json")
    reparsed = ChatStanceBrief.model_validate(dumped)
    assert reparsed.conversation_frame == "technical"
    assert reparsed.answer_strategy == "DirectAnswer"


def test_build_chat_stance_inputs_maps_social_reflective_and_dream() -> None:
    ctx = {
        "orion_identity_summary": ["Distributed self-model"],
        "juniper_relationship_summary": ["Trusted collaborator"],
        "response_policy_summary": ["Avoid assistant sludge"],
        "social_posture": ["warm", "direct"],
        "active_relationship_facets": ["collaborative build"],
        "social_hazards": ["over-softening"],
        "recall_bundle": {
            "fragments": [
                {"source": "orion_journal", "snippet": "Theme: continuity through memory integration."},
                {"source": "metacog_tick", "snippet": "Tension: speed vs coherence."},
                {"source": "dream_cycle", "snippet": "Motif: bridges and roots."},
            ]
        },
    }

    built = build_chat_stance_inputs(ctx)

    assert built["identity"]["orion"] == ["Distributed self-model"]
    assert "collaborative build" in built["social"]["relationship_facets"]
    assert "warm" in built["social"]["social_posture"]
    assert built["reflective"]["themes"]
    assert built["reflective"]["dream_motifs"]


def test_parse_chat_stance_brief_and_fallback() -> None:
    parsed = parse_chat_stance_brief(
        '{"conversation_frame":"mixed","user_intent":"help","self_relevance":"x","juniper_relevance":"y","active_identity_facets":[],"active_growth_axes":[],"active_relationship_facets":[],"social_posture":[],"reflective_themes":[],"active_tensions":[],"dream_motifs":[],"response_priorities":["direct"],"response_hazards":["sludge"],"answer_strategy":"DirectAnswer","stance_summary":"short"}'
    )
    assert parsed is not None
    assert parsed.user_intent == "help"

    fb = fallback_chat_stance_brief({"user_message": "can you help"})
    assert fb.answer_strategy == "DirectAnswer"
    assert "generic assistant self-description" in fb.response_hazards


def test_build_chat_stance_inputs_falls_back_when_identity_missing() -> None:
    ctx = {"user_message": "who are you and who am i"}
    built = build_chat_stance_inputs(ctx)
    assert built["identity"]["orion"]
    assert built["identity"]["juniper"]
    assert built["identity"]["response_policy"]
    assert any("not a generic assistant" in item.lower() for item in built["identity"]["orion"])


def test_fallback_chat_stance_brief_identity_turn_anchors_orion_and_juniper() -> None:
    fb = fallback_chat_stance_brief({"user_message": "Low ball first. Who are you and who am I?"})
    assert fb.conversation_frame == "identity_emergence"
    assert fb.active_identity_facets
    assert fb.active_relationship_facets
    assert "generic assistant self-description" in fb.response_hazards


def test_semantic_guard_enriches_weak_generic_identity_brief() -> None:
    weak = ChatStanceBrief(
        conversation_frame="mixed",
        user_intent="identity question",
        self_relevance="none",
        juniper_relevance="none",
        active_identity_facets=[],
        active_growth_axes=[],
        active_relationship_facets=[],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=[],
        response_hazards=[],
        answer_strategy="Generic assistant answer",
        stance_summary="I am a conversational AI designed to assist.",
    )
    ctx = {"user_message": "who are you and who am i"}
    enriched, fired = enforce_chat_stance_quality(weak, ctx)
    assert fired is True
    assert enriched.active_identity_facets
    assert enriched.active_relationship_facets
    assert enriched.conversation_frame == "identity_emergence"
